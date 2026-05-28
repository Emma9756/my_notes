---
title: "fp4_logic_extraction"
date: 2026-05-28
tags:
  - #LLM
  - #fp4
  - #deepseek
  - #笔记
  - #from_me
  - #待整理
status: 待整理
---

# FP4 Logic Extraction From `inference/model.py`

本文直接把关键代码节选嵌进来，避免来回跳转，重点解释：

- FP4 在哪里用
- 量化 / 反量化怎么做
- shape 怎么变化
- `model.py` 依赖到的主要函数做了什么

## 1. 结论先看

这个仓库里的 FP4 不是“全模型统一 FP4 推理”，而是两类用途：

1. `MoE Expert` 权重真的存成 FP4，推理时走 `FP8 activation x FP4 weight` 的 GEMM。
2. `Indexer` / `Compressor(rotate=True)` 对激活做 `fp4_act_quant(..., True)`，这是“量化后再反量化”的模拟路径，张量最后仍是 `bf16`。

`model.py` 里的几个关键入口如下：

```python
def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert bias is None

    if weight.dtype == torch.float4_e2m1fn_x2:
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp4_gemm(x, s, weight, weight.scale, scale_dtype)
    elif weight.dtype == torch.float8_e4m3fn:
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp8_gemm(x, s, weight, weight.scale, scale_dtype)
    else:
        return F.linear(x, weight)
```

```python
if self.rotate:
    kv = rotate_activation(kv)
    fp4_act_quant(kv, fp4_block_size, True)
else:
    act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
```

```python
q = self.wq_b(qr)
q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
apply_rotary_emb(q[..., -rd:], freqs_cis)
q = rotate_activation(q)
fp4_act_quant(q, fp4_block_size, True)
```

```python
expert_dtype = torch.float4_e2m1fn_x2 if args.expert_dtype == "fp4" else None
self.experts = nn.ModuleList([
    Expert(args.dim, args.moe_inter_dim, dtype=expert_dtype, swiglu_limit=args.swiglu_limit)
    if self.experts_start_idx <= i < self.experts_end_idx else None
    for i in range(self.n_routed_experts)
])
```

配置里默认就启用了 expert FP4：

```json
{
  "expert_dtype": "fp4",
  "torch_dtype": "bfloat16"
}
```

```json
{
  "dtype": "fp8",
  "scale_fmt": "ue8m0",
  "expert_dtype": "fp4"
}
```

## 2. `model.py` 里的 FP4 使用场景

### 2.1 MoE Expert 权重压成 FP4

`Linear.__init__()` 里对 FP4 权重的定义如下：

```python
if dtype == torch.float4_e2m1fn_x2:
    # FP4: weight is [out, in//2] in float4_e2m1fn_x2, logically [out, in] in fp4
    # Scale is [out, in//32] in float8_e8m0fnu (1 scale per 32 fp4 elements along K)
    self.weight = nn.Parameter(torch.empty(out_features, in_features // 2, dtype=torch.float4_e2m1fn_x2))
    scale_out_features = out_features
    scale_in_features = in_features // fp4_block_size
    self.weight.scale = self.scale = nn.Parameter(
        torch.empty(scale_out_features, scale_in_features, dtype=torch.float8_e8m0fnu)
    )
```

这里直接说明了几件事：

- 逻辑权重 shape 是 `[out_features, in_features]`
- 物理存储 shape 是 `[out_features, in_features // 2]`
- 最后一维除以 2，是因为 2 个 FP4 打包到 1 个存储单元
- scale shape 是 `[out_features, in_features // 32]`
- 也就是沿 K 维每 32 个元素共享 1 个 scale

`MoE` 里 routed experts 用 FP4，shared expert 不用：

```python
expert_dtype = torch.float4_e2m1fn_x2 if args.expert_dtype == "fp4" else None
self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim, dtype=expert_dtype, swiglu_limit=args.swiglu_limit) if self.experts_start_idx <= i < self.experts_end_idx else None
                               for i in range(self.n_routed_experts)])
assert args.n_shared_experts == 1
self.shared_experts = Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)
```

### 2.2 Indexer / Compressor 中的 FP4 激活模拟

这部分不是“真的把激活保存成 FP4”，而是做一遍 FP4 量化误差模拟。

`Compressor` 的 `rotate=True` 分支：

```python
kv = self.norm(kv.to(dtype))
apply_rotary_emb(kv[..., -rd:], freqs_cis)
if self.rotate:
    kv = rotate_activation(kv)
    fp4_act_quant(kv, fp4_block_size, True)
else:
    act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
```

`Indexer` 里对 `q` 也这么处理：

```python
q = self.wq_b(qr)
q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
apply_rotary_emb(q[..., -rd:], freqs_cis)
q = rotate_activation(q)
# use fp4 simulation for q and kv in indexer
fp4_act_quant(q, fp4_block_size, True)
```

`rotate_activation()` 本身就是 Hadamard 旋转：

```python
def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    from fast_hadamard_transform import hadamard_transform
    return hadamard_transform(x, scale=x.size(-1) ** -0.5)
```

这一步的意图是把能量摊匀，减轻低比特量化对少数大值维度的敏感性。

## 3. FP4 权重线性层的主路径

### 3.1 分发入口：`linear()`

主路径代码如下：

```python
def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert bias is None

    if weight.dtype == torch.float4_e2m1fn_x2:
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp4_gemm(x, s, weight, weight.scale, scale_dtype)
    elif weight.dtype == torch.float8_e4m3fn:
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp8_gemm(x, s, weight, weight.scale, scale_dtype)
    else:
        return F.linear(x, weight)
```

因此 expert 的真实计算不是：

```text
BF16 activation x FP4 weight
```

而是：

```text
(FP8 activation + act_scale) x (FP4 weight + weight_scale)
```

也就是：

1. 输入激活先按 128 分组量化成 FP8
2. 权重本身是按 32 分组保存的 FP4
3. `fp4_gemm` 内部边 GEMM 边把 scale 乘回来

### 3.2 为什么这里不是 `fp4_act_quant(x)`？

因为底层 kernel 明确就是 `FP8 act x FP4 weight` 设计：

```python
def fp4_gemm_kernel(N, K, out_dtype=BF16, accum_dtype=FP32, scale_dtype=FP32):
    """FP8 act x FP4 weight GEMM kernel.
    C[M, N] = A_fp8[M, K] @ B_fp4[N, K]^T
    Act: 1x128 quant on K (reduce dim), FP8 with configurable scale dtype
    Weight: 1x32 quant on K (reduce dim), FP4 with E8M0 scale
    B is stored as [N, K//2] in float4_e2m1fn_x2, logical [N, K] in fp4.
    """
```

这说明：

- 激活走 FP8，更稳
- 权重走 FP4，更省
- 这是一个混合精度设计，不是“全部都压到 FP4”

## 4. 量化细节：`fp4_act_quant`

### 4.1 对应实现

`fp4_act_quant` 的包装代码：

```python
def fp4_act_quant(
    x: torch.Tensor, block_size: int = 32, inplace: bool = False,
) -> torch.Tensor:
    """Block-wise FP4 quantization. inplace=True does fused quant+dequant back to BF16."""
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous()
    y = torch.empty_like(z) if inplace else z.new_empty(*z.shape[:-1], N // 2, dtype=torch.float4_e2m1fn_x2)
    s = z.new_empty(*z.size()[:-1], N // block_size, dtype=torch.float8_e8m0fnu)
    kernel = fp4_quant_kernel(N, block_size, inplace=inplace)
    kernel(z.view(-1, N), y.view(-1, y.size(-1)), s.view(-1, N // block_size))
    if inplace:
        x.copy_(y)
        return x
    return y, s
```

从这段就能直接看出：

- 最后一维必须能被 32 整除
- 非 `inplace` 时，输出数据最后一维变成 `N // 2`
- scale 最后一维是 `N // 32`
- `inplace=True` 时最终还是写回原 tensor

### 4.2 分组方式和 scale 计算

底层 kernel 的关键逻辑：

```python
fp4_max = 6.0
fp4_max_inv = 1.0 / fp4_max

T.reduce_absmax(x_local, amax_local, dim=1)
for i in T.Parallel(blk_m):
    amax_local[i] = T.max(amax_local[i], 6 * (2**-126))
    s_local[i] = fast_round_scale(amax_local[i], fp4_max_inv)
```

`fast_round_scale` 定义为：

```python
def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))
```

所以一个 block 的 scale 实际是：

```text
s = 2^ceil(log2(amax / 6))
```

不是普通的 `amax / 6`，而是向上取到 2 的幂。这就是 power-of-two scale。

### 4.3 量化公式

量化本体在 kernel 里是：

```python
y_local[i, j] = T.clamp(
    x_local[i, j] / s_local[i], -fp4_max, fp4_max
)
```

再 cast 到 FP4：

```python
T.Cast(FP4, T.clamp(
    x_local[i, j] / s_local[i], -fp4_max, fp4_max
))
```

数学上可写成：

```text
q = clip(x / s, -6, 6)
q_fp4 = cast_to_fp4(q)
```

### 4.4 FP4 到底能表示哪些值

`convert.py` 里给了明表：

```python
FP4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)
```

所以 FP4 的离散值集合就是：

```text
{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
```

### 4.5 反量化公式

`inplace=True` 的路径如下：

```python
y_local[i, j] = T.Cast(
    out_dtype,
    T.Cast(compute_dtype, T.Cast(FP4, T.clamp(
        x_local[i, j] / s_local[i], -fp4_max, fp4_max
    ))) * s_local[i],
)
```

也就是先量化成 FP4，再乘 scale 写回：

```text
x_hat = dequant(q_fp4) * s
```

如果写成一步：

```text
x_hat = float(cast_to_fp4(clip(x / s, -6, 6))) * s
```

所以 `fp4_act_quant(..., True)` 的精确定义是：

- 不是返回压缩后的 FP4 存储
- 而是把原 tensor 替换成“经过 FP4 限制后的 BF16 近似值”

## 5. GEMM 细节：`fp4_gemm`

### 5.1 输入张量语义

包装函数：

```python
def fp4_gemm(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T.
    A has per-128 act scale; B has per-32 E8M0 weight scale.
    B is stored as [N, K//2] in float4_e2m1fn_x2 (2 FP4 values per byte, packed along K)."""
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    kernel = fp4_gemm_kernel(N, K, scale_dtype=tl_dtype)
    kernel(a.view(M, K), b, c.view(M, N), a_s.view(M, -1), b_s)
    return c
```

这里可以直接读出：

- `a` 是 `[M, K]` 的 FP8 激活
- `a_s` 是 `[M, K/128]`
- `b` 物理上是 `[N, K/2]`，逻辑上代表 `[N, K]`
- `b_s` 是 `[N, K/32]`
- 输出 `c` 是 `[M, N]`

如果上层输入是 `[B, S, K]`，这里会自动变成：

```text
[B, S, K] -> [B*S, K] -> [B*S, N] -> [B, S, N]
```

### 5.2 kernel 里到底怎么乘

核心循环如下：

```python
act_group_size = 128
weight_group_size = 32
block_K = 32
n_sub = act_group_size // block_K  # 4

for k in T.Pipelined(K_iters, num_stages=2):
    T.copy(A[by * block_M, k * block_K], A_shared)
    T.copy(B[bx * block_N, k * block_K], B_fp4_shared)

    for i, j in T.Parallel(block_N, block_K):
        B_shared[i, j] = T.Cast(FP8, T.Cast(FP32, B_fp4_shared[i, j]))

    for i in T.Parallel(block_N):
        scale_b_frag[i] = T.Cast(FP32, scales_b[bx * block_N + i, k])

    for i in T.Parallel(block_M):
        scale_a_frag[i] = T.Cast(FP32, scales_a[by * block_M + i, k // n_sub])

    T.gemm(A_shared, B_shared, C_local, transpose_B=True)

    for i, j in T.Parallel(block_M, block_N):
        C_local_accum[i, j] += C_local[i, j] * scale_a_frag[i] * scale_b_frag[j]
```

这段代码说明：

1. 激活 scale 粒度是 128
2. 权重 scale 粒度是 32
3. 每轮 K 方向只推进 32
4. 因此一个 activation scale 要覆盖 4 个 `block_K`
5. 先做低精度 GEMM，再把本块对应的 `scale_a * scale_b` 乘回局部结果

### 5.3 为什么 activation scale 用 `k // 4`

因为：

```text
128 / 32 = 4
```

代码就是：

```python
n_sub = act_group_size // block_K  # 4
scale_a_frag[i] = T.Cast(FP32, scales_a[by * block_M + i, k // n_sub])
scale_b_frag[i] = T.Cast(FP32, scales_b[bx * block_N + i, k])
```

这表示：

- activation 的一个 scale 覆盖 4 个子块
- weight 的一个 scale 只覆盖 1 个子块

### 5.4 数学上等价于什么

对输出元素 `c[m, n]`，可以把它理解为：

```text
c[m, n] =
sum_subblocks(
    sum_j(a_q[m, j] * b_q[n, j]) * s_a[m, subblock//4] * s_b[n, subblock]
)
```

也就是说这里不是“先完整反量化再做 GEMM”，而是：

- 先对量化值做 GEMM
- 然后在每个 K 子块上乘回对应 scale
- 最后把所有子块累加起来

这是一种融合反量化的块级 GEMM。

## 6. Shape 变化梳理

### 6.1 Expert 线性层

假设 expert 线性层输入：

```text
x: [T, K]
weight(logical): [N, K]
```

其中 `T` 是路由到这个 expert 的 token 数。

进入 FP4 路径后：

1. `act_quant(x, 128)`
   - `x_q`: `[T, K]`
   - `x_scale`: `[T, K/128]`
2. `weight`
   - 物理存储 `[N, K/2]`
   - `weight.scale`: `[N, K/32]`
3. `fp4_gemm(...)`
   - 输出 `[T, N]`

高维输入 `[B, S, K]` 时，本质就是：

```text
[B, S, K] -> [B*S, K] -> GEMM -> [B*S, N] -> [B, S, N]
```

### 6.2 `fp4_act_quant(..., inplace=True)` 的 shape

如果输入是：

```text
q: [B, S, H, D]
```

调用：

```python
fp4_act_quant(q, 32, True)
```

那么：

- shape 不变，仍是 `[B, S, H, D]`
- dtype 不变，通常仍是 `bf16`
- 变化的是数值，变成 FP4 量化再反量化后的近似值

如果 `inplace=False` 才会出现真正压缩：

- 数据 shape：`[B, S, H, D/2]`
- scale shape：`[B, S, H, D/32]`

### 6.3 `Compressor` 中的 `kv` shape

`Compressor` 压缩后，在量化前 `kv` 大致是：

- prefill: `[B, S/ratio, head_dim]`
- decode 触发压缩时: `[B, 1, head_dim]`

对应代码：

```python
if not should_compress:
    return
kv = self.norm(kv.to(dtype))
...
if self.rotate:
    kv = rotate_activation(kv)
    fp4_act_quant(kv, fp4_block_size, True)
...
if start_pos == 0:
    self.kv_cache[:bsz, :seqlen // ratio] = kv
else:
    self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
```

因此：

- 旋转前后 shape 不变
- `fp4_act_quant(..., True)` 前后 shape 也不变
- 最后仍按 BF16 形式写进 cache

### 6.4 `Indexer` 中的 `q` shape

`Indexer.forward()` 的主要 shape 流：

```python
q = self.wq_b(qr)
q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
...
index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
```

因此：

1. `q = self.wq_b(qr)` 后是 `[B, S, n_local_heads * head_dim]`
2. `unflatten` 后是 `[B, S, n_local_heads, head_dim]`
3. `fp4_act_quant(q, 32, True)` 后 shape 不变
4. `einsum` 后变成：

```text
index_score: [B, S, H, T_compressed]
```

## 7. 相关文件中的主要函数分析

### 7.1 `act_quant`

代码：

```python
def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32, inplace: bool = False,
) -> torch.Tensor:
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous()
    y = torch.empty_like(z) if inplace else torch.empty_like(z, dtype=torch.float8_e4m3fn)
    s = z.new_empty(*z.size()[:-1], N // block_size, dtype=scale_dtype)
    ...
    if inplace:
        x.copy_(y)
        return x
    return y, s
```

作用：

- 按 128 分组把激活量化到 FP8
- 非 `inplace` 返回 `(fp8_tensor, scale)`
- FP4 expert 路径先用它把激活降成 FP8

### 7.2 `fp4_act_quant`

代码：

```python
def fp4_act_quant(
    x: torch.Tensor, block_size: int = 32, inplace: bool = False,
) -> torch.Tensor:
    N = x.size(-1)
    assert N % block_size == 0
    ...
    y = torch.empty_like(z) if inplace else z.new_empty(*z.shape[:-1], N // 2, dtype=torch.float4_e2m1fn_x2)
    s = z.new_empty(*z.size()[:-1], N // block_size, dtype=torch.float8_e8m0fnu)
```

作用：

- 按 32 分组把激活量化到 FP4
- `Indexer` 和 `Compressor(rotate=True)` 用它做误差模拟
- 不是 expert 主 GEMM 的输入量化函数

### 7.3 `fp4_gemm`

代码：

```python
def fp4_gemm(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    kernel(a.view(M, K), b, c.view(M, N), a_s.view(M, -1), b_s)
    return c
```

作用：

- 执行 `FP8 activation x FP4 weight`
- 内部融合 activation / weight 的 scale
- 这是 `model.py -> linear() -> expert` 的核心低精度算子

### 7.4 `FP4_TABLE`

代码：

```python
FP4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)
```

作用：

- 明确列出 `e2m1fn` 的离散值集合
- 这就是 FP4 真正能表达的值

### 7.5 `cast_e2m1fn_to_e4m3fn`

关键代码：

```python
x = x.view(torch.uint8)
low  = x & 0x0F
high = (x >> 4) & 0x0F
x = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(2)
```

```python
scale = scale.float().view(bOut, fp8_block_size, bIn, -1).transpose(1, 2).flatten(2)
scale_max_offset_bits = scale.amax(dim=-1, keepdim=True) / (2**MAX_OFFSET_BITS)
offset = scale / scale_max_offset_bits
offset = offset.unflatten(-1, (fp8_block_size, -1)).repeat_interleave(fp4_block_size, dim=-1)
x = (x * offset).transpose(1, 2).reshape(out_dim, in_dim)
return x.to(torch.float8_e4m3fn), scale_max_offset_bits.squeeze(-1).to(torch.float8_e8m0fnu)
```

作用：

1. 把 packed 的 FP4 权重拆成两个 nibble
2. 用 `FP4_TABLE` 解码成逻辑数值
3. 结合原来的 FP4 scale
4. 重新组织成 FP8 兼容布局

它不参与在线推理主路径，但非常适合拿来理解 checkpoint 里的 FP4 权重到底怎么编码。

## 8. 最关键的区分

`model.py` 里的 FP4 相关逻辑一定要分成两套语义看：

### 8.1 真正用于推理主算子的 FP4

- expert 权重真的以 FP4 存储
- `linear()` 里命中 `weight.dtype == torch.float4_e2m1fn_x2`
- 激活先 `act_quant` 成 FP8
- 再进入 `fp4_gemm`

### 8.2 用于模拟量化误差的 FP4

- `Indexer` / `Compressor` 先做 Hadamard rotation
- 再 `fp4_act_quant(..., True)`
- 张量最终仍以 BF16 参与后续运算
- 只是数值已经带有 FP4 量化误差

如果不区分这两类，很容易误读成：

- indexer 真的在存 FP4 的 q/kv
- 或 expert 输入也是先量化到 FP4

但代码实际不是这样。

## 9. 一句话总结

这个工程里的 FP4 核心是“MoE expert 权重 FP4 存储 + FP8 激活/FP4 权重融合 GEMM”；而 `Indexer/Compressor` 里的 `fp4_act_quant(..., True)` 主要是 QAT 风格的激活误差模拟。量化粒度上，FP4 用每 32 元素一个 power-of-two scale，FP8 激活用每 128 元素一个 scale，最终在 `fp4_gemm` 里按 K 子块把两类 scale 乘回累加结果。
