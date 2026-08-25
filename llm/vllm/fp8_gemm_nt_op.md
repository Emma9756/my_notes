# `torch.ops.vllm.fp8_gemm_nt_op` 的输入、缩放因子 dtype 与布局

本文基于本地 vLLM `c64c356990` 的实现。这里讨论的是
`vllm/model_executor/kernels/linear/scaled_mm/deep_gemm.py` 中的普通二维
block-scaled FP8 linear，不是 DeepSeek V4 MegaMoE 的 FP4 expert GEMM。

## 结论

对下面这个实际调用（Blackwell SM100、启用 DeepGEMM E8M0 scale）：

```text
A:      [8192, 4096], torch.float8_e4m3fn
B:      [1536, 4096], torch.float8_e4m3fn
output: [8192, 1536], torch.bfloat16
As:     [8192, 8],    torch.int32
Bs:     [1536, 8],    torch.int32
```

`As` 和 `Bs` 的 dtype 都是 `torch.int32`。它们不是普通整数 scale：每个
`int32` 的 32 bit 中连续打包了 4 个 8-bit UE8M0 scale。这里的 `8` 来自：

```text
K 方向逻辑 scale 数 = 4096 / 128 = 32
物理 int32 数          = 32 / 4 = 8
```

因此，不能根据 `As.shape == [8192, 8]` 得出量化 group size 是 512；逻辑上
仍然是每 128 个 K 元素一个 scale，只是 4 个 scale 被打包成了一个
`int32`。

对本例实测的 stride：

```text
A:      [4096, 1]
As:     [1, 8192]
B:      [4096, 1]
Bs:     [1, 1536]
output: [1536, 1]
```

不能据此把五个张量的 `trans` 写成 `01110`，详见后文“`trans` 应如何
理解”。

这个结论有架构和配置前提。Hopper SM90 或关闭 UE8M0 转换时，DeepGEMM
使用 FP32 scale 路径，此时 `As`、`Bs` 是 `torch.float32`，不能套用上面的
`torch.int32` dtype 和 `/4` 后的形状。

## 算子的矩阵语义

`apply_block_scaled_mm()` 分配输出：

```python
output = torch.empty(
    (A.shape[0], B.shape[0]),
    dtype=out_dtype,
    device=A.device,
)
torch.ops.vllm.fp8_gemm_nt_op(
    A, As, B, Bs, output, self.use_deep_gemm_e8m0
)
```

所以该算子计算的是：

```text
C[M, N] = A[M, K] @ B[N, K].T
```

本例中 `M=8192`、`N=1536`、`K=4096`，输出自然是
`[8192, 1536]`。`can_implement()` 又明确限制输出 dtype 为
`torch.bfloat16`，并限制 activation scale group 为 `(1, 128)`。

## `trans` 应如何理解：不是 `01110`

这里有三个容易混淆的概念：

1. **GEMM 的代数转置**：只描述矩阵操作数 A、B，不描述 scale 和输出；
2. **PyTorch tensor 的 stride/layout**：描述数据如何存储；
3. **DeepGEMM 的 major type/TMA scale layout**：kernel 如何访问矩阵或
   scale，也不等价于对张量做数学上的 `.T`。

### 从 GEMM API 名字看：只有 `NT`

`fp8_gemm_nt` 的定义就是：

```text
output[M,N] = A[M,K] @ B[N,K].T
```

所以若 `N=0`、`T=1`，仅对 GEMM 的两个矩阵操作数编码，结果是：

```text
A, B = N, T = 0, 1
```

`B` 本身依然是 shape `[N,K]`、stride `[K,1]` 的 contiguous tensor；
`T` 是 GEMM 公式中读取 B 的方式。调用方不需要先构造 shape `[K,N]`、
stride `[1,K]` 的 `B.T` view。DeepGEMM 上游 API 也先检查输入 shape 必须
满足 `[M,K] @ [N,K].T`，再分别从 A、B 的 stride 判断其 major type。

`As`、`Bs` 和 `output` 没有独立的 GEMM transpose boolean。因此，把它们
和 A、B 一起写成五位 `01110` 没有对应的 API 语义。

### 如果只是给二维物理布局编码

假设另行约定：普通 row-major/最后一维连续记作 0，第一维连续的
column-major 记作 1。按本例 stride，纯布局编码是：

```text
A / As / B / Bs / output = 0 / 1 / 0 / 1 / 0 = 01010
```

但这个 `01010` **不是** GEMM transpose flags。特别是：

- `B.stride() == (4096,1)` 是 row-major，但它在 `fp8_gemm_nt` 公式中仍然
  是被转置的右操作数；
- `As`、`Bs` 的 `(1,MN)` stride 表示 MN-major/TMA-friendly scale
  layout，不表示对 scale 做数学转置；
- `output.stride() == (1536,1)` 是正常的 row-major BF16 输出。

因此，建议调试日志分别打印 `gemm_kind=NT` 和各 tensor 的
`shape/dtype/stride`，不要合并成一个五位 `trans` 串。

## `As` 为什么是 `[8192, 8] int32`

`DeepGemmFp8BlockScaledMMKernel` 创建 `QuantFP8` 时指定：

```python
QuantFP8(
    static=False,
    group_shape=GroupShape(1, 128),
    use_ue8m0=self.use_deep_gemm_e8m0,
    tma_aligned_scales=...,
    column_major_scales=True,
)
```

因此 activation `A[M,K]` 按每个 token、每 128 个 K 元素动态量化：

```text
As 的逻辑 shape = [M, K / 128] = [8192, 32]
```

在 Blackwell SM100 且 `VLLM_USE_DEEP_GEMM_E8M0` 生效时，
`QuantFP8.forward_cuda()` 进入
`per_token_group_quant_fp8_packed_for_deepgemm()`，直接生成 DeepGEMM 所需的
packed UE8M0 scale。4 个 UE8M0 byte 打包到一个 `int32`：

```text
As 的物理 shape = [M, ceil((K / 128) / 4)]
                = [8192, 8]
As.dtype        = torch.int32
```

## `Bs` 为什么从逻辑 `[12, 32]` 变成 `[1536, 8] int32`

权重 `B[N,K]` 使用 `(128,128)` block scale。权重刚加载时，`Bs` 的逻辑
shape 是：

```text
[N / 128, K / 128] = [1536 / 128, 4096 / 128] = [12, 32]
```

但 `process_weights_after_loading()` 会调用：

```text
deepgemm_post_process_fp8_weight_block()
  -> deepgemm_post_process_weight_scale_block()
  -> transform_sf_into_required_layout(
         sf=ws,
         mn=N,
         k=K,
         recipe=(1, 128, 128),
         num_groups=1,
     )
```

在 SM100 的 UE8M0 路径中，DeepGEMM 将 `(128,128)` 的 FP32 block scale
转换为 kernel 消费的 `(1,128)` MN-major scale：N 方向的每个 block scale
被展开到对应的 128 行，同时 K 方向每 4 个 UE8M0 scale 打包成一个
`int32`。所以：

```text
Bs 的 kernel 逻辑覆盖 = [N, K / 128] = [1536, 32]
Bs 的物理 shape       = [N, ceil((K / 128) / 4)]
                      = [1536, 8]
Bs.dtype              = torch.int32
```

因此 `[12,32]` 是 checkpoint/转换前的 block-scale 逻辑矩阵，`[1536,8]`
才是本例传给 `fp8_gemm_nt_op` 的 DeepGEMM 物理布局。二者描述的是不同
阶段，不能直接比较 shape。

## SM100 kernel 底层如何索引和使用 `Bs`

下面对应本例 `B[N,K]=[1536,4096]`、`Bs.shape=[1536,8]`、
`Bs.stride()=[1,1536]`、weight scale group K=128。

### 1. 一个 `Bs[n,p]` 覆盖哪些 B 元素

令：

```text
n = B 的输出通道/行号，0 <= n < N
q = K 方向的逻辑 scale group，q = floor(k / 128)
p = packed int32 下标，p = floor(q / 4)
j = int32 内的 byte 下标，j = q % 4
```

则：

```text
Bs[n,p] 的第 j 个 byte
  -> B[n, 128*q : 128*(q+1)] 的 UE8M0 scale
  -> q = 4*p + j
```

一个 `int32` 因而覆盖同一行 B 上连续的 4 个 K=128 group，也就是连续
512 个 K 元素：

```text
Bs[n,p].byte0 -> k = 512p + [  0,127]
Bs[n,p].byte1 -> k = 512p + [128,255]
Bs[n,p].byte2 -> k = 512p + [256,383]
Bs[n,p].byte3 -> k = 512p + [384,511]
```

例如 `p=3` 时，4 个 byte 分别服务 q=12、13、14、15，即 K 区间
`[1536,1664)`、`[1664,1792)`、`[1792,1920)`、`[1920,2048)`。

### 2. 全局内存中的地址计算

`Bs.stride()=[1,1536]`，因此 PyTorch 视角下：

```text
element_offset(Bs[n,p]) = n * 1 + p * 1536
byte_offset             = 4 * (n + p * 1536)
```

也就是说，同一个 packed-K 列 `p` 的所有 N 行在内存中连续；固定 n、增加
p 时跨过整列 N。这正是 MN-major/column-major scale layout。

DeepGEMM 不让普通 CUDA 线程逐元素执行上述地址式，而是为 `Bs` 创建 TMA
descriptor。对第 `n_block_idx` 个 N tile 和第 `k_block_idx` 个
`BLOCK_K=128` tile，kernel 计算：

```cpp
sfb_n_idx = n_block_idx * BLOCK_N;
sfb_k_idx = floor(k_block_idx / 4);
```

源码写成的通用形式是：

```cpp
shape_sfb_k = ceil_div(shape_k, kGranKB * 4);
sfb_k_idx = ceil_div(k_idx, BLOCK_K * kNumSFBStagesPerLoad);
```

本例模板参数为：

```text
kGranKB                 = 128
BLOCK_K                 = 128
kNumSFBStagesPerLoad    = 4
k_idx                   = k_block_idx * 128
shape_sfb_k             = ceil(4096 / (128*4)) = 8
```

在 `k_block_idx % 4 == 0` 时，TMA 一次把当前 N tile 的
`[BLOCK_N,1]` 个 packed `uint32` 从 `(sfb_n_idx,sfb_k_idx)` 搬到
`smem_sfb[stage_idx]`。所以一个 packed word 会复用 4 个 K tile，而不是
每 128 个 K 都重新从 HBM 读取。

### 3. shared memory 到 TMEM

SM100 kernel 为每个 pipeline stage 分配：

```cpp
SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);
```

TMA 到达后，一个专用 warp 先对每 128 个 `uint32` 做 4x32 transpose：

```cpp
values[i] = smem_ptr[i * 32 + lane_idx];  // i = 0..3
smem_ptr[lane_idx * 4 + i] = values[i];
```

随后 `SM100_UTCCP_4x32dp128bit_*cta` 把重排后的 SFB 从 shared memory
复制到 Tensor Memory (TMEM)：

```cpp
cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
```

这一步的 4x32 变换是为了匹配 Blackwell UMMA 对 scale factor 的 TMEM
布局要求，不是把 UE8M0 转成 FP32。

### 4. 4 个 byte 是如何被选中的

关键点是：DeepGEMM 的 SM100 kernel **没有**在热循环里写下面这种普通
CUDA 解包：

```cpp
uint8_t e = (packed >> (8 * j)) & 0xff;
```

它构造的是带 `cutlass::float_ue8m0_t` scale 类型的 Blackwell
block-scaled UMMA 指令。每个 `BLOCK_K=128` 内部又发射 4 次
`UMMA_K=32` MMA；由于本例 B scale 粒度为 128，这 4 次 MMA 共用同一个
scale byte：

```cpp
sfb_stage_in_group_idx = k_block_idx % 4;  // 0,1,2,3
sfb_id = sfb_stage_in_group_idx;           // kGranKB == 128
runtime_instr_desc =
    make_runtime_instr_desc_with_sf_id(instr_desc, sfa_id, sfb_id);
mma_t::fma(..., runtime_instr_desc, ..., kTmemStartColOfSFB);
```

因此 `sfb_id=0/1/2/3` 由 MMA instruction descriptor 指定当前应使用 packed
scale vector 中的哪一个 UE8M0 byte；TMEM 地址给出 SFB scale tile 的基址，
硬件 block-scaled MMA 完成 byte 选择、UE8M0 解释及乘法缩放。这里不是先
由 CUDA core 解包成 4 个 FP32，再交给 Tensor Core。

### 5. 用软件等价公式理解一个 byte

为了调试，可把 kernel 的效果概念性地写成：

```python
packed = int(Bs[n, p].item()) & 0xFFFFFFFF
ue8m0_exp = (packed >> (8 * j)) & 0xFF
scale_fp32_bits = ue8m0_exp << 23
scale = bitcast_uint32_to_float32(scale_fp32_bits)
```

vLLM/DeepGEMM 的 pack 路径正是取 power-of-two FP32 scale 的 exponent
byte；反向恢复时把该 byte 放回 FP32 exponent 位，mantissa 为 0。因此
`scale` 是 2 的幂。上面的 Python 是解释和核对数据用的等价模型，不是
SM100 kernel 热路径的实际指令序列。

### 6. 代入本例的一次取数

假设要计算输出通道 `n=100`、K 坐标 `k=1700`：

```text
q = floor(1700 / 128) = 13
p = floor(13 / 4)     = 3
j = 13 % 4            = 1

Bs element offset = 100 + 3*1536 = 4708 个 int32
Bs byte offset    = 4708*4        = 18832 bytes
选中的 scale     = Bs[100,3] 的 bits[15:8]，即 byte1
```

这个 scale 与 `B[100,1664:1792]` 配对。实际 kernel 是按 N tile 和 K tile
批量执行同样的映射，并通过 TMA、shared memory、UTCCP、TMEM 和 UMMA
流水完成，而不是为单个 `(n,k)` 发起一次标量 load。

## Hopper FP32 scale 路径对照

当 `use_deep_gemm_e8m0=False`（典型情况是 SM90）时：

- `A`、`B` 仍是 `torch.float8_e4m3fn`；
- `output` 仍是 `torch.bfloat16`；
- `As` 是 `torch.float32`，逻辑 shape 为 `[M,K/128]`，本例为
  `[8192,32]`，并采用 MN-major/TMA-friendly stride；
- `(128,128)` 权重 `Bs` 保持 `torch.float32` block-scale 语义，本例逻辑
  shape 为 `[12,32]`，其合法 stride/layout 由 DeepGEMM 检查；
- 不发生“4 个 UE8M0 scale 打包为一个 `int32`”，所以不会得到第二维 8。

需要注意：`shape` 不能完整表达 DeepGEMM scale layout。即使 FP32 `As`
显示为 `[M,K/128]`，它也可能是 column-major/TMA-aligned 的非普通连续
stride；排查时应同时打印 `dtype`、`shape` 和 `stride()`。

## DeepSeek V4 MegaMoE 的 FP4 expert GEMM

### 先说结论

DeepSeek V4 的 `deep_gemm_mega_moe` 路径与前面的普通 FP8 linear 有共同的
DeepGEMM scale-layout 思路，但不是一次可表示为
`A,As,B,Bs,output` 的 `fp8_gemm_nt_op` 调用，也不存在一个可写成五位串的
`trans` 集合。

它调用的是：

```python
deep_gemm.fp8_fp4_mega_moe(
    y,
    transformed_l1_weights,
    transformed_l2_weights,
    symm_buffer,
    activation_clamp=activation_clamp,
    fast_math=fast_math,
)
```

这个 mega-kernel 融合并重叠：

```text
EP dispatch
  -> L1: FP8 activation x FP4 expert w13
  -> SwiGLU
  -> L2: FP8 activation x FP4 expert w2
  -> EP combine
  -> BF16 y
```

因此，普通 linear 的单个 `M/N/K` 被替换成按 token routing、expert 和两个
GEMM stage 组织的数据流。

### vLLM 加载时的 FP4 权重与 scale

令：

```text
E = 本 rank 的 expert 数
H = hidden_size
I = intermediate_size
```

vLLM 为 checkpoint 数据分配：

```text
w13_weight:       [E, 2I, H/2], uint8
w13_weight_scale: [E, 2I, H/32], uint8
w2_weight:        [E, H, I/2],  uint8
w2_weight_scale:  [E, H, I/32], uint8
```

权重最后一维除以 2，是因为一个 `uint8` 打包两个 4-bit FP4 值。scale 最后
一维除以 32，表示 FP4 expert weight 每 32 个 K 元素对应一个 UE8M0 scale；
checkpoint 中 scale 以 UE8M0 raw byte (`uint8`) 保存。

`finalize_weights()` 分两步预处理：

1. `_ue8m0_uint8_to_float()` 把 UE8M0 exponent byte 恢复为具有相同
   power-of-two 数值的 FP32 tensor；
2. `transform_sf_into_required_layout(..., recipe=(1,32))` 在 SM100 上再将
   scale 转成 kernel 所需的 MN-major/TMA-aligned、packed `int32` 布局，
   然后 `transform_weights_for_mega_moe()` 对 L1/L2 权重做 MegaMoE 专用的
   interleave/layout 转换。

这里同样是 4 个 UE8M0 byte 打包进一个 `int32`，但 scale 粒度从普通
FP8 linear 的 K=128 变为 FP4 expert weight 的 K=32。因此 packed scale
最后一维的逻辑推导是：

```text
原始 UE8M0 scale 数 = K / 32
packed int32 数      = ceil((K / 32) / 4) = ceil(K / 128)
```

例如 H=4096 时，checkpoint 的 `w13_weight_scale` 每行有
`4096/32=128` 个 UE8M0 byte；打包后每行对应 32 个 `int32`。实际 tensor
还带 expert 维并采用 MN-major/TMA-aligned stride，且 L1/L2 会继续经过
`transform_weights_for_mega_moe()`，所以应以转换后实际打印的
`shape/dtype/stride` 为准。

### activation、scale 与输出

普通 FP8 linear 在调用点直接传入 `A`、`As`。MegaMoE 则先由
`prepare_megamoe_inputs()` 把数据写入 symmetric buffer：

```text
hidden_states -> symm_buffer.x       # FP8 activation
activation SF -> symm_buffer.x_sf    # UE8M0 scale layout
topk_ids      -> symm_buffer.topk_idx
topk_weights  -> symm_buffer.topk_weights
```

当前 vLLM staging kernel 对 activation 也使用 K=32 的量化 group；每次处理
128 个 K 元素，把其中 4 个 UE8M0 scale 合成一个 `int32` 写入 `x_sf`。
所以对于 `[tokens,H]` activation：

```text
x:    [tokens,H],     float8_e4m3fn
x_sf: [tokens,H/128], int32
```

这里 `x_sf` 的每个 `int32` 仍代表 4 个 K=32 group，而不是一个 K=128
group。这一点与前面的普通 FP8 linear 不同：普通 linear 的每个 UE8M0
scale 本身覆盖 K=128，打包后一个 `int32` 覆盖 512 个 K 元素。

L1 的中间结果、SwiGLU 后送给 L2 的 activation 及其 scale 也位于
symmetric buffer 的专用区域，由 mega-kernel 内部生产和消费，而不是作为
Python 层的独立 `A/As` 参数出现。最终 `y` 由 vLLM 用
`torch.empty_like(hidden_states, dtype=torch.bfloat16)` 创建。

### 与普通 FP8 linear 的异同

| 项目 | 普通 `fp8_gemm_nt_op` | DeepSeek V4 MegaMoE |
|---|---|---|
| API 粒度 | 一次二维 GEMM | dispatch + 两次 expert GEMM + SwiGLU + combine |
| activation | FP8 E4M3，scale group K=128 | FP8 E4M3，scale group K=32，位于 symmetric buffer |
| weight | FP8 E4M3 | packed FP4，两个 4-bit 值/byte |
| weight scale group | 通常 `(128,128)` block | 每行、每 32 个 K 元素一个 scale |
| SM100 scale 存储 | 4 个 UE8M0/`int32` | 同样 4 个 UE8M0/`int32` |
| 输出 | 单次 `[M,N]` BF16 | routing/combine 后 `[tokens,H]` BF16 |
| 转置表达 | API 明确是 `NT` | 融合 kernel 的内部 L1/L2 layout，不能用一个 `NT` 或五位串概括 |
| 权重预处理 | scale layout 转换 | scale layout 转换 + L1/L2 专用 weight interleave |

两条路径的共同点是：scale 的 `shape/stride` 是 kernel 物理布局，不能直接
当作普通矩阵的转置状态。主要差异是 MegaMoE 的 FP4 权重、K=32 scale
粒度、多 expert/routing 维度和两级融合执行。

## 判断依据

1. `deep_gemm.py::apply_block_scaled_mm()` 用 `A.shape[0]` 和
   `B.shape[0]` 创建输出，并按 `A, As, B, Bs, output` 的顺序调用自定义
   算子；`_fp8_gemm_nt_op()` 随后把它们组成 `(A,As)`、`(B,Bs)` 传给
   DeepGEMM `fp8_gemm_nt()`。
2. 同一文件的 `can_implement()` 限定输出为 BF16、activation group 为
   `(1,128)`；构造 `QuantFP8` 时显式启用 column-major scale，并按
   `self.use_deep_gemm_e8m0` 决定是否使用 UE8M0。
3. `input_quant_fp8.py::QuantFP8.forward_cuda()` 表明，SM100 的 UE8M0
   oracle 分支调用 packed DeepGEMM quantizer；普通 group-quant 分支则分配
   `torch.float32` scale。
4. `fp8_utils.py::deepgemm_post_process_fp8_weight_block()` 断言权重是
   `torch.float8_e4m3fn`，并把 weight scale 送入
   `transform_sf_into_required_layout()`；传入 recipe
   `(1, quant_block_shape[0], quant_block_shape[1])`。
5. vLLM 的 `utils/deep_gemm.py` 明确说明 UE8M0 会以 4:1 方式打包进
   `int32`。DeepGEMM 上游测试的 PyTorch reference 也先把 FP32 power-of-two
   scale 转成 `uint8`，再 `.view(dtype=torch.int)`，并把末维改为
   `aligned_k // 4`。
6. DeepSeek V4 `nvidia/model.py::finalize_weights()` 明确以 `(1,32)` recipe
   转换 FP4 expert scale，再调用 `transform_weights_for_mega_moe()`；forward
   调用 `fp8_fp4_mega_moe()`，而不是 `fp8_gemm_nt_op()`。
7. `nvidia/ops/prepare_megamoe.py` 固定 `BLOCK_K=128`、`GROUP_K=32`，将每
   4 个 activation UE8M0 exponent byte 合成一个 `int32` 写入 symmetric
   buffer 的 `x_sf`。
8. DeepGEMM `sm100_fp8_fp4_gemm_1d1d.cuh` 用
   `shape_sfb_k=ceil(K/(kGranKB*4))` 定义 packed K 维；按 N tile 和 packed-K
   坐标 TMA-load SFB，经过 shared-memory 4x32 transpose 和 UTCCP 后放入
   TMEM，再通过 UMMA descriptor 的 `sfb_id` 选择 4 个 UE8M0 byte。

建议现场确认：

```python
for name, x in (("A", A), ("As", As), ("B", B), ("Bs", Bs),
                ("output", output)):
    print(name, x.shape, x.dtype, x.stride())
```

## 源码链接

- 本地 vLLM：
  [`model_executor/kernels/linear/scaled_mm/deep_gemm.py`](../../../vllm/vllm/model_executor/kernels/linear/scaled_mm/deep_gemm.py)
- 本地 vLLM：
  [`model_executor/layers/quantization/input_quant_fp8.py`](../../../vllm/vllm/model_executor/layers/quantization/input_quant_fp8.py)
- 本地 vLLM：
  [`model_executor/layers/quantization/utils/fp8_utils.py`](../../../vllm/vllm/model_executor/layers/quantization/utils/fp8_utils.py)
- 本地 vLLM：[`utils/deep_gemm.py`](../../../vllm/vllm/utils/deep_gemm.py)
- 本地 vLLM：
  [`models/deepseek_v4/nvidia/model.py`](../../../vllm/vllm/models/deepseek_v4/nvidia/model.py)
- 本地 vLLM：
  [`models/deepseek_v4/nvidia/ops/prepare_megamoe.py`](../../../vllm/vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py)
- DeepGEMM 上游：
  [`csrc/apis/layout.hpp`](https://github.com/deepseek-ai/DeepGEMM/blob/main/csrc/apis/layout.hpp)
- DeepGEMM 上游：
  [`csrc/apis/gemm.hpp`](https://github.com/deepseek-ai/DeepGEMM/blob/main/csrc/apis/gemm.hpp)
- DeepGEMM 上游：
  [`csrc/apis/mega.hpp`](https://github.com/deepseek-ai/DeepGEMM/blob/main/csrc/apis/mega.hpp)
- DeepGEMM 上游：
  [`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh`](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh)
- DeepGEMM 上游：
  [`tests/test_layout.py`](https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_layout.py)
