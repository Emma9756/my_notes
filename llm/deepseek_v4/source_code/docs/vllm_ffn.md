# DeepSeek V4 `deepseek_v4.py:1256` `ffn` 计算梳理

本文对应 [`vllm/model_executor/models/deepseek_v4.py`](../vllm/model_executor/models/deepseek_v4.py) 第 `1256` 行：

```python
x = self.ffn(x, input_ids)
```

重点说明：

- 这是 `DeepseekV4DecoderLayer._forward_cuda()` 路径里的 FFN/MoE 调用，不是 ROCm 分支。
- 这里的 `ffn` 不是普通 dense MLP，而是 `DeepseekV4MoE`。
- `ffn_norm` 已经融合进 `self.ffn.norm_gate`，所以第 1256 行传入的是 “已经过 MHC pre、但还没做 MoE norm/gate” 的激活。

## 1. 调用链总览

第 1256 行的完整调用链是：

1. `DeepseekV4DecoderLayer._forward_cuda()`  
   [`vllm/model_executor/models/deepseek_v4.py:1206`](../vllm/model_executor/models/deepseek_v4.py:1206)
2. `DeepseekV4MoE.forward()`  
   [`vllm/model_executor/models/deepseek_v4.py:868`](../vllm/model_executor/models/deepseek_v4.py:868)
3. 先执行 `NormGateLinear.forward()`  
   [`vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py:97`](../vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py:97)
4. 然后分叉：
   - 普通路径：`FusedMoE.forward()`  
     [`vllm/model_executor/layers/fused_moe/layer.py:1305`](../vllm/model_executor/layers/fused_moe/layer.py:1305)
   - MegaMoE 路径：`fused_topk_bias()` + `DeepseekV4MegaMoEExperts.forward()`  
     [`vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py:99`](../vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py:99)  
     [`vllm/model_executor/models/deepseek_v4.py:611`](../vllm/model_executor/models/deepseek_v4.py:611)
5. 如果配置了 shared experts，再额外执行 `DeepseekV4MLP.forward()`  
   [`vllm/model_executor/models/deepseek_v4.py:126`](../vllm/model_executor/models/deepseek_v4.py:126)

## 2. 第 1256 行之前，`x` 是怎么来的

在 CUDA 路径中，FFN 前面这段代码是：

```python
residual, post_mix, res_mix, x = self.mhc_fused_post_pre(...)
x = self.ffn(x, input_ids)
```

对应位置：
[`vllm/model_executor/models/deepseek_v4.py:1240`](../vllm/model_executor/models/deepseek_v4.py:1240)

这里的 `mhc_fused_post_pre` 来自：
[`vllm/model_executor/layers/mhc.py:228`](../vllm/model_executor/layers/mhc.py:228)

它的语义是：

- 先把 attention 输出 `x` 与 HC residual streams 做一次 `MHCPost`
- 再对更新后的 residual streams 做下一次 `MHCPre`
- 返回：
  - `residual_cur`
  - `post_mix_cur`
  - `comb_mix_cur`
  - `layer_input_cur`

其中第 4 个返回值 `layer_input_cur` 就是第 1256 行传给 `ffn` 的 `x`。

### 2.1 shape 变化

从模型入口看，token embedding 会先扩成多路 HC 表示：

```python
hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
```

对应位置：
[`vllm/model_executor/models/deepseek_v4.py:1442`](../vllm/model_executor/models/deepseek_v4.py:1442)

所以层内 residual stream 主体张量长期是：

- `residual`: `[T, hc_mult, H]`

其中：

- `T = num_tokens`
- `hc_mult = config.hc_mult`
- `H = config.hidden_size`

而 `MHCPre` 的定义写得很明确：它返回

- `layer_input = sum_i pre_mix_i * residual_i`

对应：
[`vllm/model_executor/layers/mhc.py:15`](../vllm/model_executor/layers/mhc.py:15)

因此进入第 1256 行时，`x` 已经从多路 HC residual 折叠成二维张量：

- `x`: `[T, H]`

这也解释了为什么后续 `NormGateLinear` 能直接拿它做 `RMSNorm + gate matmul`。

### 2.2 dtype

这一段代码没有显式 cast，但从本文件和相关 kernel 习惯看，CUDA 推理主激活通常是：

- `x`: 通常为 `torch.bfloat16`

原因：

- attention / MHC 路径是推理 kernel，输出一般沿用 bf16 激活；
- MegaMoE 专家输出显式写死为 `torch.bfloat16`；
- router logits 则显式提升为 `torch.float32`。

这里更稳妥的说法是：

- **激活 `x` 的运行时 dtype 通常是 `bf16`，但代码层面没有在第 1256 行前强制写死。**

## 3. `self.ffn` 是什么

在 decoder layer 初始化时：

```python
self.ffn = DeepseekV4MoE(vllm_config, prefix=f"{prefix}.ffn")
```

对应位置：
[`vllm/model_executor/models/deepseek_v4.py:1120`](../vllm/model_executor/models/deepseek_v4.py:1120)

所以第 1256 行实际调用的是：

```python
DeepseekV4MoE.forward(hidden_states=x, input_ids=input_ids)
```

它的输入输出接口是：

- 输入 `hidden_states`: `[T, H]`
- 输入 `input_ids`: `[T]` 或 `None`
- 输出：`[T, H]`

## 4. `DeepseekV4MoE.forward()` 的公共前半段

代码见：
[`vllm/model_executor/models/deepseek_v4.py:868`](../vllm/model_executor/models/deepseek_v4.py:868)

### 步骤 1: hash MoE 场景检查 `input_ids`

```python
if self.norm_gate.tid2eid is not None and input_ids is None:
    raise ValueError(...)
```

含义：

- 如果这是 hash routing 的层，那么路由不是单纯从 logits top-k 得出，而要用 `input_ids` 查 `tid2eid` 表；
- 因此 `input_ids` 不能为空。

相关参数：

- `self.norm_gate.tid2eid`
  - shape: `[vocab_size, num_experts_per_tok]`
  - dtype:
    - 普通 MoE: `torch.int32`
    - MegaMoE: `torch.int64`

见：
[`vllm/model_executor/models/deepseek_v4.py:771`](../vllm/model_executor/models/deepseek_v4.py:771)

### 步骤 2: 保存原始 shape

```python
org_shape = hidden_states.shape
```

这里通常就是：

- `org_shape = (T, H)`

### 步骤 3: `norm_gate(hidden_states)`

```python
normed_x, router_logits = self.norm_gate(hidden_states)
```

这里的 `self.norm_gate` 是：

```python
self.norm_gate = NormGateLinear(
    hidden_size=config.hidden_size,
    num_experts=config.n_routed_experts,
    rms_eps=config.rms_norm_eps,
    ...
)
```

对应位置：
[`vllm/model_executor/models/deepseek_v4.py:762`](../vllm/model_executor/models/deepseek_v4.py:762)

## 5. `NormGateLinear.forward()` 做了什么

代码见：
[`vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py:97`](../vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py:97)

它本质上做两件事：

1. `RMSNorm`
2. gate/router 线性投影

### 5.1 参数

`NormGateLinear` 内部有两个关键子模块：

1. `self.norm = RMSNorm(hidden_size, eps=rms_eps, ...)`
2. `self.gate = GateLinear(hidden_size, num_experts, out_dtype=torch.float32, ...)`

其中：

- `norm.weight`
  - shape: `[H]`
  - dtype: 参数 dtype，通常随权重加载为 `bf16`
- `gate.weight`
  - shape: `[E, H]`
  - `E = config.n_routed_experts`
  - dtype: 参数 dtype，通常是 `bf16`
- `router logits` 输出 dtype 被要求为 `torch.float32`

`GateLinear` 的定义见：
[`vllm/model_executor/layers/fused_moe/router/gate_linear.py:12`](../vllm/model_executor/layers/fused_moe/router/gate_linear.py:12)

### 5.2 shape / dtype 变化

输入：

- `x`: `[T, H]`
- dtype: 通常 `bf16`

输出 1：

- `normed_x`: `[T, H]`
- dtype: 通常与激活/权重兼容，实践上通常还是 `bf16`

输出 2：

- `router_logits`: `[T, E]`
- dtype: `float32`

### 5.3 内部实现细节

`NormGateLinear.forward()` 分两种：

#### 路径 A: DeepSeek V4 Pro 特化 fused kernel

当满足：

- `hidden_size == 7168`
- `num_experts == 384`
- CUDA specialized router kernel 可用

会调用：

```python
torch.ops.vllm.dsv4_pro_norm_gate(x, self.norm.weight, self.gate.weight, self.rms_eps)
```

对应：
[`vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py:106`](../vllm/model_executor/layers/fused_moe/router/norm_gate_linear.py:106)

它返回：

- `normed_x: [T, H]`
- `router_logits: [T, E] float32`

#### 路径 B: 通用 fallback

```python
normed_x = self.norm(x)
logits, _ = self.gate(normed_x)
return normed_x, logits
```

其中 `GateLinear.forward()` 又分 3 档：

1. `ops.dsv3_router_gemm(...)`
2. `torch.mm(x, weight.T, out_dtype=torch.float32)`
3. 普通 `ReplicatedLinear.forward()`

对应：
[`vllm/model_executor/layers/fused_moe/router/gate_linear.py:92`](../vllm/model_executor/layers/fused_moe/router/gate_linear.py:92)

不管走哪一档，router 输出都被整理成 `float32`。

## 6. 第 1256 行之后的两条主路径

`DeepseekV4MoE.forward()` 在这里分叉：

- `self.use_mega_moe == False` -> 普通 `FusedMoE`
- `self.use_mega_moe == True` -> MegaMoE

分叉条件来自：
[`vllm/model_executor/models/deepseek_v4.py:731`](../vllm/model_executor/models/deepseek_v4.py:731)

---

## 7. 普通路径: `_forward_fused_moe()`

代码见：
[`vllm/model_executor/models/deepseek_v4.py:909`](../vllm/model_executor/models/deepseek_v4.py:909)

执行代码：

```python
normed_x, router_logits = self.norm_gate(hidden_states)
final_hidden_states = self.experts(
    hidden_states=normed_x,
    router_logits=router_logits,
    input_ids=input_ids,
)
return final_hidden_states.view(org_shape)
```

其中 `self.experts` 是 `FusedMoE`：
[`vllm/model_executor/models/deepseek_v4.py:851`](../vllm/model_executor/models/deepseek_v4.py:851)

### 7.1 传入 `FusedMoE.forward()` 的张量

- `hidden_states = normed_x`
  - shape: `[T, H]`
  - dtype: 通常 `bf16`
- `router_logits`
  - shape: `[T, E]`
  - dtype: `float32`
- `input_ids`
  - shape: `[T]` 或 `None`

### 7.2 `FusedMoE.forward()` 做了什么

`FusedMoE.forward()` 本身只是转发给 runner：

```python
return self.runner.forward(hidden_states, router_logits, input_ids)
```

见：
[`vllm/model_executor/layers/fused_moe/layer.py:1305`](../vllm/model_executor/layers/fused_moe/layer.py:1305)

真正执行在：
[`vllm/model_executor/layers/fused_moe/runner/moe_runner.py:592`](../vllm/model_executor/layers/fused_moe/runner/moe_runner.py:592)

runner 的关键逻辑可以概括为：

1. 必要时对 routed 输入做 transform / pad
2. 根据 `router_logits` 做路由
3. 把 token 分发到 top-k experts
4. 执行专家 MLP
5. 按路由权重把专家输出 combine 回来
6. 如有 shared experts，与 routed 输出相加
7. 如有 TP/EP，需要做 reduce

### 7.3 专家 MLP 的结构

普通专家和 shared experts 的基本 MLP 形态与 `DeepseekV4MLP.forward()` 一致：

```python
gate_up, _ = self.gate_up_proj(x)
x = self.act_fn(gate_up)
x, _ = self.down_proj(x)
return x
```

见：
[`vllm/model_executor/models/deepseek_v4.py:126`](../vllm/model_executor/models/deepseek_v4.py:126)

shape 变化可理解为：

1. `x`: `[N, H]`
2. `gate_up_proj`: `[N, 2I]`
3. `SiluAndMul`: `[N, I]`
4. `down_proj`: `[N, H]`

其中：

- `I = moe_intermediate_size`（routed expert）
- shared expert 则是 `I * n_shared_experts`

这里 `N` 不是全局 token 数，而是分发到某个专家后的 token 数。

### 7.4 输出

- `final_hidden_states`: `[T, H]`
- dtype: 一般与 MoE 输出激活一致，通常仍是 `bf16`

最后 `.view(org_shape)` 只是恢复原 shape；这里通常前后都是 `[T, H]`。

---

## 8. MegaMoE 路径

代码见：
[`vllm/model_executor/models/deepseek_v4.py:877`](../vllm/model_executor/models/deepseek_v4.py:877)

执行顺序：

1. `normed_x, router_logits = self.norm_gate(hidden_states)`
2. `topk_weights, topk_ids = fused_topk_bias(...)`
3. `final_hidden_states = self.experts(normed_x, topk_weights, topk_ids, ...)`
4. 如果有 shared experts，再相加
5. `return final_hidden_states.view(org_shape)`

### 8.1 `fused_topk_bias()`

代码见：
[`vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py:99`](../vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py:99)

它输入：

- `hidden_states = normed_x`: `[T, H]`
- `gating_output = router_logits`: `[T, E]`, `float32`
- `topk = K = num_experts_per_tok`
- `input_tokens = input_ids`
- `hash_indices_table = tid2eid` 或 `None`

它输出：

- `topk_weights`: `[T, K]`, `float32`
- `topk_ids`: `[T, K]`, `int32` 或 `int64`

细节：

1. 根据 `scoring_func` 计算路由分数
   - DeepSeek V4 这里默认是 `sqrtsoftplus`
2. 如果有 `e_score_correction_bias`，加到选 expert 用的分数上
3. 如果是 hash routing，直接 `topk_indices = hash_indices_table[input_tokens]`
4. 否则对 expert 维做 `topk`
5. 用选中的 expert index 从分数里 gather 出 `topk_weights`
6. 如果 `renormalize=True`，对每个 token 的 top-k 权重重新归一化

### 8.2 `DeepseekV4MegaMoEExperts.forward()`

代码见：
[`vllm/model_executor/models/deepseek_v4.py:611`](../vllm/model_executor/models/deepseek_v4.py:611)

输入：

- `hidden_states`: `[T, H]`
- `topk_weights`: `[T, K]`, `float32`
- `topk_ids`: `[T, K]`, `int64`（MegaMoE 路径）  
  见 [`vllm/model_executor/models/deepseek_v4.py:772`](../vllm/model_executor/models/deepseek_v4.py:772)

它先分配输出：

```python
y = torch.empty_like(hidden_states, dtype=torch.bfloat16)
```

所以输出 dtype 在这里是明确的：

- `y`: `[T, H]`, `bfloat16`

然后调用自定义 op：

```python
torch.ops.vllm.deepseek_v4_mega_moe_experts(...)
```

这个 op 内部最终会执行 `_run_mega_moe()`：
[`vllm/model_executor/models/deepseek_v4.py:637`](../vllm/model_executor/models/deepseek_v4.py:637)

核心步骤：

1. `_stage_deepseek_v4_mega_moe_inputs(...)`
   - 把 `hidden_states [T, H]` 量化/打包到 FP8 staging buffer
   - 把 `topk_ids/topk_weights` 拷到对称缓冲区
2. `finalize_weights()`
   - 确保专家权重已转换到 deep_gemm 需要的布局
3. `deep_gemm.fp8_fp4_mega_moe(...)`
   - 用 FP8 激活 + FP4 专家权重执行 MegaMoE

### 8.3 MegaMoE 权重 dtype

从 `DeepseekV4MegaMoEExperts` 参数定义可直接看出：

- `w13_weight`: `torch.uint8`
- `w13_weight_scale`: `torch.uint8`
- `w2_weight`: `torch.uint8`
- `w2_weight_scale`: `torch.uint8`

见：
[`vllm/model_executor/models/deepseek_v4.py:431`](../vllm/model_executor/models/deepseek_v4.py:431)

这些不是普通浮点权重，而是量化后的原始存储格式；真实计算由 `deep_gemm` kernel 解码执行。

---

## 9. Shared Experts 路径

如果配置了 `n_shared_experts`，两条主路径都会额外执行：

```python
shared_output = self.shared_experts(normed_x)
final_hidden_states += shared_output
```

见：
[`vllm/model_executor/models/deepseek_v4.py:903`](../vllm/model_executor/models/deepseek_v4.py:903)

这里的 `shared_experts` 是 `DeepseekV4MLP`，不是 MoE 路由。

输入输出：

- 输入: `normed_x [T, H]`
- 输出: `shared_output [T, H]`

注意它吃的是 `normed_x`，不是原始 `x`，也不是 routed experts 的输出。

---

## 10. 一张表看完整 shape / dtype 流转

以下按 CUDA 第 1256 行对应路径整理：

| 阶段 | 张量 | shape | dtype |
|---|---|---:|---|
| 模型层内 residual stream | `residual` | `[T, hc_mult, H]` | 通常 `bf16` |
| `mhc_fused_post_pre` 输出给 FFN 的输入 | `x` | `[T, H]` | 通常 `bf16` |
| `NormGateLinear` 输出 1 | `normed_x` | `[T, H]` | 通常 `bf16` |
| `NormGateLinear` 输出 2 | `router_logits` | `[T, E]` | `float32` |
| `fused_topk_bias` 输出 | `topk_weights` | `[T, K]` | `float32` |
| `fused_topk_bias` 输出 | `topk_ids` | `[T, K]` | 普通路径 `int32`，MegaMoE `int64` |
| routed experts 输出 | `final_hidden_states` | `[T, H]` | 通常 `bf16` |
| shared experts 输出 | `shared_output` | `[T, H]` | 通常 `bf16` |
| 第 1256 行返回值 | `x` | `[T, H]` | 通常 `bf16` |

其中：

- `T = num_tokens`
- `H = hidden_size`
- `E = n_routed_experts`
- `K = num_experts_per_tok`

## 11. 和 ROCm 分支的差异

第 1256 行属于 CUDA 路径。ROCm 路径在：
[`vllm/model_executor/models/deepseek_v4.py:1259`](../vllm/model_executor/models/deepseek_v4.py:1259)

ROCm 的 FFN 段是：

```python
residual = x
x, post, comb = self.hc_pre(...)
x = self.ffn_norm(x)
x = self.ffn(x, input_ids)
x = self.hc_post(x, residual, post, comb)
```

差异点：

1. ROCm 没有走 `mhc_fused_post_pre`
2. ROCm 仍显式调用 `self.ffn_norm(x)`
3. CUDA 路径已经把 `ffn_norm` 融到 `self.ffn.norm_gate`

所以如果你专门分析第 1256 行，应该以 CUDA 路径为准，不要把 ROCm 的 `ffn_norm` 混进去。

## 12. 最简结论

第 1256 行的 `self.ffn(x, input_ids)` 可以概括为：

1. 输入 `x` 已由 MHC 多路 residual 折叠成 `[T, H]`
2. `NormGateLinear` 对 `x` 做 `RMSNorm + gate matmul`
3. 得到：
   - `normed_x [T, H]`
   - `router_logits [T, E] float32`
4. 根据 backend：
   - 普通路径：`FusedMoE` 完成 top-k 路由、dispatch、专家计算、combine
   - MegaMoE 路径：先 `fused_topk_bias` 得到 `topk_weights/topk_ids`，再调用 deep_gemm 专家 kernel
5. 如有 shared experts，再把 `DeepseekV4MLP(normed_x)` 的结果加回去
6. 最终输出恢复为 `[T, H]`

## 13. MegaMoE 路径的更细计算过程

如果你继续追 `DeepseekV4MoE.forward()` 的 MegaMoE 分支，真正的关键在于：

1. vLLM 先把输入激活和路由结果整理到 `DeepGEMM` 需要的 buffer 格式
2. `DeepGEMM` 再用一个 fused kernel，把 “EP dispatch + 两层 expert GEMM + SwiGLU + combine” 串起来做

下面按执行先后拆开。

### 13.1 入口回顾

MegaMoE 主逻辑在：
[`vllm/model_executor/models/deepseek_v4.py:877`](../vllm/model_executor/models/deepseek_v4.py:877)

```python
normed_x, router_logits = self.norm_gate(hidden_states)
topk_weights, topk_ids = fused_topk_bias(...)
final_hidden_states = self.experts(
    normed_x,
    topk_weights,
    topk_ids,
    activation_clamp=activation_clamp,
)
```

其中 `self.experts` 是 `DeepseekV4MegaMoEExperts`。

## 14. MegaMoE 的权重准备

### 14.1 参数在 vLLM 中的原始存储格式

`DeepseekV4MegaMoEExperts` 持有两层专家权重：

- L1: `w13_weight`, `w13_weight_scale`
- L2: `w2_weight`, `w2_weight_scale`

对应：
[`vllm/model_executor/models/deepseek_v4.py:431`](../vllm/model_executor/models/deepseek_v4.py:431)

shape 是：

- `w13_weight`: `[E_local, 2I, H/2]`, `uint8`
- `w13_weight_scale`: `[E_local, 2I, H/32]`, `uint8`
- `w2_weight`: `[E_local, H, I/2]`, `uint8`
- `w2_weight_scale`: `[E_local, H, I/32]`, `uint8`

这里：

- `E_local = num_local_experts`
- `H = hidden_size`
- `I = intermediate_size`

注意 `w13_weight` / `w2_weight` 的最后一维是 `/2`，因为 FP4 权重是 4 bit 打包存储，两个 FP4 元素共用 1 byte。

`*_scale` 的最后一维是 `/32`，因为 scale 是按 `gran_k=32` 分组的。

### 14.2 `finalize_weights()` 做了什么

代码见：
[`vllm/model_executor/models/deepseek_v4.py:546`](../vllm/model_executor/models/deepseek_v4.py:546)

它做三件事：

1. 校验运行时条件
2. 把 loader-side scale 布局转换成 `DeepGEMM` 所需布局
3. 进一步把权重变成 MegaMoE fused kernel 友好的格式

#### 步骤 A: 运行时约束检查

`_check_runtime_supported()` 要求：

- CUDA
- 权重在 CUDA 上
- GPU capability major == `10`，也就是 SM100
- `hidden_size % 128 == 0`
- `intermediate_size % 128 == 0`

见：
[`vllm/model_executor/models/deepseek_v4.py:530`](../vllm/model_executor/models/deepseek_v4.py:530)

#### 步骤 B: scale 从 `ue8m0 uint8` 还原为 float32

```python
def _ue8m0_uint8_to_float(sf):
    return (sf.to(torch.int32) << 23).view(torch.float32)
```

见：
[`vllm/model_executor/models/deepseek_v4.py:526`](../vllm/model_executor/models/deepseek_v4.py:526)

这里说明：

- vLLM checkpoint 加载时把 FP8/UE8M0 scale 的原始字节保存在 `uint8`
- `finalize_weights()` 再把这些指数位恢复成 `float32` 表示，供后续 layout transform 使用

#### 步骤 C: `transform_sf_into_required_layout`

```python
w13_scale = deep_gemm.transform_sf_into_required_layout(...)
w2_scale = deep_gemm.transform_sf_into_required_layout(...)
```

对应：
[`vllm/model_executor/models/deepseek_v4.py:553`](../vllm/model_executor/models/deepseek_v4.py:553)

它把 scale tensor 调整成 `DeepGEMM` GEMM kernel 要求的内存布局。

虽然这里没有在 vLLM 仓库内展开实现，但从 `DeepGEMM` 测试能看到同样做法：
[`DeepGEMM/tests/test_mega_moe.py:96`](</data/dnn/qinzq/repository/DeepGEMM/tests/test_mega_moe.py:96>)

#### 步骤 D: `transform_weights_for_mega_moe`

```python
deep_gemm.transform_weights_for_mega_moe(
    (w13_weight_int8, w13_scale),
    (w2_weight_int8, w2_scale),
)
```

vLLM 调用位置：
[`vllm/model_executor/models/deepseek_v4.py:567`](../vllm/model_executor/models/deepseek_v4.py:567)

DeepGEMM 实现位置：
[`DeepGEMM/deep_gemm/mega/__init__.py:96`](</data/dnn/qinzq/repository/DeepGEMM/deep_gemm/mega/__init__.py:96>)

它做两类变换：

1. L1 权重做 gate/up 交错排列
2. L1/L2 的 scale tensor 转成 UTCCP 所需布局

##### L1 gate/up 交错

DeepGEMM 代码：

```python
# [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
```

见：
[`DeepGEMM/deep_gemm/mega/__init__.py:75`](</data/dnn/qinzq/repository/DeepGEMM/deep_gemm/mega/__init__.py:75>)

意思是：

- 原本 L1 的 `2I` 维前半是 gate，后半是 up
- MegaMoE kernel 更喜欢把二者按小块交错，方便后面 fused SwiGLU 一起消费

##### scale 转置

DeepGEMM 代码：

```python
result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
            .transpose(2, 3)
            .reshape(num_groups, mn, packed_sf_k))
```

见：
[`DeepGEMM/deep_gemm/mega/__init__.py:87`](</data/dnn/qinzq/repository/DeepGEMM/deep_gemm/mega/__init__.py:87>)

它在调整 scale 的内存排列，以配合 UTCCP kernel 的访问模式。

### 14.3 变换后权重的含义

`finalize_weights()` 之后，vLLM 保存：

- `_transformed_l1_weights = (l1_w, l1_sf)`
- `_transformed_l2_weights = (l2_w, l2_sf)`

这两个 tuple 会被直接传给：

```python
deep_gemm.fp8_fp4_mega_moe(...)
```

此后原始 loader-side `Parameter` 会被丢掉：
[`vllm/model_executor/models/deepseek_v4.py:573`](../vllm/model_executor/models/deepseek_v4.py:573)

原因很直接：

- MegaMoE kernel 不再使用原始 layout
- 节省显存

## 15. MegaMoE 的对称通信缓冲区 `SymmBuffer`

### 15.1 vLLM 如何创建

vLLM 中：
[`vllm/model_executor/models/deepseek_v4.py:584`](../vllm/model_executor/models/deepseek_v4.py:584)

```python
symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group,
    self.num_experts,
    self.max_num_tokens,
    self.top_k,
    self.hidden_size,
    self.intermediate_size,
)
```

### 15.2 DeepGEMM 中的定义

见：
[`DeepGEMM/deep_gemm/mega/__init__.py:16`](</data/dnn/qinzq/repository/DeepGEMM/deep_gemm/mega/__init__.py:16>)

```python
(self.x, self.x_sf,
 self.topk_idx, self.topk_weights,
 self.l1_acts, self.l1_acts_sf,
 self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)
```

说明 `SymmBuffer` 至少包含这些视图：

- `x`
- `x_sf`
- `topk_idx`
- `topk_weights`
- `l1_acts`
- `l1_acts_sf`
- `l2_acts`
- `l2_acts_sf`

可以把它理解成一个统一的大块对称显存，里面切出：

1. 输入 token 激活及其 scale
2. 路由结果
3. L1 输出激活及其 scale
4. L2 前的中间激活及其 scale

### 15.3 “对称” 的含义

DeepGEMM 使用的是：

```python
torch.distributed._symmetric_memory
```

见：
[`DeepGEMM/deep_gemm/mega/__init__.py:8`](</data/dnn/qinzq/repository/DeepGEMM/deep_gemm/mega/__init__.py:8>)

这意味着：

- 各个 EP rank 上会分配结构一致的大 buffer
- kernel 可以基于 `buffer_ptrs` 直接做跨 rank 访问/通信

`fp8_fp4_mega_moe()` 调用时也把这些指针传进去了：

```python
sym_buffer.buffer,
sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
```

见：
[`DeepGEMM/deep_gemm/mega/__init__.py:117`](</data/dnn/qinzq/repository/DeepGEMM/deep_gemm/mega/__init__.py:117>)

## 16. vLLM 的输入 staging 在做什么

vLLM 侧 staging 入口：
[`vllm/model_executor/models/deepseek_v4.py:327`](../vllm/model_executor/models/deepseek_v4.py:327)

调用点：
[`vllm/model_executor/models/deepseek_v4.py:650`](../vllm/model_executor/models/deepseek_v4.py:650)

### 16.1 输入和输出

输入：

- `hidden_states`: `[T, H]`, 通常 `bf16`
- `topk_weights`: `[T, K]`, `float32`
- `topk_ids`: `[T, K]`, `int64`

输出写入 `symm_buffer`：

- `x[:T]`
- `x_sf[:T]`
- `topk_idx[:T]`
- `topk_weights[:T]`

### 16.2 `hidden_states -> FP8 + packed scale`

Triton kernel `_deepseek_v4_stage_mega_moe_inputs_kernel` 做的事情可以概括为：

1. 读一个 token 的 `hidden_states`
2. 按 `GROUP_K=32` 分组求每组绝对值最大值 `amax`
3. 用 `amax / 448.0` 生成 scale
4. 把 scale 四舍五入到 `UE8M0` 可表达的指数形式
5. 用该 scale 的倒数把激活缩放后量化成 `float8_e4m3`
6. 把 FP8 激活写到 `x_fp8`
7. 把 scale 的指数位打包后写到 `x_sf`

对应代码：
[`vllm/model_executor/models/deepseek_v4.py:256`](../vllm/model_executor/models/deepseek_v4.py:256)

所以 staging 后：

- `x`: 是 token 激活的 FP8 版本
- `x_sf`: 是每 32 通道一组的 scale，且是 packed 后的表示

### 16.3 `topk_ids / topk_weights` 直接写进 buffer

同一个 staging kernel 在 `k_block_id == 0` 时还会：

- 把 `topk_ids` 拷到 `topk_idx_out`
- 把 `topk_weights` 拷到 `topk_weights_out`

见：
[`vllm/model_executor/models/deepseek_v4.py:294`](../vllm/model_executor/models/deepseek_v4.py:294)

这里没有再做路由选择，只是把先前 `fused_topk_bias()` 已经选好的结果搬进 `DeepGEMM` buffer。

## 17. `DeepGEMM` fused kernel 在概念上等价于什么

最有参考价值的是：
[`DeepGEMM/tests/test_mega_moe.py:157`](</data/dnn/qinzq/repository/DeepGEMM/tests/test_mega_moe.py:157>)

测试里把 fused MegaMoE 和一个可读性更强的 baseline 做了 bitwise 对齐。baseline 顺序是：

1. `dispatch`
2. `L1 grouped GEMM`
3. `swiglu_apply_weight_to_fp8`
4. `L2 grouped GEMM`
5. `combine`

也就是说，`deep_gemm.fp8_fp4_mega_moe(...)` 在概念上等价于：

### 17.1 Dispatch

把每个 token 的 top-k 副本发给对应 expert 所在 rank。

测试中：

```python
recv_x, _, recv_topk_weights, handle, _ = ep_buffer.dispatch(...)
```

见：
[`DeepGEMM/tests/test_mega_moe.py:158`](</data/dnn/qinzq/repository/DeepGEMM/tests/test_mega_moe.py:158>)

dispatch 后可把收到的数据理解成：

- `recv_x`: 当前 rank 需要处理的所有 token-expert 对应的输入激活
- `recv_topk_weights`: 与这些 token-expert 对应的路由权重

这里 token 数不再是原始 `T`，而是：

- `N_recv =` 当前 EP rank 实际收到的 token-expert pair 数

如果一个 token 被路由到 8 个 expert，那么它理论上会被复制成最多 8 份参与 dispatch。

### 17.2 L1 grouped GEMM

测试中的第一层专家 matmul：

```python
deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
    recv_x, l1_weights, l1_y, handle.psum_num_recv_tokens_per_expert,
    use_psum_layout=True, recipe=(1, 1, 32))
```

见：
[`DeepGEMM/tests/test_mega_moe.py:167`](</data/dnn/qinzq/repository/DeepGEMM/tests/test_mega_moe.py:167>)

可理解成：

- 对每个 local expert，取属于它的 token 子集
- 做 `FP8 activation x FP4 weight -> BF16 output`

shape 逻辑：

- 输入 `recv_x`: `[N_recv, H]` 的 FP8 表示（加独立 scale）
- L1 权重: `[E_local, 2I, H]` 的 FP4 表示（打包存储）
- 输出 `l1_y`: `[N_recv, 2I]`, `bf16`

### 17.3 `SwiGLU + 路由权重乘法 + 再量化为 FP8`

baseline 里这一段最关键：

```python
l1_y = tilelang_ops.swiglu_apply_weight_to_fp8(
    x=l1_y,
    topk_weights=recv_topk_weights,
    ...
    output_bf16=False,
)
```

见：
[`DeepGEMM/tests/test_mega_moe.py:171`](</data/dnn/qinzq/repository/DeepGEMM/tests/test_mega_moe.py:171>)

它对应的 TileLang kernel 代码见：
[`DeepGEMM/third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py:150`](</data/dnn/qinzq/repository/DeepGEMM/third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py:150>)

核心公式就在这里：

```python
y = silu(gate) * up * topk_weight
```

更精确地说，kernel 实现为：

```python
gate / (1 + exp(-gate)) * up * topk_weight
```

见：
[`DeepGEMM/third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py:92`](</data/dnn/qinzq/repository/DeepGEMM/third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py:92>)

这一段同时做了四件事：

1. 把 `2I` 切成 `gate[I]` 和 `up[I]`
2. 计算 `silu(gate) * up`
3. 把当前 expert 路由权重 `topk_weight` 乘进去
4. 再把结果量化回 FP8，并写出对应 scale

所以这一步之后：

- 输入: `l1_y [N_recv, 2I] bf16`
- 输出激活: `[N_recv, I] fp8`
- 输出 scale: 每 32 通道一组

这一步非常重要，因为它解释了：

- **MegaMoE 不是在最终 combine 时才乘 `topk_weights`**
- 而是在 expert 的中间激活阶段就把路由权重融合进去了

### 17.4 L2 grouped GEMM

baseline 的第二层专家 matmul：

```python
deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
    l1_y, l2_weights, l2_y, handle.psum_num_recv_tokens_per_expert,
    use_psum_layout=True, recipe=(1, 1, 32))
```

见：
[`DeepGEMM/tests/test_mega_moe.py:184`](</data/dnn/qinzq/repository/DeepGEMM/tests/test_mega_moe.py:184>)

shape：

- 输入: `[N_recv, I]` 的 FP8 激活
- L2 权重: `[E_local, H, I]` 的 FP4 权重
- 输出: `l2_y [N_recv, H]`, `bf16`

### 17.5 Combine

最后：

```python
ep_buffer.combine(l2_y, handle=handle)[0]
```

见：
[`DeepGEMM/tests/test_mega_moe.py:187`](</data/dnn/qinzq/repository/DeepGEMM/tests/test_mega_moe.py:187>)

含义：

- 把各 rank 上每个 token 对应的 expert 输出送回原 token 所属位置
- 对同一 token 的多个 expert 输出做求和

由于 `topk_weight` 已经在 17.3 中融合进每条 expert 分支，这里的 combine 本质上更接近：

- “按 token id 聚合求和”

而不是再做一次显式加权和。

## 18. 用一个公式概括 MegaMoE 单 token 计算

对 token `t`，假设路由到 experts `e_1 ... e_K`，对应权重 `a_1 ... a_K`。

MegaMoE 最终输出可写成：

```text
out_t = Σ_j  W2[e_j] ( SwiGLU( W13[e_j] * x_t ) * a_j )
```

其中：

- `W13[e_j] * x_t -> [2I]`
- `SwiGLU(...) -> [I]`
- `* a_j` 在中间激活阶段完成
- `W2[e_j] -> [H, I]`
- 最后对 `j` 求和得到 `[H]`

这和普通 MoE 的数学形式一致，但 MegaMoE 在实现上把：

- dispatch
- grouped GEMM
- SwiGLU
- 路由权重乘法
- combine

更深地融合进了 `DeepGEMM` 的通信和 kernel 流水里。

## 19. MegaMoE 路径的 shape / dtype 细表

以下是更贴近 fused kernel 视角的流转：

| 阶段 | 张量 | shape | dtype |
|---|---|---:|---|
| vLLM 传入 MegaMoE | `normed_x` | `[T, H]` | 通常 `bf16` |
| 路由结果 | `topk_weights` | `[T, K]` | `float32` |
| 路由结果 | `topk_ids` | `[T, K]` | `int64` |
| staging 后输入激活 | `symm_buffer.x` | `[T, H]` | `fp8 e4m3` |
| staging 后输入 scale | `symm_buffer.x_sf` | 与 `[T, H/32]` 对应的 packed 布局 | packed UE8M0 / `int32` view |
| dispatch 后局部 token-expert 对 | `recv_x` | `[N_recv, H]` | FP8 表示 |
| L1 输出 | `l1_y` | `[N_recv, 2I]` | `bf16` |
| SwiGLU + topk_weight 后 | `l1_acts` | `[N_recv, I]` | `fp8 e4m3` |
| 对应 scale | `l1_acts_sf` | 与 `[N_recv, I/32]` 对应布局 | UE8M0 / packed |
| L2 输出 | `l2_y` | `[N_recv, H]` | `bf16` |
| combine 后 | `y` | `[T, H]` | `bf16` |

其中：

- `T = 原始 token 数`
- `K = top-k expert 数`
- `N_recv = 当前 EP rank 实际接收到的 token-expert pair 数`

`N_recv` 一般不是固定值，也不等于 `T`，它取决于：

- 路由分布
- expert parallel 切分
- 当前 rank 持有哪些 experts

## 20. 这条路径最容易忽略的三个点

### 20.1 `topk_weights` 在中间层就融合了

不是最后 combine 时再乘，而是在 `SwiGLU` 后立刻乘：

```python
y = silu(gate) * up * topk_weight
```

这会减少后续额外算子和访存。

### 20.2 L1 权重不是简单 `[gate | up]` 连续布局

为了 fused kernel 读取效率，`transform_weights_for_mega_moe()` 先把 L1 变成交错布局：

- `[gate0..7, up0..7, gate8..15, up8..15, ...]`

### 20.3 vLLM 自己做了输入量化 staging

也就是：

- `normed_x` 在进入 `DeepGEMM` 之前，就先变成 FP8 + packed scale

因此 `DeepGEMM` 主 kernel 不需要再从 BF16 临时量化输入激活。

## 21. `m_grouped_fp8_fp4_gemm_nt_contiguous` 的计算过程

这一节把 `DeepGEMM` 的
`m_grouped_fp8_fp4_gemm_nt_contiguous` 拆开讲清楚：

- 数学上到底在算什么
- `A / B / grouped_layout / D` 的 shape 是什么
- FP8 / FP4 的量化粒度是什么
- 在 `vllm` 的 `DeepGemmFP4Experts` 路径里，参数是怎么变化的

如果你想看“从 Python 入口到 TMA / UMMA / barrier 的完整调用链”，
可以直接跳到这份单独文档：
[deep_gemm_m_grouped_fp8_fp4_gemm_nt_contiguous_cuda_walkthrough.md](</data/dnn/qinzq/repository/vllm/docs/deep_gemm_m_grouped_fp8_fp4_gemm_nt_contiguous_cuda_walkthrough.md>)

相关入口：

- Python 包装：
  [`vllm/utils/deep_gemm.py:314`](</data/dnn/qinzq/repository/vllm/vllm/utils/deep_gemm.py:314>)
- C++ API：
  [`DeepGEMM/csrc/apis/gemm.hpp:144`](</data/dnn/qinzq/repository/DeepGEMM/csrc/apis/gemm.hpp:144>)
- SM100 1D1D 调度：
  [`DeepGEMM/csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:158`](</data/dnn/qinzq/repository/DeepGEMM/csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:158>)

---

## 22. 接口签名和参数语义

C++ 暴露出来的签名是：

```cpp
m_grouped_fp8_fp4_gemm_nt_contiguous(
    a=(a_data, a_scale),
    b=(b_data, b_scale),
    d,
    grouped_layout,
    recipe,
    recipe_a,
    recipe_b,
    compiled_dims,
    disable_ue8m0_cast,
    use_psum_layout,
    expected_m_for_psum_layout)
```

其中约束非常明确：

- 数学语义必须是 `[M, K] @ [G, N, K].mT`
- `d` 必须是 `[M, N]`
- `d.dtype == torch.bfloat16`
- `grouped_layout.dtype == torch.int32`
- `A` 是一个拼接后的二维矩阵，不是 `[G, M, K]`
- `B` 是按 group 划分的三维权重

从 `DeepGEMM` 的 host check 看，最终要满足：

```text
A: [M, K]
B: [G, N, K]      # 逻辑 shape
D: [M, N]
```

其中 `G` 是 group 数，在 MoE 里通常就是当前 rank 上的 `local expert` 数。

---

## 23. 数学语义

设一共有 `G` 个 group，第 `i` 个 group 的有效 token 数是 `m_i`。

逻辑上，这个 kernel 做的是：

```text
D_i = A_i @ B_i^T
```

其中：

- `A_i`: `[m_i, K]`
- `B_i`: `[N, K]`
- `D_i`: `[m_i, N]`

把所有 group 的输入沿 `M` 维拼起来：

```text
A = concat(A_0, A_1, ..., A_{G-1})    # [M, K]
M = sum_i align(m_i, alignment)
```

kernel 根据 `grouped_layout` 知道每一段 `A[start_i:end_i]` 属于哪个 group，
再去读取对应的 `B_i`，最后把结果写到 `D[start_i:end_i]`。

所以 `nt` 的含义就是：

- `A` 按 `[M, K]` 参与计算，不转置
- `B_i` 按 `[N, K]` 存储，但乘法语义是 `B_i^T`

也就是：

```text
[M, K] @ [K, N] -> [M, N]
```

---

## 24. `contiguous` 布局下的真实 shape

### 24.1 `A`

`generate_m_grouped_contiguous()` 直接构造：

```python
a = torch.randn((m, k), ...)
```

见：
[`DeepGEMM/tests/generators.py:296`](</data/dnn/qinzq/repository/DeepGEMM/tests/generators.py:296>)

所以物理 shape 不是 `[G, max_m, K]`，而是：

```text
A_data: [M, K]
```

这里：

- `actual_m[i]` 是 group `i` 的真实 token 数
- `aligned_m[i] = align(actual_m[i], alignment)`
- `M = sum_i aligned_m[i]`

padding 的那几行会在生成器里被置零：

```python
a[actual_end: aligned_end] = 0
```

### 24.2 `B`

同一个生成器直接构造：

```python
b = torch.randn((num_groups, n, k), ...)
```

见：
[`DeepGEMM/tests/generators.py:297`](</data/dnn/qinzq/repository/DeepGEMM/tests/generators.py:297>)

因此 `B` 的逻辑 shape 是：

```text
B_data: [G, N, K]
```

每个 `B[i]` 都是一份 group 独有的权重矩阵。

### 24.3 `D`

输出是：

```python
d = torch.empty((m, n), ...)
```

见：
[`DeepGEMM/tests/generators.py:299`](</data/dnn/qinzq/repository/DeepGEMM/tests/generators.py:299>)

所以：

```text
D: [M, N], dtype=bf16
```

---

## 25. `grouped_layout` 的两种编码方式

### 25.1 `use_psum_layout=False`

此时 `grouped_layout` 的 shape 是：

```text
grouped_layout: [M], int32
```

生成器写法：

```python
grouped_layout[start: actual_end] = i
grouped_layout[actual_end: aligned_end] = -1
```

见：
[`DeepGEMM/tests/generators.py:305`](</data/dnn/qinzq/repository/DeepGEMM/tests/generators.py:305>)

含义是：

- 有效行直接写 group id
- padding 行写 `-1`

也就是逐行告诉 kernel：

- 这一行归哪个 group
- 这一行是不是 padding

### 25.2 `use_psum_layout=True`

此时 `grouped_layout` 的 shape 是：

```text
grouped_layout: [G], int32
```

生成器写法：

```python
grouped_layout[i] = actual_end
```

见：
[`DeepGEMM/tests/generators.py:303`](</data/dnn/qinzq/repository/DeepGEMM/tests/generators.py:303>)

它保存的是每个 group 的累计有效结束位置：

- group 0: `[0, grouped_layout[0])`
- group 1:
  `[align(grouped_layout[0], alignment), grouped_layout[1])`
- group 2:
  `[align(grouped_layout[1], alignment), grouped_layout[2])`

测试就是这样验证的：

[`DeepGEMM/tests/test_fp8_fp4.py:103`](</data/dnn/qinzq/repository/DeepGEMM/tests/test_fp8_fp4.py:103>)

这就是名字里 `psum_layout` 的来源，本质上是 prefix-sum 风格的 end offsets。

---

## 26. 量化粒度和 dtype

### 26.1 `A` 的 FP8 量化

`A` 在 Python 侧不是一个 tensor，而是：

```text
A = (a_data, a_scale)
```

常见形态：

- `a_data`: `[M, K]`, `torch.float8_e4m3fn`
- `a_scale`: scale tensor，shape 取决于量化方式

`per_token_cast_to_fp8()` 的实现是：

[`DeepGEMM/deep_gemm/utils/math.py:21`](</data/dnn/qinzq/repository/DeepGEMM/deep_gemm/utils/math.py:21>)

它按每一行的 `K` 维切块：

```text
gran_k_a = 128   # 当前 vllm FP4 专家路径
a_scale shape = [M, ceil(K / gran_k_a)]
```

量化公式：

```text
sf_A = amax(block) / 448
a_q  = cast_fp8(a / sf_A)
```

也就是：

- 量化粒度是“每行、每个 `gran_k_a` block”
- 当前 `vllm` FP4 专家路径显式传 `recipe_a=(1, 128)`

对应代码：
[`deep_gemm_moe.py:513`](</data/dnn/qinzq/repository/vllm/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py:513>)

### 26.2 `B` 的 FP4 量化

`B` 在 Python 侧同样是 tuple：

```text
B = (b_data, b_scale)
```

在 K-major grouped 权重场景下，生成器分配的是：

```text
b_data : [G, N, K/2], int8
b_scale: [G, N, ceil(K / gran_k_b)], float
```

见：
[`DeepGEMM/tests/generators.py:251`](</data/dnn/qinzq/repository/DeepGEMM/tests/generators.py:251>)

`per_token_cast_to_fp4()` 的实现是：

[`DeepGEMM/deep_gemm/utils/math.py:85`](</data/dnn/qinzq/repository/DeepGEMM/deep_gemm/utils/math.py:85>)

它的规则是：

```text
gran_k_b = 32    # 当前 vllm FP4 专家路径
sf_B = amax(block) / 6
```

然后把每个值量化到 FP4 E2M1，再把两个 FP4 码字打包到一个 `int8`：

```text
logical K elems -> physical K/2 bytes
```

这也是为什么 `vllm` 侧在调用前要显式：

```python
w1.view(torch.int8)
w2.view(torch.int8)
```

对应代码：
[`deep_gemm_moe.py:510`](</data/dnn/qinzq/repository/vllm/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py:510>)

当前 `vllm` FP4 专家路径显式传：

```text
recipe_b = (1, 32)
```

### 26.3 scale dtype

这里要区分“Python 侧原始 scale”和“kernel 真正吃到的 scale layout”：

- `per_token_cast_to_fp8 / fp4` 直接返回的 scale 是 `torch.float`
- API 入口会调用
  `transform_sf_pair_into_required_layout(...)`
  把 scale 重排到 kernel 要求的布局
- SM100 + FP8xFP4 路径下，host check 最终要求 `sfa.scalar_type() == torch::kInt`

也就是说：

- Python 调用者通常看到的是 `float` scale
- 进入 SM100 kernel 前，会被转换成更紧凑、适配硬件的数据布局

---

## 27. `vllm` 里的真实参数 shape 变化

### 27.1 FC1

`DeepGemmFP4Experts.forward()` 里第一处调用：

[`deep_gemm_moe.py:508`](</data/dnn/qinzq/repository/vllm/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py:508>)

```python
m_grouped_fp8_fp4_gemm_nt_contiguous(
    (a1q, a1q_scale),
    (w1.view(torch.int8), self.w1_scale),
    mm1_out,
    expert_ids,
    recipe_a=(1, 128),
    recipe_b=(1, 32),
)
```

shape 对应关系：

- `a1q`: `[M_sum, K]`, `float8_e4m3fn`
- `a1q_scale`: 逻辑上 `[M_sum, ceil(K / 128)]`
- `w1.view(torch.int8)`: `[G, N, K/2]`, `int8`
- `self.w1_scale`: 逻辑上 `[G, N, ceil(K / 32)]`
- `mm1_out`: `[M_sum, N]`, `bfloat16`
- `expert_ids`: 一般是 psum layout，shape `[G]`, `int32`

这里：

- `K` 是模型 hidden size
- `N` 是第一层 expert 线性层输出维度
- `M_sum` 是按 DeepGEMM 对齐后的 token-expert pair 总数

### 27.2 FC2

第二处调用：

[`deep_gemm_moe.py:528`](</data/dnn/qinzq/repository/vllm/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py:528>)

```python
m_grouped_fp8_fp4_gemm_nt_contiguous(
    (a2q, a2q_scale),
    (w2.view(torch.int8), self.w2_scale),
    mm2_out,
    expert_ids,
    recipe_a=(1, 128),
    recipe_b=(1, 32),
)
```

shape 变化为：

- `a2q`: `[M_sum, activation_out_dim]`, `float8_e4m3fn`
- `a2q_scale`: 逻辑上 `[M_sum, ceil(activation_out_dim / 128)]`
- `w2.view(torch.int8)`: `[G, K, activation_out_dim / 2]`, `int8`
- `self.w2_scale`: 逻辑上 `[G, K, ceil(activation_out_dim / 32)]`
- `mm2_out`: `[M_sum, K]`, `bfloat16`

这里第二次 GEMM 的“输入 K 维”已经从原始 hidden size 变成了 `activation_out_dim`。

---

## 28. 形状如何从 token 级输入变成 grouped GEMM

在 `vllm` 里，前面会先做一次按 expert 的重排：

[`deep_gemm_moe.py:498`](</data/dnn/qinzq/repository/vllm/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py:498>)

```python
a1q, a1q_scale, expert_ids, inv_perm = deepgemm_moe_permute(...)
```

这一步做了三件事：

1. 把原始 token 按 expert 分桶
2. 把每个 expert 的 token 行按 `M` 维连续拼起来
3. 生成 `expert_ids`，让 GEMM 知道每段行该匹配哪个 expert 权重

所以从上游视角看，shape 变化大致是：

```text
原始 hidden_states/topk
    -> 按 expert 展开后的 token-expert pairs
    -> 按 expert 连续拼接成 [M_sum, K]
    -> m_grouped_fp8_fp4_gemm_nt_contiguous
    -> [M_sum, N] / [M_sum, K]
```

---

## 29. kernel 内部可近似理解成什么

虽然 CUDA 主循环不会显式反量化成 BF16 再做 matmul，但数值语义可以理解成：

```text
D_i[r, c] =
Σ_t (
    A_q[r, t] * sf_A[r, floor(t / gran_k_a)]
) * (
    B_q[c, t] * sf_B[c, floor(t / gran_k_b)]
)
```

其中：

- `A_q` 是 FP8 数据
- `B_q` 是 FP4 E2M1 解码后的值
- `sf_A` 的粒度是每行每 `128` 列一组
- `sf_B` 的粒度是每行每 `32` 列一组

也就是说，这条路径不是“先完整反量化、再 GEMM”，而是“量化数据 + scale 在 Tensor Core 数据通路里边算边还原”。

---

## 30. 为什么 `A` 要做成 contiguous grouped layout

因为 MoE 场景下每个 expert 收到的 `m_i` 往往：

- 很小
- 不均匀
- 动态变化

如果每个 expert 单独发一个 GEMM：

- launch 开销大
- 小矩阵效率差
- 很难吃满硬件

把它们拼成一个 `[M, K]` 的大 `A` 后，可以：

- 一次 launch 处理多个 expert
- 让 `A` 的访存沿 `M` 维连续
- 用 `grouped_layout` 做轻量的 group 切换

这就是 `m_grouped` 和 `contiguous` 组合起来的核心价值。

---

## 31. 一张表总结

| 项 | 当前 FP4 专家路径的含义 |
|---|---|
| 数学形式 | 对每个 group 做 `D_i = A_i @ B_i^T` |
| 分组方向 | 沿 `M` 维分组 |
| `A` 逻辑 shape | `[M, K]` |
| `A` dtype | `float8_e4m3fn` |
| `A` scale shape | 逻辑上 `[M, ceil(K / 128)]` |
| `A` 量化粒度 | 每行、每 `128` 列一个 scale |
| `B` 逻辑 shape | `[G, N, K]` |
| `B` 物理数据 shape | `[G, N, K/2]` |
| `B` dtype | packed FP4，物理存储为 `int8` |
| `B` scale shape | 逻辑上 `[G, N, ceil(K / 32)]` |
| `B` 量化粒度 | 每行、每 `32` 列一个 scale |
| 输出 `D` | `[M, N]`, `bfloat16` |
| `grouped_layout` 非 psum | `[M]`，逐行 group id，padding 为 `-1` |
| `grouped_layout` psum | `[G]`，每组累计结束 offset |
| MoE 里的 group | local expert |

---

## 32. 最简结论

`m_grouped_fp8_fp4_gemm_nt_contiguous` 可以直接理解成：

1. 先把不同 expert 的 token 行拼成一个连续的 `A:[M,K]`
2. 每个 expert 保留自己的一份 `B_i:[N,K]`
3. 用 `grouped_layout` 描述 `A` 的每一段归属哪个 expert
4. 按 `A_i @ B_i^T` 的语义做 grouped GEMM
5. 其中 `A` 用 FP8，当前粒度是 `K` 维每 `128` 列一组
6. `B` 用 packed FP4，当前粒度是 `K` 维每 `32` 列一组
7. 输出 `D` 为 `bf16`

在 `vllm` 的 `DeepGemmFP4Experts` 里，这条算子正好对应两次专家线性层：

- FC1: `[M_sum, K] x [G, N, K]^T -> [M_sum, N]`
- FC2: `[M_sum, activation_out_dim] x [G, K, activation_out_dim]^T -> [M_sum, K]`
