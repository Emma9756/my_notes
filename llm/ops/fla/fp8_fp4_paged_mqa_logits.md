---
title: fp8_fp4_paged_mqa_logits
date: 2026-05-27
tags:
  - fla
  - ds_v4
status: 更新中
---

## Intro

请解释fp8_fp4_paged_mqa_logits这个函数的运算过程，含各个参数的数据类型和shape变化

https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/utils/deep_gemm.py#L407

下面按 vLLM `v0.21.0` 的 wrapper 来解释。这个 Python 函数本身**不做显式 reshape/转置**，主要是做 lazy init、检查 DeepGEMM 后端符号是否存在，然后把参数原样转交给 DeepGEMM 的 CUDA kernel。源码里就是 `_lazy_init()` 后调用 `_fp8_fp4_paged_mqa_logits_impl(...)`。([vLLM][1])

## 1. 它计算什么

`fp8_fp4_paged_mqa_logits` 用 paged KV cache 计算 MQA logits，本质上是对每个 query token、每个历史 KV token 计算类似：

$$\text{logit}_{b,t,s}
=
\text{kv\_scale}_{b,s}
\cdot
\sum_{h=0}^{H-1}
\text{weight}_{b,t,h}
\cdot
\max\left(0,\left\langle Q_{b,t,h,:}, K_{b,s,:}\right\rangle\right)$$

也就是 `Q @ K^T`，然后做 ReLU、乘每个 head 的权重、再乘 KV scale。DeepGEMM 的文档也把这个逻辑描述为：先做矩阵乘 `Q @ K^T`，再 ReLU，再乘 head weight，最后乘 KV scale。([DeepWiki][2])

这里叫 MQA，因为 K/V 是单 KV head 或共享 KV 表示，而 Q 仍然有多个 query heads：`q` shape 里有 `H`，`kv_cache` 里没有 `H` 维度。

## 2. 参数 dtype 和 shape

| 参数                  |                      dtype / 类型 |                                                                                                         shape | 作用                                                                                              |
| ------------------- | ------------------------------: | ------------------------------------------------------------------------------------------------------------: | ----------------------------------------------------------------------------------------------- |
| `q`                 | `tuple[Tensor, Tensor \| None]` | FP8 路径：`q_values=[B,next_n,H,D]`, `q_scale=None`；FP4 路径：`q_values` 是 packed `uint8`，`q_scale` 是对应 block scale | Query。wrapper 统一 FP8 / MXFP4 调度；FP8 时 scale 不单独传，FP4 时需要 companion scale。([vLLM][1])            |
| `kv_cache`          |                   `torch.uint8` |                                                                     FP8 布局：`[num_blocks, block_size, 1, D+4]` | Paged KV cache。前 `D` 字节是 FP8 K，最后 4 字节存该 token 的 float dequant scale。([vLLM][1])                |
| `weights`           |                 `torch.float32` |                                                                                             `[B * next_n, H]` | 每个 query row、每个 head 的 ReLU 后加权系数。([vLLM][1])                                                   |
| `context_lens`      |                  `int32 Tensor` |                                                                                                         `[B]` | 每个 batch 序列有效 KV 长度。([vLLM][1])                                                                 |
| `block_tables`      |                  `int32 Tensor` |                                                                                             `[B, max_blocks]` | logical block id → physical block id 的映射。([vLLM][1])                                            |
| `schedule_metadata` |            Tensor，通常 int32 后端格式 |                                                                                              backend-specific | 由 `get_paged_mqa_logits_metadata(context_lens, block_size, num_sms)` 生成，用于把工作分配到 SM。([vLLM][1]) |
| `max_model_len`     |                    Python `int` |                                                                                                            标量 | 输出 logits 的列数。                                                                                  |
| `clean_logits`      |                   Python `bool` |                                                                                                            标量 | 是否把未填充位置清成 `-inf`。([vLLM][1])                                                                   |

返回值：

```python
logits: torch.Tensor  # dtype=torch.float32
shape = [B * next_n, max_model_len]
```

vLLM 文档明确说明返回 shape 是 `[B * next_n, max_model_len]`，dtype 是 `torch.float32`。([vLLM][1])

## 3. shape 如何变化

最重要的 shape 变化是：`q` 的前两维 `[B,next_n]` 被逻辑展平为输出的行维。

```text
q_values:  [B, next_n, H, D]
weights:   [B * next_n, H]

逻辑 query row:
m = b * next_n + t

输出:
logits:    [B * next_n, max_model_len]
```

对每个 `m = b * next_n + t`，kernel 只对该 batch 的有效历史长度 `context_lens[b]` 计算 logits。`max_model_len` 只是输出 buffer 的固定列宽；如果 `clean_logits=True`，超出有效长度或未填充位置会被置为 `-inf`。

KV cache 的分页访问过程是：

```text
s: 逻辑 KV token 位置, 0 <= s < context_lens[b]

logical_block = s // block_size
offset        = s % block_size
physical_block = block_tables[b, logical_block]

K bytes + scale = kv_cache[physical_block, offset, 0, :]
```

DeepGEMM 的 paged MQA 说明也指出，paged variant 通过 `block_table` 和 `context_lens` 处理非连续 KV cache，并用调度器把 GPU block 映射到 batch 和 KV block。([DeepWiki][2])

## 4. 逐步运算过程

**第一步：初始化和分发。**
Python wrapper 调 `_lazy_init()`，确认 DeepGEMM 里有 `fp8_fp4_paged_mqa_logits` 符号，然后把 `q, kv_cache, weights, context_lens, block_tables, schedule_metadata, max_model_len, clean_logits` 原样传给底层 kernel。vLLM 在 lazy init 时把 DeepGEMM 的 `fp8_fp4_paged_mqa_logits` 绑定到 `_fp8_fp4_paged_mqa_logits_impl`。([vLLM][1])

**第二步：调度 paged KV 工作。**
`schedule_metadata` 决定每个 SM 处理哪些 `(batch, query atom, KV block range)`。DeepGEMM 文档提到 paged kernel 使用 `PagedMQALogitsScheduler` 映射 GPU blocks，并支持把 KV 维度拆分到多个 SM 来提高并行度。([DeepWiki][2])

**第三步：加载 Q、K、scale、weights。**
对一个 query row `m=b*next_n+t`，kernel 加载：

```text
Q[m]         -> [H, D]
weights[m]   -> [H]
K[b, s]      -> [D]  # 从 paged kv_cache 通过 block_tables 找到
kv_scale[s]  -> scalar float
```

FP8 KV cache 中，`D+4` 的最后 4 字节就是该 KV token 的反量化 scale。([vLLM][1])

**第四步：低精度 dot + FP32 epilogue。**
每个 head 先做点积：

```text
score_h = dot(Q[b,t,h,:], K[b,s,:])
```

然后：

```text
weighted_h = max(score_h, 0) * weights[b*next_n+t, h]
```

最后跨 head 求和并乘 KV scale：

```text
logits[b*next_n+t, s] =
    kv_scale[b,s] * sum_h weighted_h
```

DeepGEMM 文档说明这个 epilogue 在 math warps 中以高精度执行，再写回全局内存。([DeepWiki][2])

**第五步：写出固定宽度 logits。**

```text
有效位置:    s < context_lens[b]  写实际 logit
无效/未填充: s >= context_lens[b]  clean_logits=True 时写 -inf
输出 shape: [B * next_n, max_model_len]
```

## 5. 一个具体 shape 例子

假设：

```text
B = 4
next_n = 2
H = 128
D = 128
block_size = 64
max_model_len = 163840
```

那么：

```text
q_values:     [4, 2, 128, 128]      float8_e4m3fn  # FP8 path
q_scale:      None

weights:      [8, 128]              float32
context_lens: [4]                   int32
block_tables: [4, max_blocks]       int32
kv_cache:     [num_blocks, 64, 1, 132] uint8  # 128 FP8 bytes + 4 scale bytes

output:       [8, 163840]           float32
```

其中输出第 `m=5` 行对应：

```text
b = 5 // 2 = 2
t = 5 % 2  = 1
```

也就是 batch 2 的第 2 个 speculative / next token 的所有历史 KV logits。

## 6. 注意点

`q_scale` 的 shape 在 wrapper 文档里没有展开，只说 FP4 路径下它是 packed `q_values` 的 companion block-scale tensor；所以精确 FP4 scale layout 要看 DeepGEMM MXFP4 packer / kernel 约定，而不是这个 vLLM wrapper 本身。wrapper 对 FP8 路径明确要求 `q_scale=None`，对 FP4 路径则把 `(q_values, q_scale)` 直接传入底层。([vLLM][1])

此外，vLLM 的 indexer 代码里有一个重要限制：FP4 indexer cache 要求 Blackwell datacenter GPU；并且 `next_n` 原生支持列表是 `[1, 2]`，其它 `next_n` 会走 flattening path。([github.com][3])

[1]: https://docs.vllm.ai/en/latest/api/vllm/utils/deep_gemm/ "deep_gemm - vLLM"
[2]: https://deepwiki.com/deepseek-ai/DeepGEMM/4.3-mqa-kernel-implementation "MQA Kernel Implementation | deepseek-ai/DeepGEMM | DeepWiki"
[3]: https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/v1/attention/backends/mla/indexer.py "vllm/vllm/v1/attention/backends/mla/indexer.py at v0.21.0 · vllm-project/vllm · GitHub"
