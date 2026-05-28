---
title: "Indexer_ratio4_analysis"
date: 2026-05-28
tags:
  - #LLM
  - #deepseek
  - #笔记
  - #from_me
  - #待整理
status: 待整理
---

# `compress_ratio=4` 下 `Indexer` 详细解析

本文只聚焦 `compress_ratio=4` 时 `Indexer` 的执行过程，也就是它如何从压缩后的历史 KV 中选出最值得看的压缩块。

相关代码位置：

- `Indexer`：[inference/model.py](./inference/model.py#L380)
- `Attention.forward` 中调用 `Indexer`：[inference/model.py](./inference/model.py#L508)

## 1. 一句话概括

当 `compress_ratio=4` 时，主 `Compressor` 先把历史压成 `compressed kv`，而 `Indexer` 的职责不是再生成主 attention 用的 KV，而是：

```text
把压缩后的历史块当作候选库
-> 用一套小维度 query 去打分
-> 再从这些压缩块里选出最值得看的 top-k 位置
```

所以：

- 主 `Compressor` 负责“压缩”
- `Indexer` 负责“检索和选点”

## 2. 为什么只有 `ratio=4` 才有 `Indexer`

在 [inference/model.py](./inference/model.py#L466)：

```python
if self.compress_ratio:
    self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
    if self.compress_ratio == 4:
        self.indexer = Indexer(args, self.compress_ratio)
    else:
        self.indexer = None
```

原因是：

- `ratio=4` 压缩得比较细
- 压完后的压缩块数仍然很多
- 如果这些压缩块全部参与 attention，稀疏性不够

所以还需要一层 learned selection。

而 `ratio=128` 压缩块本身已经很少，因此不再需要 `Indexer`。

## 3. `Indexer` 的输入输出

`Indexer.forward` 的签名是：

```python
def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
```

输入有 4 个：

- `x: [B, S, 7168]`
  - 主干 hidden state
- `qr: [B, S, 1536]`
  - 主 attention Q 路径中的低秩 query 表示
- `start_pos`
  - 当前是 prefill 还是 decode，以及对应的全局位置
- `offset`
  - 压缩块在最终 attention 索引空间中的起始偏移

输出只有一个：

- `topk_idxs: [B, S, Kc]`

这里的 `Kc` 表示压缩历史候选数，后面会与窗口索引拼接：

```python
topk_idxs = torch.cat([window_topk_idxs, compress_topk_idxs], dim=-1)
```

因此 `Indexer` 输出的不是新的 KV，也不是最终 attention 分数，而是：

- 压缩区里“应该访问哪些位置”的索引

## 4. `Indexer` 的内部参数

初始化代码在 [inference/model.py](./inference/model.py#L384)：

```python
self.n_heads = args.index_n_heads
self.n_local_heads = args.index_n_heads // world_size
self.head_dim = args.index_head_dim
self.rope_head_dim = args.rope_head_dim
self.index_topk = args.index_topk
self.q_lora_rank = args.q_lora_rank
self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
self.weights_proj = ColumnParallelLinear(self.dim, self.n_heads, dtype=torch.bfloat16)
self.compressor = Compressor(args, compress_ratio, self.head_dim, True)
self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len // compress_ratio, self.head_dim), persistent=False)
```

这里最关键的几点：

- `Indexer` 有自己独立的 head 数和 head_dim
- 它的 `head_dim` 不是主 attention 的 `512`，而是更小的 `128`
- 它有自己的一套 `Compressor`
- 它也有自己的一套压缩缓存：
  - `Indexer.kv_cache: [B, T, 128]`

所以 `Indexer` 是一个“小维度检索器”，不是直接复用主 attention 的大维 `compressed kv`。

## 5. `Indexer` 和主 `Compressor` 的关系

这两者最容易混淆。

主 `Attention` 里有两套压缩逻辑并行存在：

1. 主 `Compressor`
   - 生成真正给主 `sparse_attn` 使用的：
   - `compressed kv: [B, T, 512]`
2. `Indexer` 内部自己的 `Compressor`
   - 生成只用于检索评分的：
   - `compressed kv: [B, T, 128]`

所以可以理解成：

- 大维 `512` 版本：给主 attention 用
- 小维 `128` 版本：给 `Indexer` 检索用

## 6. `Indexer` 的 query 是怎么来的

在 `Attention.forward` 中，主 Q 路径先得到：

```python
qr = q = self.q_norm(self.wq_a(x))
```

因此：

- `qr: [B, S, 1536]`

然后传给 `Indexer`。

在 `Indexer.forward` 中：

```python
q = self.wq_b(qr)
q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
apply_rotary_emb(q[..., -rd:], freqs_cis)
q = rotate_activation(q)
fp4_act_quant(q, fp4_block_size, True)
```

shape 变化：

1. `wq_b`
   - `[B, S, 1536] -> [B, S, 64*128] = [B, S, 8192]`
2. reshape
   - `q: [B, S, 64, 128]`
3. 对最后 `64` 维做 RoPE
4. Hadamard rotate
5. FP4 量化模拟

所以 `Indexer` 的 query 本质上是：

- 主低秩 query `qr`
- 再投影成一套小维检索 query

## 7. `Indexer` 的压缩 KV 是怎么来的

在 `Indexer.forward` 中：

```python
if self.compressor.kv_cache is None:
    self.compressor.kv_cache = self.kv_cache
    self.compressor.freqs_cis = self.freqs_cis
...
self.compressor(x, start_pos)
```

也就是说，`Indexer` 会调用自己内部那套：

- `Compressor(args, compress_ratio=4, head_dim=128, rotate=True)`

因此它生成的是：

- `Indexer.kv_cache: [B, T, 128]`

其中：

- prefill 时：`T = floor(S / 4)` 或当前累计长度
- decode 时：每 4 个 token 新增 1 个压缩块

由于这里 `rotate=True`，所以这套压缩表示还会做：

- Hadamard rotate
- FP4 量化模拟

这说明 `Indexer` 更偏向“高效检索表示”，而不是高保真的主 attention 表示。

## 8. 真正的打分过程

核心代码：

```python
weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
```

### 8.1 先算 query 和每个压缩块的相关性

`einsum("bshd,btd->bsht", ...)` 的 shape：

- `q: [B, S, 64, 128]`
- `kv_cache: [B, T, 128]`
- 输出：
  - `index_score_raw: [B, S, 64, T]`

含义是：

- 对每个 query token
- 对每个 index head
- 对每个压缩块
- 都算一个相关性分数

### 8.2 再给不同 index head 加权

`weights_proj(x)`：

- `[B, S, 7168] -> [B, S, 64]`

表示每个 token 对 64 个 index head 的偏好权重。

然后：

```python
(index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
```

shape：

- `relu(index_score_raw): [B, S, 64, T]`
- `weights.unsqueeze(-1): [B, S, 64, 1]`
- 对 head 维求和后：
  - `index_score: [B, S, T]`

因此最终每个 query token 对每个压缩块只保留 1 个总分。

## 9. 为什么先做 `relu`

代码里：

```python
index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
```

这表示负相关会被直接截断，不再参与候选排序。

直观上可以理解成：

- `Indexer` 更关心“哪些压缩块和当前 query 有正向匹配”
- 而不是把负相关也带进 ranking

所以它更像一个召回相关块的 retriever。

## 10. prefill 时的 masking

当 `start_pos == 0` 时：

```python
mask = torch.arange(seqlen // ratio).repeat(seqlen, 1) >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
index_score += torch.where(mask, float("-inf"), 0)
```

作用是因果约束。

因为 prefill 一次会同时处理整段序列，而压缩块对应的是整组 token。对某个 query token 来说：

- 不能访问未来才形成的压缩块

所以这些非法压缩块会被置成 `-inf`。

随后 topk 之后又做了一次合法性处理：

```python
mask = topk_idxs >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
topk_idxs = torch.where(mask, -1, topk_idxs + offset)
```

含义：

- 非法压缩块索引改成 `-1`
- 合法压缩块索引再加上 `offset`

## 11. decode 时的行为

在 decode 中：

- 通常 `S=1`
- `end_pos = start_pos + 1`
- 可用压缩块数是 `end_pos // 4`

然后直接：

```python
topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
topk_idxs += offset
```

因为 decode 时因果性已经天然满足：

- 只能看到当前 cache 中已有的压缩块
- 不需要像 prefill 那样对整段序列再额外 mask

## 12. `offset` 的作用

`Indexer` 选出来的原始 top-k 下标，是相对于它自己的：

- `Indexer.kv_cache[:bsz, :T]`

也就是：

- `0..T-1`

但主 `sparse_attn` 用的索引空间不是这个，而是主 attention 的 `kv` 索引空间。

因此必须加偏移：

- prefill 时：`offset = seqlen`
  - 因为主 `kv` 是 `[local kv | kv_compress]`
- decode 时：`offset = win = 128`
  - 因为主 `kv_cache` 是 `[window cache | compressed cache]`

这样 `compress_topk_idxs` 才能和 `window_topk_idxs` 拼接成同一个索引空间里的 `topk_idxs`。

## 13. 一个具体例子

设：

- `B=2`
- `S=256`
- `compress_ratio=4`

### 13.1 prefill

- `qr: [2, 256, 1536]`
- `q_index: [2, 256, 64, 128]`
- `Indexer.kv_cache: [2, 64, 128]`

相关性计算：

- `[2, 256, 64, 128] x [2, 64, 128] -> [2, 256, 64, 64]`

head 汇总后：

- `[2, 256, 64]`

topk 后：

- `compress_topk_idxs: [2, 256, 64]`

由于当前只有 `64` 个压缩块，而 `index_topk=1024`，所以这里实际上是：

- 把 64 个压缩块全取出来
- 只是顺序由 learned ranking 决定

### 13.2 decode，设 `start_pos=255`

- `q_index: [2, 1, 64, 128]`
- `Indexer.kv_cache: [2, 64, 128]`

相关性：

- `[2, 1, 64, 128] x [2, 64, 128] -> [2, 1, 64, 64]`

head 汇总后：

- `[2, 1, 64]`

topk 后：

- `[2, 1, 64]`

然后统一加 `offset=128`，就映射到主 cache 的压缩区位置。

## 14. 本质总结

`compress_ratio=4` 时，`Indexer` 本质上是一个小维度 learned retriever：

1. 用 `qr` 生成检索 query
2. 用自己那套 `128` 维压缩器生成检索用 compressed KV
3. 计算 query 与每个压缩块的相关性
4. 用 head 权重汇总成每个压缩块的总分
5. 取 top-k，输出压缩区索引
6. 再与窗口索引拼接成最终 `topk_idxs`

所以它不是 attention 本体，而是进入 `sparse_attn` 之前的候选召回器。
