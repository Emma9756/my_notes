---
title: "Compressor_ratio128_analysis"
date: 2026-05-28
tags:
  - #LLM
  - #deepseek
  - #笔记
  - #from_me
  - #待整理
status: 待整理
---

# `compress_ratio=128` 下 `Compressor / compress_kv` 详细解析

本文只聚焦 `compress_ratio=128` 时主 `Compressor` 的执行过程，也就是远历史 `compressed kv` 是如何生成的。

相关代码位置：

- `Compressor`：[inference/model.py](./inference/model.py#L279)
- `Attention.forward` 中调用 `Compressor`：[inference/model.py](./inference/model.py#L523)

## 1. 一句话概括

当 `compress_ratio=128` 时，`compress_kv` 的本质是：

```text
每 128 个 token 的 hidden state
-> 先映射成 128 份候选 kv 和 gating score
-> 再按通道做 softmax 加权池化
-> 压成 1 个 512 维 compressed kv
```

这里没有 `ratio=4` 时的 overlap，也没有 `Indexer` 的二次筛选。

## 2. 输入不是主 Attention 路径里的 `kv`

`Compressor.forward(x, start_pos)` 的输入是主干 hidden state：

- `x: [B, S, 7168]`

它不是直接压 `Attention` 主路径里已经算好的：

- `kv_main = self.wkv(x): [B, S, 512]`

而是自己内部重新生成一套压缩专用特征：

```python
kv = self.wkv(x)
score = self.wgate(x)
```

当 `compress_ratio=128` 时：

- `overlap = False`
- `coff = 1`

因此：

- `kv: [B, S, 512]`
- `score: [B, S, 512]`

这里的 `score` 不是 attention score，而是压缩池化时的 gating score。

## 3. `kv` 和 `score` 分别表示什么

可以这样理解：

- `kv[b, t, :]`
  - 第 `t` 个 token 提供给压缩器的候选内容向量
- `score[b, t, :]`
  - 第 `t` 个 token 在每个通道上应该占多大权重

这不是“每个 token 一个标量权重”的平均池化，而是逐通道的 gating：

- 不是“128 个 token 共用一组标量权重”
- 而是“对 512 个通道分别决定更偏向哪几个 token”

因此压缩器是 learned gated pooling，不是 average pooling。

## 4. prefill 时的压缩过程

当 `start_pos == 0` 时，走 prefill 分支。

### 4.1 第一步：判断当前轮能不能压

```python
should_compress = seqlen >= ratio
remainder = seqlen % ratio
cutoff = seqlen - remainder
```

当 `ratio=128` 时：

- `should_compress = (S >= 128)`
- `cutoff = S - (S mod 128)`

含义：

- 只有完整的 128-token group 才参与当前轮压缩
- 尾部不满 128 的 token 不会立即压缩

例如：

- `S=512` -> `cutoff=512, remainder=0`
- `S=300` -> `cutoff=256, remainder=44`
- `S=127` -> `cutoff=0, remainder=127`

### 4.2 第二步：保存尾部未凑满的 token

当 `remainder > 0` 时：

```python
kv, self.kv_state[:bsz, offset : offset+remainder] = kv.split([cutoff, remainder], dim=1)
self.score_state[:bsz, offset : offset+remainder] = score[:, cutoff:] + self.ape[:remainder]
score = score[:, :cutoff]
```

由于 `ratio=128` 时：

- `overlap=False`
- `offset=0`

所以这里的真实含义是：

- 前 `cutoff` 个 token 立刻参与压缩
- 后 `remainder` 个 token 缓存进 `kv_state / score_state`
- 等后面 token 到来凑满 128 再一起压

例如 `B=2, S=300`：

- 输入 `kv: [2, 300, 512]`
- `cutoff=256, remainder=44`

拆分后：

- 当前用于压缩的部分：`[2, 256, 512]`
- 缓存在状态里的尾巴：`kv_state[:2, :44]: [2, 44, 512]`

### 4.3 第三步：按 128 个 token 分组

```python
kv = kv.unflatten(1, (-1, ratio))
score = score.unflatten(1, (-1, ratio)) + self.ape
```

shape 变化：

- `kv: [B, cutoff, 512] -> [B, cutoff/128, 128, 512]`
- `score: [B, cutoff, 512] -> [B, cutoff/128, 128, 512]`

这里每个 group 对应一段连续的 128 个 token。

例如 `B=2, cutoff=512`：

- `kv: [2, 4, 128, 512]`
- `score: [2, 4, 128, 512]`

### 4.4 第四步：加入组内相对位置偏置 `ape`

`self.ape` 的 shape 是：

- `[128, 512]`

广播到每个 group 后，相当于：

- 对组内第 `0..127` 个相对位置
- 分别加一个可学习的 `512` 维位置偏置

因此这里的压缩 gate 实际上是：

```text
gate = 内容分数 + 组内相对位置偏置
```

### 4.5 第五步：对 128 个 token 做逐通道 softmax 池化

真正的压缩发生在：

```python
kv = (kv * score.softmax(dim=2)).sum(dim=2)
```

shape 变化：

- `kv: [B, G, 128, 512]`
- `score.softmax(dim=2): [B, G, 128, 512]`
- 输出 `kv_compress: [B, G, 512]`

其中：

- `G = cutoff / 128`

更细地说，对某个 batch、某个压缩组、某个通道 `d`：

```text
compressed_kv[d] =
sum_{i=0..127} softmax(score[i, d]) * kv[i, d]
```

这说明：

- 每个通道都会自己决定更依赖哪几个 token
- 不是简单平均
- 也不是整向量共用一组统一权重

### 4.6 一个完整的 prefill 例子

设：

- `B=2`
- `S=512`
- `ratio=128`

则：

1. 输入：
   - `x: [2, 512, 7168]`
2. 投影：
   - `kv = wkv(x): [2, 512, 512]`
   - `score = wgate(x): [2, 512, 512]`
3. 因为 `512 % 128 = 0`
   - `cutoff=512`
   - `remainder=0`
4. 分组：
   - `kv -> [2, 4, 128, 512]`
   - `score -> [2, 4, 128, 512]`
5. 池化：
   - `kv_compress: [2, 4, 512]`

含义：

- 原来每 128 个 token 为一组
- 每组压成 1 个 512 维向量
- 最终得到 4 个 compressed KV

## 5. decode 时的压缩过程

当 `start_pos > 0` 时，走 decode 分支。通常输入为：

- `x: [B, 1, 7168]`
- `kv: [B, 1, 512]`
- `score: [B, 1, 512]`

### 5.1 第一步：判断当前步是否凑满一个 128-token group

```python
should_compress = (start_pos + 1) % self.compress_ratio == 0
```

当 `ratio=128` 时：

- 只有每累计 128 个 token，才会真正生成 1 个 compressed KV

例如：

- `start_pos=126` -> 不压
- `start_pos=127` -> 生成第 0 个 compressed KV
- `start_pos=255` -> 生成第 1 个 compressed KV
- `start_pos=511` -> 生成第 3 个 compressed KV

### 5.2 第二步：先把当前 token 写入状态缓存

```python
self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
```

因此 `kv_state / score_state` 是一个长度为 128 的组缓存。

例如在 `start_pos=511` 时：

- `start_pos % 128 = 127`
- 当前 token 会写入第 127 个槽

这表示：

- 第 `384..511` 这 128 个 token 此时刚好凑满一组

### 5.3 第三步：decode 时 `ape` 的加入方式

在 decode 分支里，位置偏置不是先 reshape 再统一加，而是每来一个 token 单独加上：

```python
score += self.ape[start_pos % ratio]
```

也就是：

- 组内第 0 个 token 加 `ape[0]`
- 组内第 1 个 token 加 `ape[1]`
- ...
- 组内第 127 个 token 加 `ape[127]`

所以当一组 128 个 token 收齐后，`score_state` 内已经带好了整组的位置偏置。

### 5.4 第四步：凑满 128 个时做池化

只有当 `should_compress=True` 时，才真正压缩：

```python
kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
```

shape：

- `kv_state[:bsz]: [B, 128, 512]`
- `score_state[:bsz].softmax(dim=1): [B, 128, 512]`
- 输出 `kv_compress_new: [B, 1, 512]`

这和 prefill 的池化公式本质一致，只是：

- prefill 一次压很多组
- decode 每次最多只压 1 组

## 6. 压缩后的后处理

只要真的压出了 `kv_compress`，无论 prefill 还是 decode，后面都会做相同的后处理。

### 6.1 RMSNorm

```python
kv = self.norm(kv.to(dtype))
```

shape 不变：

- prefill：`[B, G, 512]`
- decode：`[B, 1, 512]`

### 6.2 对最后 64 维加 RoPE

```python
if start_pos == 0:
    freqs_cis = self.freqs_cis[:cutoff:ratio]
else:
    freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
apply_rotary_emb(kv[..., -rd:], freqs_cis)
```

其中：

- `rd = 64`

所以：

- 前 `448` 维是 non-rope 部分
- 后 `64` 维是 rope 部分

prefill 时：

- 每个 compressed group 取一个按 `128` 步采样的位置编码

decode 时：

- 新压出来的这个 group 只取 1 个对应位置的 rope

### 6.3 只量化 non-rope 的前 448 维

```python
act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
```

即：

- `kv[..., :448]` 做量化模拟
- `kv[..., 448:512]` 保持 bf16，以保留位置精度

### 6.4 写入 compressed cache

```python
if start_pos == 0:
    self.kv_cache[:bsz, :seqlen // ratio] = kv
else:
    self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
```

因此：

- prefill 时，一次把当前完整 chunk 中压出来的所有 compressed KV 顺序写入 cache
- decode 时，每凑满 128 个 token，只新增写入 1 个 compressed KV

## 7. 本质总结

`compress_ratio=128` 时的 `compress_kv` 本质上是：

1. 从 hidden state `x` 重新生成压缩专用的 `kv` 与 `score`
2. 每 128 个 token 分成一组
3. 对 512 个通道分别做 softmax gating 池化
4. 每组只保留 1 个 `512` 维 compressed KV
5. 再做 norm、RoPE、量化模拟并写入 compressed cache
