---
title: "Attention_prefill_decode"
date: 2026-05-28
tags:
  - #LLM
  - #flash_attention
  - #deepseek
  - #笔记
  - #from_me
  - #待整理
status: 待整理
---

# Attention 中 `sparse_attn` 的 `prefill` / `decode` 计算过程

本文只聚焦 `compress_ratio=4` 时，`Attention.forward` 里 `sparse_attn` 的执行过程。

相关代码位置：

- `Attention.forward`：[inference/model.py](./inference/model.py#L489)
- `sparse_attn_kernel`：[inference/kernel.py](./inference/kernel.py#L277)
- `sparse_attn` 包装函数：[inference/kernel.py](./inference/kernel.py#L355)

## 1. 前置背景

当 `compress_ratio=4` 时，`Attention` 不是对完整历史直接做全量注意力，而是先构造两类候选：

1. 局部窗口候选
   - 来自最近 `window_size=128` 个 token 的原始 `local kv`
2. 压缩历史候选
   - 每 `4` 个 token 压成 `1` 个 compressed KV
   - 再由 `Indexer` 从这些压缩块里选出要看的位置

最终 `topk_idxs` 由两部分拼接得到：

```text
topk_idxs = [window_topk_idxs | compress_topk_idxs]
```

然后把 `q`、`kv`、`topk_idxs` 送入 `sparse_attn`。

## 2. `prefill` 的 `sparse_attn` 计算过程

`prefill` 对应 [inference/model.py](./inference/model.py#L528)：

```python
o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
```

这里的 `kv` 不是总缓存，而是当前轮临时拼出来的：

```text
kv = [local kv | kv_compress]
```

### 2.1 `prefill` 进入 `sparse_attn` 前的输入

用一个具体例子说明：

- `B=2`
- `S=256`
- `compress_ratio=4`

则在进入 `sparse_attn` 前，典型 shape 是：

- `q: [2, 256, 128, 512]`
- `local kv: [2, 256, 512]`
- 主 `Compressor` 输出 `kv_compress: [2, 64, 512]`
- 拼接后 `kv: [2, 320, 512]`
- `window_topk_idxs: [2, 256, 128]`
- `compress_topk_idxs: [2, 256, 64]`
- `topk_idxs: [2, 256, 192]`
- `attn_sink: [128]`

这里 `kv` 的索引空间是：

- `0..255`：当前 chunk 的原始 token 级 `local kv`
- `256..319`：压缩后的 `compressed kv`

因此 `topk_idxs[b, s, :]` 的每个元素，都是在这个长度为 `320` 的第二维上取位置。

### 2.2 `prefill` 中 `sparse_attn` 的外层逻辑

包装函数在 [inference/kernel.py](./inference/kernel.py#L355)：

```python
o = torch.empty_like(q)
kernel = sparse_attn_kernel(q.size(2), d, softmax_scale)
kernel(q, kv, o, attn_sink, topk_idxs)
```

输出 shape 与 `q` 相同：

- `o: [B, S, H, D]`

在本例中：

- `o: [2, 256, 128, 512]`

### 2.3 kernel 的并行粒度

kernel 在 [inference/kernel.py](./inference/kernel.py#L294)：

```python
with T.Kernel(m, b, threads=threads) as (bx, by):
```

这里：

- `m = S`
- `b = B`

所以每个 kernel 实例处理一个固定的：

- `batch = by`
- `query_pos = bx`

也就是：

```text
一次处理一个 query token，
但同时处理该 token 的全部 attention heads
```

### 2.4 第一步：把当前 query token 的所有 head 读入

```python
T.copy(q[by, bx, :, :], q_shared)
```

得到：

- `q_shared: [H, D]`

本例中：

- `q_shared: [128, 512]`

### 2.5 第二步：按 `topk_idxs` 分块 gather 候选 KV

kernel 固定：

- `block = 64`

所以如果 `topk=192`，就分 3 轮处理：

1. 第 1 轮处理 `0..63`
2. 第 2 轮处理 `64..127`
3. 第 3 轮处理 `128..191`

先取这一轮的候选索引：

```python
idxs[i] = topk_idxs[by, bx, t * block + i]
```

得到：

- `idxs: [64]`

再从 `kv` 中 gather：

```python
kv_shared[i, j] = kv[by, idxs[i], j]
```

得到：

- `kv_shared: [64, 512]`

含义是：

- 当前 query token 在这一轮只看 64 个候选位置
- 每个候选位置对应一个 `512` 维 latent KV

如果某个索引是 `-1`，就视为无效位置：

- 对应 `kv_shared` 填 0
- 对应 score 置成 `-inf`

### 2.6 第三步：计算 `q · k`

代码：

```python
T.gemm(q_shared, kv_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
```

shape 为：

- `q_shared: [H, D]`
- `kv_shared: [64, D]`
- 输出 `acc_s: [H, 64]`

即：

```text
acc_s[h, j] = dot(q_head_h, kv_candidate_j)
```

这里的 `kv_shared` 同时扮演 `K` 和 `V`：

- 先作为 `K` 参与打分
- 后续又作为 `V` 参与加权求和

这正是这里 MLA 风格共享 latent KV 的特征。

### 2.7 第四步：做在线 softmax

先乘缩放：

```python
acc_s[i, j] *= scale
```

随后进入在线 softmax 累积逻辑。关键变量：

- `scores_max`
  - 到当前为止每个 head 见过的最大分数
- `scores_max_prev`
  - 上一轮 block 结束时的最大分数
- `scores_scale`
  - 旧累积需要乘的修正系数
- `scores_sum`
  - 当前 block 的指数和
- `sum_exp`
  - 截至当前 block 的总 softmax 分母

代码：

```python
T.copy(scores_max, scores_max_prev)
T.reduce_max(acc_s, scores_max, dim=1, clear=False)
for i in T.Parallel(h):
    scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
for i, j in T.Parallel(h, block):
    acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
T.reduce_sum(acc_s, scores_sum, dim=1)
for i in T.Parallel(h):
    sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]
```

这意味着：

1. 当前 block 先更新每个 head 的全局最大值
2. 旧 block 的累积结果按新的最大值重标定
3. 当前 block 计算 `exp(score - max)`
4. 更新整体 softmax 分母

因此不需要显式存完整的 `[H, topk]` 分数矩阵。

### 2.8 第五步：在线累计输出

代码：

```python
T.copy(acc_s, acc_s_cast)
for i, j in T.Parallel(h, d):
    acc_o[i, j] *= scores_scale[i]
T.gemm(acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
```

shape 为：

- `acc_s_cast: [H, 64]`
- `kv_shared: [64, 512]`
- `acc_o: [H, 512]`

含义是：

```text
acc_o[h] += sum_j softmax_numerator[h, j] * kv_shared[j]
```

同时旧的 `acc_o` 也要先乘 `scores_scale`，因为在线 softmax 更新了归一化基准。

### 2.9 第六步：加入 `attn_sink`

所有 block 处理完后：

```python
sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
for i, j in T.Parallel(h, d):
    acc_o[i, j] /= sum_exp[i]
```

这表示每个 head 的 softmax 分母里还额外加了一个可学习 sink 项。

它的特点是：

- 只影响 softmax 分母
- 不对应具体 value 向量
- 会分走一部分注意力质量

### 2.10 第七步：写回输出

```python
T.copy(acc_o, o_shared)
T.copy(o_shared, o[by, bx, :, :])
```

于是当前 query token 的输出是：

- `[H, D]`

整个 `prefill` 完成后：

- `o: [2, 256, 128, 512]`

## 3. `decode` 的 `sparse_attn` 计算过程

`decode` 对应 [inference/model.py](./inference/model.py#L533)：

```python
o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
```

这里和 `prefill` 最大的区别是：

- `prefill` 传入的是本轮临时拼出来的 `kv`
- `decode` 传入的是累计好的总缓存 `self.kv_cache`

因此 `decode` 更准确地说是：

```text
先写 cache，
再对 cache 做 sparse attention
```

### 3.1 `decode` 进入 `sparse_attn` 前的输入

用一个具体例子说明：

- `B=2`
- `S=1`
- `start_pos=255`
- `compress_ratio=4`

先得到当前 token 的 query 和 local KV：

- `q: [2, 1, 128, 512]`
- `kv_current: [2, 1, 512]`

然后当前 token 的 `local kv` 写入窗口缓存：

```python
self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
```

此时窗口区逻辑长度仍是：

- `128`

接着主 `Compressor` 更新压缩状态。

由于：

```text
(255 + 1) % 4 == 0
```

所以这一步会新生成一个 compressed KV，并写入压缩区。于是这一时刻的压缩缓存逻辑长度为：

- `64`

同时 `Indexer` 产出：

- `window_topk_idxs: [2, 1, 128]`
- `compress_topk_idxs: [2, 1, 64]`
- `topk_idxs: [2, 1, 192]`

### 3.2 `decode` 时 `kv_cache` 的索引空间

`decode` 传入 kernel 的 `kv` 是：

- `self.kv_cache[:bsz]`

其逻辑结构分成两段：

1. `0..127`
   - 窗口区，存最近 `128` 个 token 的原始 `local kv`
2. `128..191`
   - 压缩区，存当前已有的 `64` 个 compressed KV

所以这一时刻 kernel 实际看到的输入可视为：

- `kv: [2, 192, 512]`

这也是为什么 `decode` 下 `compress_topk_idxs` 要统一加 `offset=128`，因为压缩区在总 cache 中固定从窗口区之后开始。

### 3.3 `decode` 中 `sparse_attn` 的输入

因此本例进入 kernel 时：

- `q: [2, 1, 128, 512]`
- `kv: [2, 192, 512]`
- `topk_idxs: [2, 1, 192]`
- `attn_sink: [128]`

这里 `topk_idxs[0, 0, :]` 的值会落在两个区间：

- 窗口候选：`0..127`
- 压缩候选：`128..191`

### 3.4 kernel 的并行粒度

与 `prefill` 相同，kernel 仍然是：

```python
with T.Kernel(m, b, threads=threads) as (bx, by):
```

但在 `decode` 中：

- `m = 1`

因此每个 batch 只处理当前这一个 query token。

### 3.5 第一步：读出当前 token 的 query

```python
T.copy(q[by, bx, :, :], q_shared)
```

得到：

- `q_shared: [128, 512]`

### 3.6 第二步：按 `topk_idxs` 从总 cache 中 gather 候选

仍然固定按 `block=64` 分块处理。

如果 `topk=192`，就仍然是 3 轮，每轮：

1. 从 `topk_idxs` 取 64 个候选位置
2. 从 `self.kv_cache[:bsz]` 中 gather 出：
   - `kv_shared: [64, 512]`

这一步和 `prefill` 的唯一区别是：

- `prefill` gather 的源是本轮临时拼接的 `kv`
- `decode` gather 的源是持久化总缓存 `kv_cache`

### 3.7 第三步：计算 `q · k`

代码完全相同：

```python
T.gemm(q_shared, kv_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
```

得到：

- `acc_s: [128, 64]`

即：

```text
acc_s[h, j] = dot(q_head_h, kv_candidate_j)
```

### 3.8 第四步：在线 softmax

`decode` 与 `prefill` 在 kernel 内部没有分支，仍然是同一套在线 softmax 更新：

1. 当前 block 更新 `scores_max`
2. 计算 `scores_scale`
3. 当前 block 做 `exp(score - max)`
4. 更新 `sum_exp`

因此从数学形式上说，`decode` 和 `prefill` 的 softmax 完全一致。

### 3.9 第五步：在线累计输出

仍然是：

```python
acc_o *= scores_scale
T.gemm(acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
```

得到：

- `acc_o: [128, 512]`

也就是当前新 token 在所有 head 上的注意力输出。

### 3.10 第六步：加入 `attn_sink`

最终和 `prefill` 一样：

```python
sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
acc_o[i, j] /= sum_exp[i]
```

所以 `attn_sink` 在 `decode` 中的作用没有变化，仍然只是修正 softmax 分母。

### 3.11 第七步：写回输出

最终写回：

- `o: [2, 1, 128, 512]`

之后再由 `Attention.forward` 继续做 inverse rotary 和输出投影，回到 hidden size。

## 4. `prefill` 与 `decode` 的核心区别

如果只看 `sparse_attn_kernel` 本身，两者几乎一样。真正不同的是 kernel 之外的数据组织方式。

### 4.1 `prefill`

执行顺序更像：

```text
x
-> q
-> local kv
-> main compressor 生成 kv_compress
-> Indexer 生成 compress_topk_idxs
-> kv = [local kv | kv_compress]
-> sparse_attn(q, kv, topk_idxs)
```

特点：

- `kv` 是本轮现算现拼的临时张量
- `topk_idxs` 对这个临时张量索引

### 4.2 `decode`

执行顺序更像：

```text
new token
-> q / local kv
-> 写入窗口 cache
-> 必要时生成一个新的 compressed kv
-> Indexer 生成 compress_topk_idxs
-> sparse_attn(q, kv_cache, topk_idxs)
```

特点：

- `kv` 是累计好的总缓存
- `topk_idxs` 对总 cache 索引

## 5. 一句话总结

`prefill` 和 `decode` 的 `sparse_attn` 内核计算方式是相同的：

1. 根据 `topk_idxs` gather 候选 KV
2. 分块做 `q · k`
3. 用在线 softmax 累积分母
4. 用同一份 KV 做加权求和
5. 加入 `attn_sink`
6. 输出 `[B, S, H, D]`

二者真正的差别只在于：

- `prefill` 面向本轮临时拼出的 `[local kv | compressed kv]`
- `decode` 面向已经维护好的 `kv_cache`
