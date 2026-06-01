---
title: "Attention_prefill_decode_ratio128"
date: 2026-05-28
tags:
  - #LLM
  - #flash_attention
  - #deepseek
  - #from_me
  - #待整理
status: 待整理
---

# Attention 中 `sparse_attn` 的 `prefill` / `decode` 计算过程（`compress_ratio=128`）

本文只聚焦 `compress_ratio=128` 时，`Attention.forward` 里 `sparse_attn` 的执行过程。

相关代码位置：

- `Attention.forward`：[inference/model.py](./inference/model.py#L489)
- `get_compress_topk_idxs`：[inference/model.py](./inference/model.py#L269)
- `Compressor`：[inference/model.py](./inference/model.py#L279)
- `sparse_attn_kernel`：[inference/kernel.py](./inference/kernel.py#L277)
- `sparse_attn` 包装函数：[inference/kernel.py](./inference/kernel.py#L355)

## 1. 前置背景

当 `compress_ratio=128` 时，`Attention` 的路径比 `ratio=4` 简单很多，关键区别只有一条：

- `ratio=128` 没有 `Indexer`
- 压缩后的历史 KV 不做 learned top-k 二次筛选
- 压缩块基本全量进入稀疏候选

所以最终候选只分成两类：

1. 局部窗口候选
   - 来自最近 `window_size=128` 个 token 的原始 `local kv`
2. 压缩历史候选
   - 每 `128` 个 token 压成 `1` 个 `compressed kv`
   - 这些压缩块直接通过 `get_compress_topk_idxs(...)` 生成候选索引

最终：

```text
topk_idxs = [window_topk_idxs | compress_topk_idxs]
```

然后把 `q`、`kv`、`topk_idxs` 送进 `sparse_attn`。

## 2. `ratio=128` 时 `Attention` 的关键分支

在 [inference/model.py](./inference/model.py#L508)：

```python
topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
if self.compress_ratio:
    offset = kv.size(1) if start_pos == 0 else win
    if self.indexer is not None:
        compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
    else:
        compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
    topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
```

当 `compress_ratio=128` 时：

- `self.indexer is None`
- 所以压缩候选索引来自 `get_compress_topk_idxs(...)`

它的含义很直接：

- `prefill` 时：把当前已经形成的压缩块都作为候选
- `decode` 时：把 cache 中当前已经形成的压缩块都作为候选

因此 `ratio=128` 的“稀疏”主要来自：

- 局部窗口只看最近 `128` 个 token
- 更远历史只看按 `128:1` 压缩后的块，而不是原始 token

## 3. `prefill` 的 `sparse_attn` 计算过程

`prefill` 对应 [inference/model.py](./inference/model.py#L528)：

```python
o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
```

这里的 `kv` 不是总缓存，而是当前轮临时拼出来的：

```text
kv = [local kv | kv_compress]
```

### 3.1 `prefill` 进入 `sparse_attn` 前的输入

为了让 shape 直观，取一个具体例子：

- `B=2`
- `S=512`
- `compress_ratio=128`

这时 Q 路径仍是：

- `q: [2, 512, 128, 512]`

当前 chunk 的原始 token 级 KV：

- `local kv: [2, 512, 512]`

主 `Compressor` 的行为：

1. `wkv(x)` 和 `wgate(x)` 都得到：
   - `[2, 512, 512]`
2. 以 `128` 个 token 为一组：
   - `[2, 512, 512] -> [2, 4, 128, 512]`
3. 在组内对 `128` 个 token 做 softmax gating 池化：
   - 输出 `kv_compress: [2, 4, 512]`

于是：

- `kv_compress: [2, 4, 512]`

拼接后：

- `kv: [2, 516, 512]`

### 3.2 `prefill` 下的候选索引

窗口部分：

- `window_topk_idxs: [2, 512, 128]`

压缩部分由 `get_compress_topk_idxs(128, bsz, seqlen, start_pos=0, offset=512)` 生成。

因为：

- 当前一共有 `512 / 128 = 4` 个压缩块

所以：

- `compress_topk_idxs: [2, 512, 4]`

拼接后：

- `topk_idxs: [2, 512, 132]`

这里 `kv` 的索引空间是：

- `0..511`：当前 chunk 的原始 `local kv`
- `512..515`：当前 chunk 压出来的 `compressed kv`

因此：

- `window_topk_idxs` 的值落在 `0..511`
- `compress_topk_idxs` 的值落在 `512..515`

### 3.3 `prefill` 中 `sparse_attn` 的外层逻辑

包装函数在 [inference/kernel.py](./inference/kernel.py#L355)：

```python
o = torch.empty_like(q)
kernel = sparse_attn_kernel(q.size(2), d, softmax_scale)
kernel(q, kv, o, attn_sink, topk_idxs)
```

输出 shape 与 `q` 相同：

- `o: [B, S, H, D]`

本例中：

- `o: [2, 512, 128, 512]`

### 3.4 kernel 的并行粒度

kernel 在 [inference/kernel.py](./inference/kernel.py#L294)：

```python
with T.Kernel(m, b, threads=threads) as (bx, by):
```

这里：

- `m = S = 512`
- `b = B = 2`

所以每个 kernel 实例处理一个固定的：

- `batch = by`
- `query_pos = bx`

即一次处理一个 query token，但同时处理该 token 的全部 attention heads。

### 3.5 第一步：读入当前 query token 的全部 head

```python
T.copy(q[by, bx, :, :], q_shared)
```

得到：

- `q_shared: [128, 512]`

### 3.6 第二步：按 `topk_idxs` 分块 gather 候选 KV

kernel 固定：

- `block = 64`

本例中：

- `topk = 132`

因此需要：

- `ceil(132 / 64) = 3` 个 block

每轮先取 64 个候选索引：

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

如果某个索引无效，逻辑与 `ratio=4` 时一样：

- `kv_shared` 填 0
- score 置成 `-inf`

### 3.7 第三步：计算 `q · k`

代码：

```python
T.gemm(q_shared, kv_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
```

shape 为：

- `q_shared: [128, 512]`
- `kv_shared: [64, 512]`
- 输出 `acc_s: [128, 64]`

即：

```text
acc_s[h, j] = dot(q_head_h, kv_candidate_j)
```

这里同样没有分开的 `K` / `V` 张量，`kv_shared` 先参与打分，后续又作为 value 做加权求和。

### 3.8 第四步：在线 softmax

先做缩放：

```python
acc_s[i, j] *= scale
```

随后进入在线 softmax 累积：

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

含义与 `ratio=4` 完全一致：

1. 当前 block 更新每个 head 的全局最大值
2. 旧结果按新的最大值重标定
3. 当前 block 计算 `exp(score - max)`
4. 更新总 softmax 分母

### 3.9 第五步：在线累计输出

代码：

```python
T.copy(acc_s, acc_s_cast)
for i, j in T.Parallel(h, d):
    acc_o[i, j] *= scores_scale[i]
T.gemm(acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
```

得到：

- `acc_o: [128, 512]`

含义是：

```text
acc_o[h] += sum_j softmax_numerator[h, j] * kv_shared[j]
```

### 3.10 第六步：加入 `attn_sink`

所有 block 结束后：

```python
sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
for i, j in T.Parallel(h, d):
    acc_o[i, j] /= sum_exp[i]
```

所以 `attn_sink` 仍然只是额外修正 softmax 分母，不对应真实 value 向量。

### 3.11 第七步：写回输出

最终：

- `o: [2, 512, 128, 512]`

## 4. `decode` 的 `sparse_attn` 计算过程

`decode` 对应 [inference/model.py](./inference/model.py#L533)：

```python
o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
```

这里与 `prefill` 的关键差别是：

- `prefill` 传入的是当前轮临时拼出来的 `kv`
- `decode` 传入的是累计好的总缓存 `self.kv_cache`

因此 `decode` 更准确地说是：

```text
先写 cache，
必要时生成一个新的 compressed kv，
再对总 cache 做 sparse attention
```

### 4.1 `decode` 进入 `sparse_attn` 前的输入

为了让 `ratio=128` 的增量路径清楚，取一个具体例子：

- `B=2`
- `S=1`
- `start_pos=511`
- `compress_ratio=128`

先得到当前 token 的 query 和 local KV：

- `q: [2, 1, 128, 512]`
- `kv_current: [2, 1, 512]`

然后当前 token 的 `local kv` 写入窗口缓存：

```python
self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
```

这里 `win=128`，所以窗口区逻辑长度始终是：

- `128`

接着主 `Compressor` 更新压缩状态。

由于：

```text
(511 + 1) % 128 == 0
```

所以这一时刻正好凑满一个新的压缩组，会生成一个新的 compressed KV，并写入压缩区。

这意味着到当前位置时：

- 压缩缓存逻辑长度是 `512 / 128 = 4`

### 4.2 `decode` 下的候选索引

窗口部分：

- `window_topk_idxs: [2, 1, 128]`

压缩部分由：

```python
get_compress_topk_idxs(128, bsz, seqlen=1, start_pos=511, offset=128)
```

生成。

因为当前已经有 `4` 个压缩块，所以：

- `compress_topk_idxs: [2, 1, 4]`

拼接后：

- `topk_idxs: [2, 1, 132]`

### 4.3 `decode` 时 `kv_cache` 的索引空间

`decode` 时传入 kernel 的 `kv` 是：

- `self.kv_cache[:bsz]`

它逻辑上分成两段：

1. `0..127`
   - 窗口区，存最近 `128` 个 token 的原始 `local kv`
2. `128..131`
   - 压缩区，存当前已有的 `4` 个 `compressed kv`

因此这一时刻 kernel 实际看到的输入可视为：

- `kv: [2, 132, 512]`

也正因为如此，`decode` 下压缩索引统一加的偏移量是：

- `offset = win = 128`

### 4.4 kernel 的执行过程

进入 kernel 后，计算流程与 `prefill` 没有数学差异，仍然是：

1. 读出当前 query token 的全部 heads：
   - `q_shared: [128, 512]`
2. 按 `topk_idxs` 从总 cache 中分块 gather 候选：
   - 每 block `64` 个位置
   - `kv_shared: [64, 512]`
3. 计算 `q · k`：
   - `acc_s: [128, 64]`
4. 用在线 softmax 更新：
   - `scores_max`
   - `scores_scale`
   - `sum_exp`
5. 用同一份 `kv_shared` 累计输出：
   - `acc_o: [128, 512]`
6. 加入 `attn_sink`
7. 写回当前 token 输出

最终：

- `o: [2, 1, 128, 512]`

### 4.5 一个细节：`decode` 下不是每步都会新增压缩块

`ratio=128` 时，主 `Compressor` 在 decode 阶段的关键判断是：

```python
should_compress = (start_pos + 1) % self.compress_ratio == 0
```

所以：

- 大多数 decode 步只会更新窗口 cache
- 只有每累计 `128` 个新 token，才会新增 `1` 个 compressed KV

这意味着：

- `window_topk_idxs` 几乎每步都在变化
- `compress_topk_idxs` 只会在跨过新的 `128` 边界时增加一个新压缩位置

## 5. `prefill` 与 `decode` 的核心区别

如果只看 `sparse_attn_kernel` 本身，`ratio=128` 下的 `prefill` 和 `decode` 几乎完全相同。真正不同的是 `kv` 的组织方式。

### 5.1 `prefill`

执行顺序更像：

```text
x
-> q
-> local kv
-> main compressor 生成 kv_compress
-> get_compress_topk_idxs
-> kv = [local kv | kv_compress]
-> sparse_attn(q, kv, topk_idxs)
```

特点：

- `kv` 是本轮现算现拼的临时张量
- 压缩块的索引偏移量是 `offset = seqlen`

### 5.2 `decode`

执行顺序更像：

```text
new token
-> q / local kv
-> 写入窗口 cache
-> 每逢 128 个 token 生成一个 compressed kv
-> get_compress_topk_idxs
-> sparse_attn(q, kv_cache, topk_idxs)
```

特点：

- `kv` 是累计好的总缓存
- 压缩块的索引偏移量固定是 `offset = window_size = 128`

## 6. 一句话总结

当 `compress_ratio=128` 时，`sparse_attn` 的计算内核和 `ratio=4` 没有本质不同，差别只在候选构造：

1. 近处仍然看窗口内的原始 token KV
2. 远处不再看原始 token，而是看每 `128:1` 压缩后的块
3. 这些压缩块不经过 `Indexer` 二次筛选，而是基本全量作为压缩候选
4. `prefill` 面向临时拼出的 `[local kv | compressed kv]`
5. `decode` 面向已经维护好的 `kv_cache`
