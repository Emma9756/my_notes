# DeepSeek-V4 逐步解析与 Shape 标注

ALL CORRECT (after reading 0513 17:30)

本文基于仓库中的实现文件 [inference/model.py](./inference/model.py) 和配置文件 [inference/config.json](./inference/config.json)，在默认 `world_size=1` 的前提下，按实际 `forward` 路径逐步拆解 `DeepSeek-V4`，并给出每一步的张量 shape。

如果你已经熟悉 `DeepSeek V3`，可以先记住这几个核心变化：

- 普通 residual 被 `Hyper-Connections (HC)` 替代
- 注意力从常规 MHA/MLA 演化为 `MLA + sliding window + compressed sparse attention`
- FFN 仍然是 `MoE`，但前 `3` 层支持 `hash routing`
- 主干外还多了 `1` 个 `MTP` block

## 1. 真实超参与记号

配置来自 [inference/config.json](./inference/config.json)：

- `vocab_size = 129280`
- `dim = D = 7168`
- `n_layers = L = 61`
- `n_heads = H = 128`
- `head_dim = 512`
- `rope_head_dim = 64`
- `q_lora_rank = 1536`
- `o_groups = 16`
- `o_lora_rank = 1024`
- `moe_inter_dim = 3072`
- `n_routed_experts = 384`
- `n_activated_experts = 6`
- `n_hash_layers = 3`
- `hc_mult = 4`
- `window_size = 128`
- `n_mtp_layers = 1`

下文统一使用这些记号：

- `B`: batch size
- `S`: 当前一次前向输入的序列长度
- `V = 129280`
- `D = 7168`
- `hc = 4`
- `H = 128`
- `d_head = 512`
- `d_rope = 64`
- `d_nope = 448`

## 2. 总体结构

`Transformer.forward` 在 [inference/model.py](./inference/model.py#L802)：

```python
h = self.embed(input_ids)
h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
for layer in self.layers:
    h = layer(h, start_pos, input_ids)
logits = self.head(h, ...)
```

整体数据流如下：

1. `input_ids`: `[B, S]`
2. `embed`: `[B, S, D] = [B, S, 7168]`
3. `HC` 扩成 4 路状态：`[B, S, 4, 7168]`
4. 经过 `61` 个 `Block`：shape 保持 `[B, S, 4, 7168]`
5. `head` 先把 `HC` 合并回单路，再做词表投影
6. 输出 `logits: [B, V] = [B, 129280]`

注意：这个实现里 `lm_head` 只取最后一个 token 做 logits，而不是输出 `[B, S, V]`。原因是：

```python
return F.linear(x[:, -1].float(), self.weight)
```

因此最终输出是 `[B, V]`。

## 3. 从输入到输出的主干 Shape

### Step 1: Token Embedding

位置：[inference/model.py](./inference/model.py#L83)

- 输入 `input_ids`: `[B, S]`
- `ParallelEmbedding`
- 输出 `h`: `[B, S, 7168]`

这一步和 `V3` 的 token embedding 本质一致。

### Step 2: 扩成 Hyper-Connections 多路状态

位置：[inference/model.py](./inference/model.py#L804)

```python
h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
```

- 输入：`[B, S, 7168]`
- `unsqueeze(2)`: `[B, S, 1, 7168]`
- `repeat(..., 4, ...)`: `[B, S, 4, 7168]`

这一步是 `V4` 和 `V3` 的第一处结构性差异。后续 block 不再维护单路 hidden state，而是维护 `4` 路并行的 HC 状态。

### Step 3: 经过 61 个 Block

位置：[inference/model.py](./inference/model.py#L786)

每个 block 的外部输入输出都是：

- 输入：`[B, S, 4, 7168]`
- 输出：`[B, S, 4, 7168]`

但 block 内部分两段：

1. `HC + Attention`
2. `HC + MoE`

下面详细拆。

## 4. 一个 Block 的完整 Shape

`Block.forward` 在 [inference/model.py](./inference/model.py#L689)。

### 4.1 Attention 子层前的 HC 压缩

输入：

- `x: [B, S, 4, 7168]`

调用 `hc_pre`：

```python
x = x.flatten(2).float()
mixes = F.linear(x, hc_fn) * rsqrt
pre, post, comb = hc_split_sinkhorn(...)
y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
```

逐步 shape：

1. 原始输入：`[B, S, 4, 7168]`
2. `flatten(2)`：`[B, S, 4*7168] = [B, S, 28672]`
3. `mix_hc = (2 + hc) * hc = (2 + 4) * 4 = 24`
4. `F.linear(..., hc_fn)`：`[B, S, 24]`
5. `hc_split_sinkhorn` 输出：
   - `pre: [B, S, 4]`
   - `post: [B, S, 4]`
   - `comb: [B, S, 4, 4]`
6. 按 `pre` 对 4 路状态加权求和：
   - 输出 `y: [B, S, 7168]`

所以 `hc_pre` 的作用可以理解成：

- 输入多路状态：`[B, S, 4, 7168]`
- 输出单路状态：`[B, S, 7168]`
- 同时产出后面恢复多路所需的 `post` 和 `comb`

### 4.2 Attention 子层

先过 `attn_norm`：

- 输入：`[B, S, 7168]`
- 输出：`[B, S, 7168]`

再进入 `Attention.forward`。

### 4.3 Attention 后的 HC 恢复

`Attention` 输出仍是：

- `[B, S, 7168]`

随后调用 `hc_post`：

```python
y = post.unsqueeze(-1) * x.unsqueeze(-2) + \
    torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
```

shape：

- `x`: `[B, S, 7168]`
- `residual`: `[B, S, 4, 7168]`
- `post`: `[B, S, 4]`
- `comb`: `[B, S, 4, 4]`
- 输出：`[B, S, 4, 7168]`

也就是把单路 attention 结果重新扩回 4 路 HC 状态。

### 4.4 FFN/MoE 子层

这部分与 attention 子层完全同构：

1. `hc_pre`: `[B, S, 4, 7168] -> [B, S, 7168]`
2. `ffn_norm`: `[B, S, 7168] -> [B, S, 7168]`
3. `MoE`: `[B, S, 7168] -> [B, S, 7168]`
4. `hc_post`: `[B, S, 7168] -> [B, S, 4, 7168]`

因此一个 block 的完整外部行为是：

```text
[B, S, 4, 7168]
  -> hc_pre
[B, S, 7168]
  -> Attention
[B, S, 7168]
  -> hc_post
[B, S, 4, 7168]
  -> hc_pre
[B, S, 7168]
  -> MoE
[B, S, 7168]
  -> hc_post
[B, S, 4, 7168]
```

## 5. Attention 逐步 Shape

`Attention` 在 [inference/model.py](./inference/model.py#L436)。

输入来自 `hc_pre`：

- `x: [B, S, 7168]`

### 5.1 Q 路径

```python
qr = q = self.q_norm(self.wq_a(x))
q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
apply_rotary_emb(q[..., -rd:], freqs_cis)
```

逐步 shape：

1. `wq_a`: `[B, S, 7168] -> [B, S, 1536]`
2. `q_norm`: `[B, S, 1536]`
3. `wq_b`: `[B, S, 1536] -> [B, S, 128*512] = [B, S, 65536]`
4. `unflatten`: `[B, S, 128, 512]`
5. 最后 `64` 维做 rotary：
   - non-rope 部分：`[B, S, 128, 448]`
   - rope 部分：`[B, S, 128, 64]`

最终 `q` 的 shape：

- `q: [B, S, 128, 512]`

### 5.2 KV 路径

```python
kv = self.wkv(x)
kv = self.kv_norm(kv)
apply_rotary_emb(kv[..., -rd:], freqs_cis)
```

逐步 shape：

1. `wkv`: `[B, S, 7168] -> [B, S, 512]`
2. `kv_norm`: `[B, S, 512]`
3. rotary 只作用在最后 `64` 维

最终 `kv`：

- `kv: [B, S, 512]`

这说明这里不是标准 MHA 那种每个 head 都单独存 `K/V`，而是典型 MLA 风格的“共享 latent KV 表示”。

### 5.3 Window 索引

```python
topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
```

输出是一个索引张量，表示当前每个 query 可以看到的局部窗口位置：

- `topk_idxs_window: [B, S, W]`

其中 `W` 不超过 `window_size = 128`。

### 5.4 压缩 KV 路径

若该层 `compress_ratio = r > 0`，则会额外构造压缩过的历史 KV。

#### 情况 A: `r = 128`

压缩后大致变成：

- `kv_compress: [B, floor(S/128), 512]`

对应的压缩索引大致是：

- `compress_topk_idxs: [B, S, floor(S/128)]`

#### 情况 B: `r = 4`

这时会额外启用 `Indexer`，不是把所有压缩块都拿来，而是先打分，再选 top-k。

`Indexer.forward` 内部关键 shape：

1. `qr`: `[B, S, 1536]`
2. `wq_b(qr)`: `[B, S, 64*128] = [B, S, 8192]`
3. reshape: `[B, S, 64, 128]`
4. 压缩缓存 `kv_cache`: `[B, T, 128]`
   - 这里 `T = end_pos // 4`
5. 打分：
   - `index_score = einsum("bshd,btd->bsht", ...)`
   - 输出 `[B, S, 64, T]`
6. head 加权求和后：
   - `[B, S, T]`
7. `topk` 后：
   - `[B, S, min(index_topk, T)]`

这里 `index_topk = 1024`。

### 5.5 稀疏注意力

prefill 阶段：

```python
o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
```

shape：

- `q: [B, S, 128, 512]`
- `kv`: `[B, S + S_compress, 512]` 或更小
- `topk_idxs: [B, S, K]`
- 输出 `o: [B, S, 128, 512]`

decode 阶段 `start_pos > 0` 时：

- 输入通常是单 token，即 `S = 1`
- 读取的是缓存 `self.kv_cache[:bsz]`
- 输出仍然是 `[B, 1, 128, 512]`

### 5.6 `sparse_attn_kernel` 详细计算过程

kernel 实现在 [inference/kernel.py](./inference/kernel.py#L277)。

这一节只分析 kernel 本体，不分析上层如何构造 `topk_idxs`。kernel 的职责非常明确：

- 对每个 `(batch, query_pos)`，按 `topk_idxs` 从 `kv` 中 gather 稀疏候选
- 计算所有 head 的 `q · k`
- 用在线 `softmax` 方式逐块归一化
- 得到每个 head 的加权输出

#### 5.6.1 输入与输出 shape

kernel 签名：

```python
q:         (b, m, h, d)
kv:        (b, n, d)
o:         (b, m, h, d)
attn_sink: (h,)
topk_idxs: (b, m, topk)
```

对应到当前模型：

- `b = B`
- `m = S`
- `h = 128`
- `d = 512`
- `n` 是可访问的 KV 总长度
  - prefill 时通常是 `S + S_compress`
  - decode 时通常是 `window + cache_compress`
- `topk` 是当前 query 真正要访问的稀疏候选数

单个 `(by, bx)` kernel 实例处理的是：

- 第 `by` 个 batch
- 第 `bx` 个 query 位置

也就是说，一次 kernel 实例的逻辑输入可以视为：

- `q_cur = q[by, bx]`: `[h, d]`
- `idx = topk_idxs[by, bx]`: `[topk]`

输出：

- `o[by, bx]`: `[h, d]`

#### 5.6.2 线程块与 tile

kernel 内部设定：

- `block = 64`
- `num_blocks = ceil(topk / 64)`

因此 `topk` 候选不会一次性全部处理，而是按每 `64` 个索引分块流式处理。

对应共享/寄存器缓存：

- `q_shared: [h, d]`
- `kv_shared: [64, d]`
- `acc_s: [h, 64]`
- `acc_o: [h, d]`
- `scores_max: [h]`
- `sum_exp: [h]`

可以把它理解成：

- `q_shared` 放当前 query 的全部 head
- `kv_shared` 放当前这一个 `64`-sized tile 的候选 KV
- `acc_s` 放这一 tile 内所有 `head x key` 的分数
- `acc_o` 放到当前为止累计的 attention 输出

#### 5.6.3 第一步：加载当前 query

```python
T.copy(q[by, bx, :, :], q_shared)
```

shape：

- 全局内存中的 `q[by, bx]`: `[h, d]`
- 搬到共享内存 `q_shared`: `[h, d]`

在当前模型里就是：

- `[128, 512]`

#### 5.6.4 第二步：按 64 个索引 gather 稀疏 KV

```python
idxs[i] = topk_idxs[by, bx, t * block + i] or -1
kv_shared[i, j] = kv[by, idxs[i], j] or 0
```

对第 `t` 个 tile：

1. 取出 `64` 个候选索引
   - `idxs: [64]`
2. 从 `kv[by]` 中 gather 对应向量
   - `kv_shared: [64, d]`

如果最后一个 tile 不满 `64` 个，越界位置会被填成：

- `idx = -1`
- 对应 `kv_shared` 行填 `0`
- 对应 score 初始化为 `-inf`

所以无效候选不会参与 softmax。

#### 5.6.5 第三步：计算这一 tile 的 attention score

```python
T.gemm(q_shared, kv_shared, acc_s, transpose_B=True, ...)
acc_s[i, j] *= scale
```

这里本质上是在做：

```text
acc_s = q_shared @ kv_shared^T
```

shape：

- `q_shared: [h, d]`
- `kv_shared^T: [d, 64]`
- `acc_s: [h, 64]`

代入本模型维度：

- `[128, 512] @ [512, 64] -> [128, 64]`

随后乘上 `scale = 1 / sqrt(d)`，也就是：

- `scale = 1 / sqrt(512)`

于是得到这一 tile 内每个 head 对 64 个候选 key 的 logits：

- `scores_tile: [128, 64]`

#### 5.6.6 第四步：在线 softmax 的 running max

代码：

```python
T.copy(scores_max, scores_max_prev)
T.reduce_max(acc_s, scores_max, dim=1, clear=False)
scores_scale[i] = exp(scores_max_prev[i] - scores_max[i])
```

这里的核心是 FlashAttention 风格的在线 softmax。

对每个 head 维护：

- `scores_max_prev[h]`: 处理前面 tile 后的历史最大值
- `scores_max[h]`: 加上当前 tile 后的新最大值

如果记第 `t` 个 tile 的分数为 `s_t`，那么对每个 head：

```text
new_max = max(old_max, max(s_t))
rescale = exp(old_max - new_max)
```

这么做的目的：

- 不需要把所有 `topk` 分数一次性存下来
- 每处理一块，都可以安全地更新 softmax 分母和分子
- 避免直接 `exp(大数)` 导致数值溢出

#### 5.6.7 第五步：在线 softmax 的分母累加

代码：

```python
acc_s[i, j] = exp(acc_s[i, j] - scores_max[i])
T.reduce_sum(acc_s, scores_sum, dim=1)
sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]
```

对当前 tile：

1. 先减去新的 `scores_max`
2. 再做 `exp`
3. 沿着 `64` 个 key 求和

shape：

- `acc_s`: `[h, 64]`
- `scores_sum`: `[h]`
- `sum_exp`: `[h]`

数学上，`sum_exp` 维护的是到当前为止的 softmax 分母：

```text
sum_exp_new
= sum_exp_old * exp(old_max - new_max)
 + sum_j exp(score_tile_j - new_max)
```

这正是分块 softmax 的标准递推写法。

#### 5.6.8 第六步：在线累加 softmax 分子

代码：

```python
for i, j in T.Parallel(h, d):
    acc_o[i, j] *= scores_scale[i]
T.gemm(acc_s_cast, kv_shared, acc_o, ...)
```

这里 `acc_o` 是 softmax 分子的累计值。

shape：

- `acc_o`: `[h, d]`
- `acc_s_cast`: `[h, 64]`
- `kv_shared`: `[64, d]`

其中：

```text
acc_s_cast = exp(scores_tile - new_max)
```

于是这一 tile 的增量是：

```text
acc_s_cast @ kv_shared
```

shape：

- `[h, 64] @ [64, d] -> [h, d]`

在本模型里就是：

- `[128, 64] @ [64, 512] -> [128, 512]`

完整递推公式：

```text
acc_o_new
= acc_o_old * exp(old_max - new_max)
 + Σ_j exp(score_tile_j - new_max) * kv_j
```

因此 `acc_o` 始终表示：

- 已处理候选上的“未除以 softmax 分母”的加权和

#### 5.6.9 第七步：`attn_sink` 如何参与

代码：

```python
sum_exp[i] += exp(attn_sink[i] - scores_max[i])
```

这是这个 kernel 最特别的一点。

`attn_sink` 的 shape 是：

- `[h]`

也就是每个 head 一个标量偏置。它只进入 softmax 分母，不进入 softmax 分子。

等价地说，对每个 head，最终归一化分母是：

```text
Z = Σ_j exp(score_j - max_score) + exp(attn_sink - max_score)
```

但分子仍然只有真实 KV 项：

```text
N = Σ_j exp(score_j - max_score) * kv_j
```

最终输出：

```text
o = N / Z
```

因此 `attn_sink` 的效果可以理解为：

- 给每个 head 添加一个“额外的虚拟注意力槽”
- 这个槽占据一部分 softmax 概率质量
- 但它不对应任何 value 向量，所以不会给分子贡献内容

从结果上看，它会把输出向量整体缩小一些，类似一种可学习的“泄漏概率”或“吸收槽”。

这里“虚拟槽”的说法是对代码行为的解释，不是源码里的命名。

#### 5.6.10 第八步：最终归一化

代码：

```python
acc_o[i, j] /= sum_exp[i]
```

shape：

- `acc_o`: `[h, d]`
- `sum_exp`: `[h]`
- 广播后输出仍然是 `[h, d]`

在本模型中：

- `[128, 512] / [128] -> [128, 512]`

随后写回：

```python
o[by, bx, :, :] = acc_o
```

所以单个 `(batch, query_pos)` 的最终输出是：

- `[128, 512]`

整体输出是：

- `[B, S, 128, 512]`

#### 5.6.11 用公式总结整个 kernel

对固定的 `batch=b`、`query=t`、`head=h0`，设稀疏候选集合为：

```text
I = topk_idxs[b, t]
```

则 kernel 计算的是：

```text
score_j = (q[b,t,h0] · kv[b, I_j]) / sqrt(d)
```

分子：

```text
N = Σ_j exp(score_j) * kv[b, I_j]
```

分母：

```text
Z = Σ_j exp(score_j) + exp(attn_sink[h0])
```

最终输出：

```text
o[b,t,h0] = N / Z
```

实际 kernel 没有直接按上面公式一次性算，而是：

- 把 `I` 分成多个 `64` 大小的 tile
- 用 running max / running sum 的在线 softmax 递推完成同样的结果

#### 5.6.12 和普通 dense attention 的差异

这个 kernel 和标准 dense attention 相比有 4 个关键差异：

1. `K/V` 不是全长扫描，而是先按 `topk_idxs` 稀疏 gather。
2. `K` 和 `V` 共用同一个 `kv` 张量，没有显式拆成 `k`、`v` 两份。
3. softmax 不是一次性在全部 key 上做，而是分 `64` 个候选一块在线累加。
4. 归一化分母里额外加入了 `attn_sink`，但分子没有对应项。

### 5.7 `Compressor + Indexer + sparse_attn` 的完整串联路径

这一节把上面的部件串起来，回答一个更关键的问题：

- 原始历史 token 是如何变成最终的稀疏候选集合的？
- `ratio=128` 和 `ratio=4` 的路径到底差在哪？

先给结论：

1. 最近的 token 永远走 `window_size=128` 的精细窗口路径。
2. 更远的历史 token 会先经过 `Compressor` 压成更短的 KV 序列。
3. 若 `compress_ratio=128`，压缩后的 KV 基本全量参与候选。
4. 若 `compress_ratio=4`，压缩后的 KV 还会经过 `Indexer` 二次打分，只保留 learned top-k 候选。
5. 最终 `window` 候选和 `compressed` 候选拼接成 `topk_idxs`，再送进 `sparse_attn_kernel`。

#### 5.7.1 Attention 层里的两类 KV

`Attention.forward` 里实际上维护了两类 KV：

1. 局部窗口 KV
   - 来自当前层的 `wkv(x)`
   - shape: `[B, S, 512]`
   - 存在 `self.kv_cache[:, :window]` 这一段里

2. 压缩历史 KV
   - 来自 `Compressor(x, start_pos)`
   - shape: `[B, S_compress, 512]` 或 decode 时 `[B, 1, 512]`
   - 存在 `self.kv_cache[:, window:]` 这一段里

所以 `self.kv_cache` 的逻辑布局是：

```text
self.kv_cache =
  [ recent sliding-window KV | compressed-history KV ]
```

shape：

```text
[B, window + max_seq_len / ratio, 512]
```

其中：

- 前半段长度固定为 `window = 128`
- 后半段长度取决于 `compress_ratio`

#### 5.7.2 Compressor 在做什么

`Compressor` 的输入是主干 hidden state：

- `x: [B, S, 7168]`

它先做两次线性变换：

```python
kv = self.wkv(x)
score = self.wgate(x)
```

shape：

- `kv: [B, S, coff * 512]`
- `score: [B, S, coff * 512]`

其中：

- 当 `ratio != 4` 时，`overlap=False`，`coff=1`
- 当 `ratio == 4` 时，`overlap=True`，`coff=2`

所以：

- `ratio=128` 时：
  - `kv: [B, S, 512]`
  - `score: [B, S, 512]`
- `ratio=4` 时：
  - `kv: [B, S, 1024]`
  - `score: [B, S, 1024]`

这里的 `score` 不是 attention score，而是压缩池化时的 gating score。

#### 5.7.3 `ratio=128` 时的压缩路径

prefill 时，若 `ratio=128`：

1. 去掉不能凑满一个 group 的尾巴
   - `cutoff = S - (S mod 128)`
2. reshape 成 group
   - `kv.unflatten(1, (-1, 128))`
   - `[B, cutoff, 512] -> [B, cutoff/128, 128, 512]`
3. `score` 做同样的 reshape
   - `[B, cutoff, 512] -> [B, cutoff/128, 128, 512]`
4. 在 group 内沿着 `128` 个 token 做 softmax 加权池化
   - `(kv * score.softmax(dim=2)).sum(dim=2)`
   - 输出 `[B, cutoff/128, 512]`

这个结果就是压缩历史 KV：

- `kv_compress: [B, floor(S/128), 512]`

随后：

1. `RMSNorm`
2. 对最后 `64` 维加 RoPE
3. 对前 `448` 维做量化模拟
4. 写入 `self.kv_cache[:, window:]`

所以 `ratio=128` 的本质是：

- 每 `128` 个历史 token 压成 `1` 个 `512` 维 KV 向量
- 不再额外 learned 筛选，后续直接作为稀疏候选来源之一

#### 5.7.4 `ratio=4` 时的压缩路径

`ratio=4` 比较特殊，因为这里开启了 `overlap=True`。

这时：

- `coff = 2`
- `kv` 和 `score` 都是 `[B, S, 1024]`
- 可以看成每个 token 同时产出两套 `512` 维特征

代码里的 `overlap_transform` 会把它改造成：

```text
[B, num_groups, 4, 1024]
-> [B, num_groups, 8, 512]
```

其中 `8 = 2 * ratio`，对应两种窗口：

1. 正常的当前 `4` token 窗口
2. 与前一组重叠的 `4` token 窗口

随后：

1. 对这 `8` 个候选位置做 softmax gating
2. 加权求和
3. 输出一个压缩后的 KV

最终仍然得到：

- `kv_compress: [B, floor(S/4), 512]`

但这个压缩结果比 `ratio=128` 更细，且带有重叠窗口信息，边界更平滑。

#### 5.7.5 Indexer 如何从压缩 KV 里再选 top-k

`Indexer` 只在 `compress_ratio=4` 时启用。

它的目标不是生成新的 KV，而是从已经压好的 `kv_cache` 中选出最值得看的压缩块。

输入：

- 主 attention 的低秩 query `qr: [B, S, 1536]`
- 原始 hidden state `x: [B, S, 7168]`
- 压缩缓存 `self.kv_cache: [B, T, 128]`
  - 这里 `T = end_pos // 4`

注意：`Indexer` 自己内部有一套小维度 compressor：

- 它不是用主 attention 的 `512` 维压缩 KV
- 而是构造 `128` 维的压缩 KV 用于检索评分

具体步骤：

1. `q = self.wq_b(qr)`
   - `[B, S, 1536] -> [B, S, 64*128] = [B, S, 8192]`
2. reshape
   - `[B, S, 64, 128]`
3. rotary on last `64` dims
4. Hadamard rotate + FP4 act quant
5. `self.compressor(x, start_pos)` 生成检索用压缩 KV
   - 写入 `Indexer.kv_cache`
   - 形状是 `[B, T, 128]`
6. `weights_proj(x)`
   - `[B, S, 7168] -> [B, S, 64]`
7. 检索打分：
   - `einsum("bshd,btd->bsht", q, kv_cache)`
   - `[B, S, 64, 128] x [B, T, 128] -> [B, S, 64, T]`
8. `relu` 后乘 head 权重，再对 head 求和
   - `[B, S, 64, T] -> [B, S, T]`
9. `topk`
   - `[B, S, T] -> [B, S, min(index_topk, T)]`

因此，`ratio=4` 时真正送给 `sparse_attn` 的压缩候选，并不是所有压缩块，而是：

- `Indexer` 从 `[B, T]` 个压缩位置里挑出的 learned top-k

#### 5.7.6 最终 `topk_idxs` 是怎么拼起来的

`Attention.forward` 里这部分逻辑最关键：

```python
topk_idxs = get_window_topk_idxs(...)
if self.compress_ratio:
    if self.indexer is not None:
        compress_topk_idxs = self.indexer(...)
    else:
        compress_topk_idxs = get_compress_topk_idxs(...)
    topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
```

所以最终候选集合分成两部分：

1. 局部窗口候选
   - `window_topk_idxs: [B, S, <=128]`
2. 压缩历史候选
   - `ratio=128` 时：
     - `compress_topk_idxs: [B, S, floor(S/128)]`
   - `ratio=4` 时：
     - `compress_topk_idxs: [B, S, min(1024, floor(S/4))]`

拼接后：

- `topk_idxs: [B, S, K_total]`

其中：

```text
K_total = K_window + K_compress
```

这个 `topk_idxs` 就是后面 `sparse_attn_kernel` 的直接输入。

#### 5.7.7 prefill 与 decode 的差别

prefill 时：

- `S` 可能很大
- 主 attention 的 `kv` 是当前整段序列的 `wkv(x)`
- `Compressor` 一次性把整段历史按组压缩
- `sparse_attn` 直接用当前构造出的 `kv` 和 `kv_compress`

所以 prefill 的逻辑更像：

```text
current chunk tokens
  -> local KV
  -> compressed KV
  -> build topk_idxs
  -> sparse_attn
```

decode 时：

- 通常 `S = 1`
- 当前 token 的局部 KV 写入滑动窗口缓存
- `Compressor` 只在凑满一个压缩组时，额外产出 `1` 个新的 compressed KV
- `sparse_attn` 直接读缓存 `self.kv_cache[:bsz]`

所以 decode 更像：

```text
new token
  -> append local window cache
  -> maybe append one compressed KV
  -> update topk_idxs
  -> sparse_attn over cache
```

#### 5.7.8 一张路径图总结

```text
x: [B, S, 7168]
  -> wkv
local kv: [B, S, 512]
  -> put into window cache

x: [B, S, 7168]
  -> Compressor
compressed kv: [B, S_compress, 512]
  -> put into compressed cache

if ratio == 4:
  x, qr
    -> Indexer
  compressed top-k idxs: [B, S, K_compress]
else:
  use all valid compressed positions

window idxs: [B, S, K_window]
compressed idxs: [B, S, K_compress]
  -> concat
topk_idxs: [B, S, K_total]

q: [B, S, 128, 512]
kv cache / current kv: [B, N, 512]
topk_idxs: [B, S, K_total]
  -> sparse_attn_kernel
o: [B, S, 128, 512]
```

这就是 `V4` 中“长历史先压缩，再选择，再稀疏注意力”的完整链路。

### 5.8 具体例子：`prefill` 与 `decode` 的实际 shape

这一节给两个具体例子，把前面的符号全部代成实际数字。

#### 5.8.1 例子 A：`prefill`，设 `B=2, S=256, ratio=4`

假设当前层：

- `batch size B = 2`
- 一次性输入长度 `S = 256`
- `compress_ratio = 4`
- `window_size = 128`
- 当前是 `start_pos = 0`

##### 第一步：主 attention 的输入

- `x: [2, 256, 7168]`

Q 路径：

1. `wq_a`
   - `[2, 256, 7168] -> [2, 256, 1536]`
2. `wq_b`
   - `[2, 256, 1536] -> [2, 256, 65536]`
3. reshape
   - `[2, 256, 128, 512]`

所以：

- `q: [2, 256, 128, 512]`
- `qr: [2, 256, 1536]`

KV 路径：

1. `wkv`
   - `[2, 256, 7168] -> [2, 256, 512]`
2. `kv_norm`
   - `[2, 256, 512]`

所以：

- `local kv: [2, 256, 512]`

##### 第二步：窗口候选索引

`window_size = 128`，因此：

- `window_topk_idxs: [2, 256, 128]`

含义：

- 对每个 query token，都给出最多 `128` 个局部窗口候选位置

##### 第三步：主 Compressor 生成压缩 KV

因为 `ratio=4`，所以 `overlap=True`, `coff=2`。

1. `self.wkv(x)`
   - `[2, 256, 7168] -> [2, 256, 1024]`
2. `self.wgate(x)`
   - `[2, 256, 7168] -> [2, 256, 1024]`

reshape 成 4-token 组：

3. `kv.unflatten(1, (-1, 4))`
   - `[2, 256, 1024] -> [2, 64, 4, 1024]`
4. `score.unflatten(1, (-1, 4))`
   - `[2, 64, 4, 1024]`

overlap 变换后：

5. `overlap_transform(kv)`
   - `[2, 64, 4, 1024] -> [2, 64, 8, 512]`
6. `overlap_transform(score)`
   - `[2, 64, 8, 512]`

group 内 softmax 池化：

7. `(kv * score.softmax(dim=2)).sum(dim=2)`
   - `[2, 64, 8, 512] -> [2, 64, 512]`

所以主 attention 用的压缩 KV 是：

- `kv_compress: [2, 64, 512]`

这里的 `64 = 256 / 4`。

##### 第四步：Indexer 生成压缩候选索引

因为 `ratio=4`，这一层会启用 `Indexer`。

Indexer 的 query 路径：

1. `Indexer.wq_b(qr)`
   - `[2, 256, 1536] -> [2, 256, 8192]`
2. reshape
   - `[2, 256, 64, 128]`

所以：

- `q_index: [2, 256, 64, 128]`

Indexer 的小型 Compressor 会生成检索用压缩 KV：

- `indexer.kv_cache: [2, 64, 128]`

对应打分：

3. `einsum("bshd,btd->bsht", q, kv_cache)`
   - `[2, 256, 64, 128] x [2, 64, 128]`
   - 输出 `[2, 256, 64, 64]`
4. 对 `64` 个 index-head 加权求和
   - `[2, 256, 64, 64] -> [2, 256, 64]`
5. `topk(min(index_topk, T))`
   - 这里 `T = 64`
   - `index_topk = 1024`
   - 所以取 `topk(64)`

得到：

- `compress_topk_idxs: [2, 256, 64]`

因为当前只有 `64` 个压缩块，所以这里实际上是“全取 64 个”，但顺序是 learned 排序后的。

##### 第五步：拼接最终稀疏候选

1. `window_topk_idxs: [2, 256, 128]`
2. `compress_topk_idxs: [2, 256, 64]`
3. 拼接：
   - `topk_idxs: [2, 256, 192]`

于是每个 query 最终访问：

- `128` 个局部窗口位置
- `64` 个压缩历史位置

##### 第六步：送入 `sparse_attn_kernel`

prefill 时，主代码会做：

```python
kv = torch.cat([kv, kv_compress], dim=1)
o = sparse_attn(q, kv, attn_sink, topk_idxs, scale)
```

所以 kernel 输入变成：

1. `q: [2, 256, 128, 512]`
2. `kv` 原本是当前 chunk 的局部 KV：
   - `[2, 256, 512]`
3. 拼接压缩 KV：
   - `[2, 256, 512] + [2, 64, 512] -> [2, 320, 512]`
4. `topk_idxs: [2, 256, 192]`

最终：

- `o: [2, 256, 128, 512]`

再经过输出投影：

1. `[2, 256, 128, 512]`
2. `view -> [2, 256, 16, 4096]`
3. `wo_a -> [2, 256, 16, 1024]`
4. flatten -> `[2, 256, 16384]`
5. `wo_b -> [2, 256, 7168]`

##### 第七步：这个例子里的直观理解

当 `B=2, S=256, ratio=4` 时，prefill 的 attention 可以理解成：

- 当前 `256` 个 token 先生成 `256` 个精细 `512` 维局部 KV
- 每 `4` 个 token 再压成 `1` 个 compressed KV，共 `64` 个
- 每个 query 最终看 `192` 个候选：
  - 最近窗口 `128`
  - 压缩历史 `64`

#### 5.8.2 例子 B：`decode`，设 `B=2, S=1, start_pos=255, ratio=4`

现在看单步解码。

假设：

- `B = 2`
- 当前只输入一个新 token，所以 `S = 1`
- `start_pos = 255`
- `compress_ratio = 4`

这意味着：

- 当前正在生成第 `256` 个位置的 token
- 之前已经积累了 `255` 个历史位置

##### 第一步：Q 和当前 token 的局部 KV

输入：

- `x: [2, 1, 7168]`

Q 路径：

1. `wq_a`
   - `[2, 1, 7168] -> [2, 1, 1536]`
2. `wq_b`
   - `[2, 1, 1536] -> [2, 1, 65536]`
3. reshape
   - `[2, 1, 128, 512]`

所以：

- `q: [2, 1, 128, 512]`

KV 路径：

1. `wkv`
   - `[2, 1, 7168] -> [2, 1, 512]`
2. `kv_norm`
   - `[2, 1, 512]`

所以：

- `kv_cur: [2, 1, 512]`

##### 第二步：更新滑动窗口缓存

代码：

```python
self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
```

因为：

- `start_pos = 255`
- `win = 128`
- `255 % 128 = 127`

所以当前 token 会写到：

- `self.kv_cache[:, 127]`

窗口缓存逻辑上仍是：

- `window cache: [2, 128, 512]`

##### 第三步：当前步是否生成新的 compressed KV

decode 时 Compressor 的关键判断是：

```python
should_compress = (start_pos + 1) % ratio == 0
```

代入：

- `(255 + 1) % 4 = 0`

所以这一步：

- `should_compress = True`

意味着第 `252,253,254,255` 这 4 个位置，正好凑成一个新的压缩块。

于是主 Compressor 会输出：

- `kv_compress_new: [2, 1, 512]`

并写入：

- `self.kv_cache[:, start_pos // ratio]`
- `255 // 4 = 63`

也就是 compressed cache 的第 `63` 个位置。

因此 decode 到这个时刻时：

- 压缩缓存逻辑长度是 `64`
- shape 可视为 `[2, 64, 512]`

##### 第四步：Indexer 在 decode 时的压缩候选

Indexer 也会在这一时刻更新自己的压缩缓存。

1. index query
   - `[2, 1, 1536] -> [2, 1, 64, 128]`
2. index compressed cache
   - `[2, 64, 128]`
3. 打分
   - `[2, 1, 64, 128] x [2, 64, 128] -> [2, 1, 64, 64]`
4. 对 `64` 个 index-head 求和
   - `[2, 1, 64]`
5. `topk(min(1024,64))`
   - `[2, 1, 64]`

所以：

- `compress_topk_idxs: [2, 1, 64]`

##### 第五步：decode 时最终送给 sparse attention 的索引

窗口部分：

- `window_topk_idxs: [2, 1, 128]`

压缩部分：

- `compress_topk_idxs: [2, 1, 64]`

拼接：

- `topk_idxs: [2, 1, 192]`

##### 第六步：decode 时 `sparse_attn` 直接读取缓存

decode 时代码不是拼当前 `kv + kv_compress`，而是直接读总缓存：

```python
o = sparse_attn(q, self.kv_cache[:bsz], attn_sink, topk_idxs, scale)
```

这里逻辑上的缓存可以视为：

1. 窗口部分：
   - `[2, 128, 512]`
2. 压缩部分：
   - `[2, 64, 512]`
3. 总缓存：
   - `[2, 192, 512]`

所以 kernel 输入是：

- `q: [2, 1, 128, 512]`
- `kv: [2, 192, 512]`
- `topk_idxs: [2, 1, 192]`

输出：

- `o: [2, 1, 128, 512]`

再走输出投影：

1. `[2, 1, 128, 512]`
2. `view -> [2, 1, 16, 4096]`
3. `wo_a -> [2, 1, 16, 1024]`
4. flatten -> `[2, 1, 16384]`
5. `wo_b -> [2, 1, 7168]`

##### 第七步：decode 例子的直观理解

当 `start_pos=255` 时：

- 当前新 token 先写入滑动窗口缓存
- 因为正好凑满 `4` 个 token，所以再生成 `1` 个新的 compressed KV
- query 最终从总缓存里稀疏读取 `192` 个位置
- 输出仍然是单 token 的：
  - `[2, 1, 128, 512] -> [2, 1, 7168]`

#### 5.8.3 两个例子的核心区别

`prefill` 和 `decode` 的数学形式是一致的，但工程路径不同：

1. `prefill`
   - 当前整段 `x` 一次性生成整段 `local kv`
   - 当前整段 `x` 一次性生成整段 `compressed kv`
   - `sparse_attn` 直接用本次构造出的 `kv`

2. `decode`
   - 每次只输入一个 token
   - 局部窗口缓存逐 token 更新
   - 压缩缓存只有在凑满一个 group 时才新增 `1` 个条目
   - `sparse_attn` 直接读累计好的 cache

如果只记一句话，可以记成：

- `prefill` 是“现算现拼后注意力”
- `decode` 是“写 cache、必要时补压缩、再对 cache 做稀疏注意力”

### 5.9 输出投影

```python
o = o.view(bsz, seqlen, self.n_local_groups, -1)
wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
x = self.wo_b(o.flatten(2))
```

逐步 shape：

1. `o`: `[B, S, 128, 512]`
2. 共有 `128*512 = 65536` 个通道
3. 分成 `16` 组：
   - `[B, S, 16, 4096]`
4. `wo_a` 做组内低秩投影：
   - `[B, S, 16, 4096] -> [B, S, 16, 1024]`
5. flatten：
   - `[B, S, 16384]`
6. `wo_b`：
   - `[B, S, 16384] -> [B, S, 7168]`

所以整个 attention 可以压缩成：

```text
[B, S, 7168]
  -> Q: [B, S, 128, 512]
  -> KV: [B, S, 512]
  -> sparse gather by topk_idxs
  -> sparse_attn
[B, S, 128, 512]
  -> grouped low-rank O projection
[B, S, 7168]
```

## 6. MoE 逐步 Shape

`MoE` 在 [inference/model.py](./inference/model.py#L609)。

输入：

- `x: [B, S, 7168]`

### 6.1 展平 token

```python
x = x.view(-1, self.dim)
```

- `[B, S, 7168] -> [B*S, 7168]`

### 6.2 Gate 打分

```python
scores = linear(x.float(), self.weight.float())
```

- `weight: [384, 7168]`
- `scores: [B*S, 384]`

然后根据 `score_func = sqrtsoftplus`：

- 仍然保持 `[B*S, 384]`

### 6.3 路由选择

前 `3` 层：

- `hash routing`
- `indices = tid2eid[input_ids]`
- `indices: [B*S, 6]`

后续层：

- `topk(scores, 6)`
- `indices: [B*S, 6]`

路由权重：

- `weights = original_scores.gather(1, indices)`
- `weights: [B*S, 6]`

归一化后仍是：

- `weights: [B*S, 6]`

### 6.4 单个 Expert 的内部 Shape

`Expert` 在 [inference/model.py](./inference/model.py#L587)。

对某个 expert 来说，假设它拿到了 `N_i` 个 token：

1. 输入：`[N_i, 7168]`
2. `w1`: `[N_i, 3072]`
3. `w3`: `[N_i, 3072]`
4. `SiLU(w1) * w3`: `[N_i, 3072]`
5. 若带路由权重：
   - `weights[idx, top, None]` 是 `[N_i, 1]`
   - 广播后结果仍是 `[N_i, 3072]`
6. `w2`: `[N_i, 3072] -> [N_i, 7168]`

### 6.5 汇总所有 routed experts + shared expert

路由专家累计后：

- `y: [B*S, 7168]`

再加上 shared expert：

```python
y += self.shared_experts(x)
```

其中：

- `shared_experts(x): [B*S, 7168]`

最后 reshape 回去：

- `[B*S, 7168] -> [B, S, 7168]`

因此 MoE 总结为：

```text
[B, S, 7168]
  -> flatten
[B*S, 7168]
  -> gate scores
[B*S, 384]
  -> topk/hash routing
indices: [B*S, 6]
weights: [B*S, 6]
  -> selected experts
[B*S, 7168]
  -> reshape
[B, S, 7168]
```

## 7. 输出 Head 的 Shape

`ParallelHead` 在 [inference/model.py](./inference/model.py#L704)。

输入：

- `x: [B, S, 4, 7168]`

### 7.1 HC Head 聚合

```python
x = x.flatten(2).float()
mixes = F.linear(x, hc_fn) * rsqrt
pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
```

shape：

1. `[B, S, 4, 7168]`
2. flatten 后：`[B, S, 28672]`
3. `mixes: [B, S, 4]`
4. `pre: [B, S, 4]`
5. 加权求和后：
   - `[B, S, 7168]`

### 7.2 归一化与词表投影

1. `norm`: `[B, S, 7168]`
2. 只取最后一个 token：
   - `x[:, -1]: [B, 7168]`
3. 词表投影：
   - `[B, 7168] -> [B, 129280]`

最终：

- `logits: [B, 129280]`

## 8. MTP Block 的 Shape

`MTPBlock` 在 [inference/model.py](./inference/model.py#L739)。

输入：

- `x: [B, S, 4, 7168]`
- `input_ids: [B, S]`

步骤：

1. `embed(input_ids)`：
   - `e: [B, S, 7168]`
2. `enorm(e)`：
   - `[B, S, 7168]`
3. `hnorm(x)`：
   - `[B, S, 4, 7168]`
4. `e_proj(e)`：
   - `[B, S, 7168]`
5. `e_proj(e).unsqueeze(2)`：
   - `[B, S, 1, 7168]`
6. `h_proj(x)`：
   - `[B, S, 4, 7168]`
7. 广播相加：
   - `[B, S, 1, 7168] + [B, S, 4, 7168]`
   - 输出 `[B, S, 4, 7168]`
8. 再过一个完整 `Block`
9. `head` 输出：
   - `[B, 129280]`

这个结构可以理解成：主干 hidden state 和当前 token embedding 再融合一次，用于 MTP 预测。

## 9. 压缩层分布

`compress_ratios` 来自配置：

```json
[128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0]
```

主干 `61` 层对应前 `61` 个值，因此可以粗略理解为：

- 第 `0, 1` 层：压缩比 `128`
- 从第 `2` 层开始，`4` 和 `128` 交替出现
- 最后一层主干的压缩比也是 `4`
- 第 `62` 个值 `0` 对应额外 `MTP` 层

也就是说，`V4` 主干大多数层都不是纯局部 window attention，而是：

- 最近 `128` token 走精细窗口
- 更久远历史走压缩 KV
- 某些层压缩比较激进 (`128`)
- 某些层压缩比较细 (`4`)

## 10. 和 V3 对照时最值得记的点

如果你已经有 `DeepSeek V3` 基础，建议把 `V4` 记成下面这个映射：

- `V3 residual` -> `V4 Hyper-Connections (4路状态)`
- `V3 attention` -> `V4 MLA + sliding window + compressed sparse attention`
- `V3 MoE` -> `V4 MoE + 前3层 hash routing + shared expert`
- `V3 head` -> `V4 HC聚合后再出 logits`
- `V3 主干输出` 常常可理解成单路 hidden state
- `V4 主干输出` 始终是 `[B, S, 4, 7168]`，直到 head 才压回单路

## 11. 一页版总览

```text
input_ids
  [B, S]

embed
  -> [B, S, 7168]

HC expand
  -> [B, S, 4, 7168]

repeat 61 blocks:
  hc_pre
    [B, S, 4, 7168] -> [B, S, 7168]
  attention
    Q:  [B, S, 128, 512]
    KV: [B, S, 512]
    O:  [B, S, 7168]
  hc_post
    [B, S, 7168] -> [B, S, 4, 7168]
  hc_pre
    [B, S, 4, 7168] -> [B, S, 7168]
  MoE
    [B, S, 7168]
      -> [B*S, 7168]
      -> gate [B*S, 384]
      -> topk/hash [B*S, 6]
      -> experts [B*S, 7168]
      -> [B, S, 7168]
  hc_post
    [B, S, 7168] -> [B, S, 4, 7168]

HC head merge
  [B, S, 4, 7168] -> [B, S, 7168]

final lm_head on last token
  [B, 7168] -> [B, 129280]
```

## 12. 代码定位

- 主入口：[inference/model.py](./inference/model.py#L770)
- Block：[inference/model.py](./inference/model.py#L648)
- Attention：[inference/model.py](./inference/model.py#L436)
- MoE：[inference/model.py](./inference/model.py#L609)
- MTPBlock：[inference/model.py](./inference/model.py#L739)
- 配置：[inference/config.json](./inference/config.json)
