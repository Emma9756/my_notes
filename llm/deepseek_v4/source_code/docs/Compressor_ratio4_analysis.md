# `compress_ratio=4` 下 `Compressor / compress_kv` 详细解析

本文只聚焦 `compress_ratio=4` 时主 `Compressor` 的执行过程，也就是 `compressed kv` 是如何生成的。

相关代码位置：

- `Compressor`：[inference/model.py](./inference/model.py#L279)
- `Attention.forward` 中调用 `Compressor`：[inference/model.py](./inference/model.py#L523)

## 1. 一句话概括

当 `compress_ratio=4` 时，`Compressor` 不是简单把 4 个 token 压成 1 个向量，而是：

```text
每个 token 先产出两套 512 维特征
-> 每 4 个 token 分组
-> 通过 overlap_transform 构造 8 个候选位置
-> 对这 8 个位置做逐通道 softmax gating 池化
-> 输出 1 个 512 维 compressed kv
```

这里的关键在于：

- `ratio=4` 开启了 `overlap=True`
- 会显式引入与前一组重叠的压缩窗口
- 压缩边界比 `ratio=128` 更平滑

## 2. 为什么 `ratio=4` 特别

在 `__init__` 里：

```python
self.overlap = compress_ratio == 4
coff = 1 + self.overlap
```

因此当 `compress_ratio=4` 时：

- `overlap = True`
- `coff = 2`

这会直接改变压缩器内部特征维度：

- `self.wkv: 7168 -> 2 * 512 = 1024`
- `self.wgate: 7168 -> 2 * 512 = 1024`

所以 `ratio=4` 时，每个 token 不是只生成 1 套 `512` 维特征，而是同时生成 2 套：

- 前 512 维：给 overlap 路径用
- 后 512 维：给正常当前组路径用

这是后面 `overlap_transform` 的基础。

## 3. 输入与基础投影

`Compressor.forward(x, start_pos)` 的输入仍然是主干 hidden state：

- `x: [B, S, 7168]`

先转成 fp32，再做两次线性：

```python
x = x.float()
kv = self.wkv(x)
score = self.wgate(x)
```

因此当 `ratio=4` 时：

- `kv: [B, S, 1024]`
- `score: [B, S, 1024]`

这里：

- `kv` 是压缩候选内容
- `score` 是压缩池化的 gating score

和 `ratio=128` 一样，`score` 不是 attention score，而是逐通道池化权重。

## 4. `overlap_transform` 到底在做什么

这是 `ratio=4` 的核心。

函数定义：

```python
def overlap_transform(self, tensor: torch.Tensor, value=0):
    # tensor: [b,s,r,2d]
    b, s, _, _ = tensor.size()
    ratio, d = self.compress_ratio, self.head_dim
    new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
    new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
    new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
    return new_tensor
```

输入 shape：

- `[B, num_groups, 4, 1024]`

输出 shape：

- `[B, num_groups, 8, 512]`

可以把 `1024` 看成两半：

- 前半 `:512`
- 后半 `512:`

变换后的 8 个位置分成两段：

1. `new_tensor[:, :, 4:]`
   - 来自当前 group 的后半 `512` 维
   - 对应“正常当前窗口”
2. `new_tensor[:, 1:, :4]`
   - 来自前一个 group 的前半 `512` 维
   - 对应“与前一组重叠的窗口”

所以对第 `g` 个 group 来说，真正参与池化的 8 个候选位置可以理解成：

- 前 4 个：上一组留下来的 overlap 候选
- 后 4 个：当前组自己的 normal 候选

第 0 组没有上一组，因此它的前 4 个位置会被填成默认值：

- `kv` 路径填 `0`
- `score` 路径填 `-inf`

这样第 0 组实际上只会用当前组的 4 个 normal 候选。

## 5. prefill 时的压缩过程

当 `start_pos == 0` 时，走 prefill 分支。

### 5.1 第一步：判断能不能压

```python
should_compress = seqlen >= ratio
remainder = seqlen % ratio
cutoff = seqlen - remainder
offset = ratio if overlap else 0
```

当 `ratio=4` 时：

- `should_compress = (S >= 4)`
- `cutoff = S - (S mod 4)`
- `offset = 4`

例如：

- `S=256` -> `cutoff=256, remainder=0`
- `S=258` -> `cutoff=256, remainder=2`
- `S=3` -> `cutoff=0, remainder=3`

### 5.2 第二步：保存最后一个完整 group 作为 overlap 起点

这段代码只在 overlap 分支触发：

```python
if overlap and cutoff >= ratio:
    self.kv_state[:bsz, :ratio] = kv[:, cutoff-ratio : cutoff]
    self.score_state[:bsz, :ratio] = score[:, cutoff-ratio : cutoff] + self.ape
```

含义是：

- 把当前 prefill 里最后一个完整的 4-token group 存到状态里
- 后续 decode 时，这一组会作为“上一组 overlap 候选”的起点

如果 `B=2, S=256`：

- `kv[:, 252:256]: [2, 4, 1024]`

会被存到：

- `kv_state[:2, :4]: [2, 4, 1024]`

### 5.3 第三步：保存尾部 remainder

当 `remainder > 0` 时：

```python
kv, self.kv_state[:bsz, offset : offset+remainder] = kv.split([cutoff, remainder], dim=1)
self.score_state[:bsz, offset : offset+remainder] = score[:, cutoff:] + self.ape[:remainder]
score = score[:, :cutoff]
```

因为这里 `offset=4`，所以：

- `kv_state[:bsz, :4]` 留给上一完整组
- `kv_state[:bsz, 4:4+remainder]` 留给当前未凑满的尾巴

例如 `B=2, S=258`：

- `cutoff=256, remainder=2`
- `kv_state[:2, :4]` 存最后一个完整组 `252..255`
- `kv_state[:2, 4:6]` 存尾巴 `256..257`

这说明 `ratio=4` 的 decode 状态并不只存“当前未满 4 个 token”，还要额外保留上一完整组，供 overlap 使用。

### 5.4 第四步：按 4 个 token 分组

```python
kv = kv.unflatten(1, (-1, ratio))
score = score.unflatten(1, (-1, ratio)) + self.ape
```

shape 变化：

- `kv: [B, cutoff, 1024] -> [B, cutoff/4, 4, 1024]`
- `score: [B, cutoff, 1024] -> [B, cutoff/4, 4, 1024]`

例如 `B=2, cutoff=256`：

- `kv: [2, 64, 4, 1024]`
- `score: [2, 64, 4, 1024]`

这里每个 group 对应连续 4 个 token。

### 5.5 第五步：做 overlap 变换

对于 `kv`：

```python
kv = self.overlap_transform(kv, 0)
```

得到：

- `[B, G, 4, 1024] -> [B, G, 8, 512]`

对于 `score`：

```python
score = self.overlap_transform(score, float("-inf"))
```

也得到：

- `[B, G, 8, 512]`

只是前面无效的 overlap 位置在 `score` 中用 `-inf` 填充，这样 softmax 后不会贡献权重。

### 5.6 第六步：对 8 个候选位置做逐通道 softmax 池化

真正的压缩发生在：

```python
kv = (kv * score.softmax(dim=2)).sum(dim=2)
```

shape 变化：

- `kv: [B, G, 8, 512]`
- `score.softmax(dim=2): [B, G, 8, 512]`
- 输出 `kv_compress: [B, G, 512]`

其中：

- `G = cutoff / 4`

更细地说，对某个 batch、某个 group、某个通道 `d`：

```text
compressed_kv[d] =
sum_{i=0..7} softmax(score[i, d]) * kv[i, d]
```

所以 `ratio=4` 的压缩不是“4 选 1”，而是“最多 8 个候选位置做逐通道 gated pooling”。

### 5.7 一个完整的 prefill 例子

设：

- `B=2`
- `S=256`
- `ratio=4`

则：

1. 输入：
   - `x: [2, 256, 7168]`
2. 投影：
   - `kv = wkv(x): [2, 256, 1024]`
   - `score = wgate(x): [2, 256, 1024]`
3. 因为 `256 % 4 = 0`
   - `cutoff=256`
   - `remainder=0`
4. 分组：
   - `kv -> [2, 64, 4, 1024]`
   - `score -> [2, 64, 4, 1024]`
5. overlap 变换：
   - `kv -> [2, 64, 8, 512]`
   - `score -> [2, 64, 8, 512]`
6. 池化：
   - `kv_compress: [2, 64, 512]`

含义：

- 每 4 个 token 会压出 1 个 compressed KV
- 但这个压缩结果融合了当前组和前一组的 overlap 信息

## 6. decode 时的压缩过程

当 `start_pos > 0` 时，走 decode 分支。通常：

- `x: [B, 1, 7168]`
- `kv: [B, 1, 1024]`
- `score: [B, 1, 1024]`

### 6.1 第一步：判断当前步是否凑满一个 4-token group

```python
should_compress = (start_pos + 1) % self.compress_ratio == 0
```

当 `ratio=4` 时：

- 每累计 4 个 token，才会真正生成 1 个 compressed KV

例如：

- `start_pos=2` -> 不压
- `start_pos=3` -> 生成第 0 个 compressed KV
- `start_pos=7` -> 生成第 1 个 compressed KV
- `start_pos=255` -> 生成第 63 个 compressed KV

### 6.2 第二步：先给当前 token 加组内位置偏置

```python
score += self.ape[start_pos % ratio]
```

这意味着：

- 组内第 0 个 token 加 `ape[0]`
- 组内第 1 个 token 加 `ape[1]`
- 组内第 2 个 token 加 `ape[2]`
- 组内第 3 个 token 加 `ape[3]`

### 6.3 第三步：把当前 token 写入“当前组”缓存

```python
self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
```

因为 `ratio=4`，所以这里写入的是：

- `kv_state[:bsz, 4 + pos_in_group]`
- `score_state[:bsz, 4 + pos_in_group]`

也就是说：

- `kv_state[:bsz, :4]` 存“上一完整组”
- `kv_state[:bsz, 4:8]` 存“当前正在积累的组”

这是 overlap decode 路径最关键的状态布局。

### 6.4 第四步：凑满 4 个时，拼出 8 个候选位置

只有当 `should_compress=True` 时，才会真正压缩：

```python
kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1)
score_state = torch.cat([self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1)
```

注意这里不是直接拿完整的 `1024` 维状态做池化，而是手工拼两半：

1. `self.kv_state[:bsz, :4, :512]`
   - 上一完整组的前半 512 维
2. `self.kv_state[:bsz, 4:, 512:]`
   - 当前组的后半 512 维

拼完后：

- `kv_state: [B, 8, 512]`
- `score_state: [B, 8, 512]`

这就与 prefill 中 `overlap_transform` 产出的 8 个候选位置完全对应。

### 6.5 第五步：对 8 个候选位置做池化

```python
kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
```

shape：

- `kv_state: [B, 8, 512]`
- `score_state.softmax(dim=1): [B, 8, 512]`
- 输出 `kv_compress_new: [B, 1, 512]`

这说明 decode 和 prefill 的数学本质一致：

- 都是在 8 个 overlap 候选位置上做逐通道 softmax pooling

只是：

- prefill 一次压很多组
- decode 每次最多只压 1 组

### 6.6 第六步：更新“上一完整组”状态

压完后：

```python
self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
```

含义是：

- 当前组压缩完后，它就变成下一轮的“上一完整组”

因此下一次再凑满新的 4-token group 时，就能继续形成 overlap。

## 7. 压缩后的后处理

只要真的压出了 `kv_compress`，无论 prefill 还是 decode，后面都会做相同的后处理。

### 7.1 RMSNorm

```python
kv = self.norm(kv.to(dtype))
```

shape：

- prefill：`[B, G, 512]`
- decode：`[B, 1, 512]`

### 7.2 对最后 64 维加 RoPE

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

- 前 448 维是 non-rope
- 后 64 维是 rope

### 7.3 只量化 non-rope 的前 448 维

```python
act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
```

即：

- `kv[..., :448]` 做量化模拟
- `kv[..., 448:512]` 保留 bf16

### 7.4 写入 compressed cache

```python
if start_pos == 0:
    self.kv_cache[:bsz, :seqlen // ratio] = kv
else:
    self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
```

因此：

- prefill 时，一次写入当前 chunk 中所有 `[B, floor(S/4), 512]` 的 compressed KV
- decode 时，每 4 个 token 新增写入 1 个 `[B, 1, 512]`

## 8. 和 `ratio=128` 的本质区别

`ratio=128`：

- 每组只看当前这 128 个 token
- 池化候选数是 128
- 无 overlap

`ratio=4`：

- 每组不是只看当前 4 个 token
- 还会额外引入前一组的 overlap 候选
- 池化候选数是最多 8 个
- 边界更平滑，压缩粒度也更细

## 9. 本质总结

`compress_ratio=4` 时的 `compress_kv` 本质上是：

1. 从 hidden state `x` 生成 `1024` 维的压缩专用 `kv` 与 `score`
2. 把每个 token 拆成两套 `512` 维特征
3. 每 4 个 token 形成一组，并与前一组构成 overlap 候选
4. 在最多 8 个候选位置上做逐通道 softmax gating 池化
5. 输出 1 个 `512` 维 compressed KV
6. 再做 norm、RoPE、量化模拟并写入 compressed cache
