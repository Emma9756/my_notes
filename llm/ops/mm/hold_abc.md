我没有看到你提到的图示，但在 CUDA GEMM 里，“hold A / hold B / hold C”通常指 **在计算过程中优先让 A、B、C 中哪一个矩阵块长期驻留在片上资源中**，例如 shared memory、register 或 warp/thread tile 中。它们本质上都是在算：

$$C = A \times B + C$$

假设：

$$A \in \mathbb{R}^{M \times K},\quad
B \in \mathbb{R}^{K \times N},\quad
C \in \mathbb{R}^{M \times N}$$

GEMM 的核心循环是：

```cpp
for m in M:
  for n in N:
    for k in K:
      C[m][n] += A[m][k] * B[k][n]
```

三种算法的区别，就是在这个三重循环里，**谁被固定住，谁被不断流入流出**。

---

## 1. Hold C：输出驻留 / Output Stationary

这是 CUDA GEMM 中最常见、最重要的策略。

### 核心思想

每个线程或 warp 负责一个 C 的小 tile，把这个 C tile 一直放在寄存器里累加。

伪代码类似：

```cpp
C_reg = 0

for k_tile in K:
    load A_tile
    load B_tile
    C_reg += A_tile * B_tile

store C_reg to global memory
```

也就是说：

* A tile 一段一段加载；
* B tile 一段一段加载；
* C tile 一直留在寄存器中，直到整个 K 维累加结束；
* 最后只写回一次 C。

### 优点

**最大优点是 C 的读写次数最少。**

对于一个 C 元素：

```cpp
C[m][n] = sum_k A[m][k] * B[k][n]
```

如果不 hold C，每次 k 循环都要读写 C，会非常慢。

Hold C 的方式通常是：

```cpp
float acc = 0;
for k:
    acc += A[m][k] * B[k][n];
C[m][n] = acc;
```

这样 C 只在最后写一次。

### 缺点

需要大量寄存器保存 accumulator。

如果每个线程负责太多 C 元素，比如一个线程算 8×8 个输出，就需要 64 个 accumulator register。这样会：

* 增加寄存器压力；
* 降低 occupancy；
* 可能产生 register spill；
* 对 tile shape 设计要求高。

### 适合场景

这是现代高性能 CUDA GEMM 的主流策略，尤其适合：

* 标准 GEMM；
* Tensor Core GEMM；
* cuBLAS 风格的 block/warp/thread tiling；
* K 维较大，需要长时间累加的情况。

---

## 2. Hold A：A 驻留 / A Stationary

### 核心思想

把 A 的一个 tile 固定在片上，让它和多个 B tile 相乘，生成多个 C tile。

可以理解为：

```cpp
load A_tile once

for different B_tile:
    load B_tile
    update corresponding C_tile
```

图像上通常表现为：

```text
          B0   B1   B2   B3
        +----+----+----+----+
A_tile  | C0 | C1 | C2 | C3 |
        +----+----+----+----+
```

一个 A tile 被重复用于多个 N 方向上的 B tile。

### 优点

A 的复用率高。

如果一个 A tile 可以和多个 B tile 相乘，那么 A 从 global memory 加载一次后，可以服务多个输出 tile。

这对如下情况有利：

* M 方向较小；
* N 方向较大；
* 同一批 A 要和很多不同的 B 块相乘；
* A 的加载成本较高，或者 A 在 cache/shared memory 中复用价值大。

### 缺点

C 的管理更复杂。

因为一个 A tile 会更新多个 C tile，所以你可能需要：

* 同时维护多个 C tile accumulator；
* 或者多次读写 C；
* 或者跨 block 协调 C 的归约。

如果 C 不能很好地驻留在寄存器中，可能导致 C 的 global memory 访问增加。

### 适合场景

Hold A 更适合：

* A 被多个 B 复用；
* batch GEMM 中 A 固定、B 多变；
* 某些右乘场景，例如：

$$C_i = A \times B_i$$

其中 A 不变，B_i 很多。

---

## 3. Hold B：B 驻留 / B Stationary

### 核心思想

和 Hold A 对称。把 B 的一个 tile 固定在片上，让多个 A tile 来和它相乘，生成多个 C tile。

可以理解为：

```cpp
load B_tile once

for different A_tile:
    load A_tile
    update corresponding C_tile
```

图像上通常表现为：

```text
        B_tile
          |
+----+  +----+
| A0 |  | C0 |
+----+  +----+
| A1 |  | C1 |
+----+  +----+
| A2 |  | C2 |
+----+  +----+
| A3 |  | C3 |
+----+  +----+
```

一个 B tile 被重复用于多个 M 方向上的 A tile。

### 优点

B 的复用率高。

如果同一个 B tile 可以服务很多 A tile，那么 B 的 global memory 访问可以显著减少。

适合：

* M 方向较大；
* N 方向较小；
* 多个 A 共享同一个 B；
* 某些推理或 batch 场景中权重 B 固定。

例如神经网络里常见的：

$$C_i = A_i \times W$$

其中 W 相当于 B，多个输入 A_i 共用同一个权重矩阵 W。此时 hold B 很自然。

### 缺点

和 Hold A 类似，C 的分布与累加可能更复杂。

如果多个 A tile 使用同一个 B tile，输出 C 会分布在不同的 M 区域。需要合理安排 block mapping，否则可能造成：

* shared memory 使用不均；
* global memory 写回不连续；
* C accumulator 数量过多；
* occupancy 下降。

### 适合场景

Hold B 适合：

* B 是权重矩阵，被大量输入复用；
* 多 batch 推理；
* 矩阵右侧操作数复用明显；
* A 多变、B 固定的场景。

---

## 三者对比

| 策略     | 固定谁           | 流动谁 | 最大复用对象  | 主要优势                | 主要问题                     |
| ------ | ------------- | --- | ------- | ------------------- | ------------------------ |
| Hold C | C accumulator | A、B | C 的累加结果 | C 只写回一次，最适合高性能 GEMM | 寄存器压力大                   |
| Hold A | A tile        | B、C | A       | A 加载一次，多次使用         | C 管理复杂，可能增加 C 访存         |
| Hold B | B tile        | A、C | B       | B 加载一次，多次使用         | C 管理复杂，block mapping 更敏感 |

---

## 从访存角度看

GEMM 的性能核心是减少 global memory 访问，提高片上复用。

### Hold C

```text
A: 多次分块加载
B: 多次分块加载
C: 寄存器中累加，最后写一次
```

优点是避免了反复读写 C。

这是最重要的，因为 C 是累加结果，如果每次 partial sum 都写回 global memory，会非常慢。

---

### Hold A

```text
A: 加载一次，服务多个 B tile
B: 多次加载
C: 多个输出 tile 被更新
```

优势是 A 的 reuse 高。

但是如果 C 不能很好地留在寄存器或 shared memory，就会导致 C 多次读写。

---

### Hold B

```text
B: 加载一次，服务多个 A tile
A: 多次加载
C: 多个输出 tile 被更新
```

优势是 B 的 reuse 高。

在神经网络推理里尤其常见，因为权重矩阵通常可以被多组输入复用。

---

## 从 CUDA 实现角度看

现代 CUDA GEMM 通常不是三者完全割裂，而是**分层组合**。

典型层级是：

```text
Global memory
    ↓
Shared memory tile of A/B
    ↓
Register fragment of A/B
    ↓
Register accumulator C
    ↓
Global memory C
```

所以实际高性能 GEMM 往往是：

* 在 thread/block 层面：hold C；
* 在 shared memory 层面：复用 A 和 B；
* 在 warp/tensor core 层面：使用 A/B fragments；
* 最后统一写回 C。

也就是说，真正的 cuBLAS/CUTLASS 风格实现通常是：

```text
C 在 register 中 hold
A/B 在 shared memory 中 tile reuse
A/B fragment 在 register 中短暂 hold
```

因此，**标准高性能 GEMM 最核心的是 hold C，而 hold A / hold B 更多体现为 block 调度或数据复用方向的选择。**

---

## 优劣总结

### Hold C 最通用

优点：

* C 只写一次；
* 避免 partial sum 频繁访存；
* 适合 Tensor Core；
* 是现代 GEMM 主流。

缺点：

* accumulator register 多；
* tile 太大会降低 occupancy。

**一般标准 GEMM 首选 Hold C。**

---

### Hold A 适合 A 复用明显

优点：

* A tile 加载一次，可用于多个 B tile；
* 适合 A 固定、B 多变的情况。

缺点：

* C 可能分散；
* 累加管理复杂；
* 不如 hold C 通用。

---

### Hold B 适合 B 复用明显

优点：

* B tile 加载一次，可用于多个 A tile；
* 适合权重矩阵固定的推理场景。

缺点：

* C 的分布和写回需要精心设计；
* 对 block mapping 和 memory layout 更敏感。

---

## 一句话理解

**Hold C 是让结果不动，A 和 B 流过来累加；Hold A 是让左操作数不动，多个 B 流过来；Hold B 是让右操作数不动，多个 A 流过来。**

在实际 CUDA GEMM 中，最常见、最高效的基本策略是：

```text
寄存器 hold C，
shared memory 复用 A/B，
根据矩阵形状和业务场景决定更偏向复用 A 还是 B。
```

可以。下面以 **H100 SXM / SXM5，也就是非 PCIe 版本**为例说明。NVIDIA 官方数据表给出的 H100 SXM 规格包括：FP32 60 TFLOPS、FP64 Tensor Core 60 TFLOPS、TF32 Tensor Core 1000 TFLOPS（稀疏）、BF16/FP16 Tensor Core 2000 TFLOPS（稀疏）、FP8 Tensor Core 4000 TFLOPS（稀疏）、80GB 显存、约 3TB/s 显存带宽、700W TDP；数据表也说明这些带 `*` 的 Tensor Core 指标采用稀疏技术，不采用稀疏时约为一半。也就是说，做 dense FP16/BF16 GEMM 时，可用峰值大约是 **1000 TFLOPS** 量级，HBM 带宽约 **3TB/s**。([NVIDIA 图片][1])

---

# 1. 先用 roofline 给 H100 GEMM 定位

Roofline 模型的核心是：

$$P = \min(P_\text{peak},\ I \times B_\text{mem})$$

其中：

* $P_\text{peak}$：计算峰值；
* $B_\text{mem}$：显存带宽；
* $I$：算术强度，单位是 FLOP/Byte。

对 H100 SXM dense FP16/BF16 Tensor Core：

$$P_\text{peak} \approx 1000\ \text{TFLOP/s}$$

$$B_\text{HBM} \approx 3\ \text{TB/s}$$

所以全局 HBM roofline 的拐点约为：

$$I_\text{ridge} =
\frac{1000\ \text{TFLOP/s}}{3\ \text{TB/s}}
\approx 333\ \text{FLOP/Byte}$$

含义是：

* 如果 GEMM 的有效算术强度 (I < 333)，主要受 HBM 带宽限制；
* 如果 (I > 333)，才有机会进入 Tensor Core compute-bound 区域。

对 FP32 CUDA core GEMM：

$$I_\text{ridge} =
\frac{60\ \text{TFLOP/s}}{3\ \text{TB/s}}
= 20\ \text{FLOP/Byte}$$

所以 FP32 GEMM 更容易被算力限制，而 FP16/BF16 Tensor Core GEMM 对数据复用要求高得多。

---

# 2. 理想 GEMM 的算术强度

GEMM：

$$C_{M \times N} = A_{M \times K} B_{K \times N}$$

总计算量：

$$2MNK\ \text{FLOPs}$$

如果理想情况下 A、B、C 都只从 HBM 读写一次，且 A/B/C 都按 FP16 2 bytes 估算，不考虑 beta 读 C，则 HBM 字节数近似为：

$$2(MK + KN + MN)\ \text{Bytes}$$

所以：

$$I_\text{ideal}=
 \frac{2MNK}{2(MK + KN + MN)}=
\frac{MNK}{MK + KN + MN}$$

如果是方阵 (M=N=K=L)：

$$I_\text{ideal}=
\frac{L}{3}$$

因此对 H100 dense FP16/BF16 Tensor Core：

$$\frac{L}{3} > 333
\Rightarrow
L > 999$$

也就是说，**大约 1K 以上的方阵 GEMM，在理想数据复用下就有机会进入 compute-bound**。但这只是理想下界，真实 kernel 是否能接近它，取决于 hold A / hold B / hold C 的数据流设计、L2 命中率、shared memory 复用、寄存器复用、TMA/cp.async 管线、warp-level MMA 排布等。

---

# 3. hold A / hold B / hold C 的本质区别

三者不是数学算法不同，而是 **dataflow 不同**。

| 策略     | 驻留对象          | 主要复用谁         | 典型名称              | 核心目标                     |
| ------ | ------------- | ------------- | ----------------- | ------------------------ |
| hold C | C accumulator | C partial sum | output stationary | 避免 partial C 反复读写 HBM    |
| hold A | A tile        | A operand     | A stationary      | 让一个 A tile 服务多个 B/C tile |
| hold B | B tile        | B operand     | B stationary      | 让一个 B tile 服务多个 A/C tile |

---

# 4. Hold C：output stationary

这是标准高性能 CUDA GEMM 最常见的策略。

## 当前 tile 视图

```text
A tile: BM x BK          B tile: BK x BN

        BK                      BN
   +----------+            +----------+
BM | A_panel  |    x   BK  | B_panel  |
   +----------+            +----------+
             \              /
              \            /
               v          v

             C tile: BM x BN
        +----------------------+
   BM   |   C_acc in register  |
        +----------------------+
                 BN
```

每个 CTA / warp group / warp 负责一个 (C_{BM \times BN}) tile。

伪代码：

```cpp
acc[BM][BN] = 0;

for k0 in 0..K step BK:
    load A[BM][BK];
    load B[BK][BN];
    acc += mma(A, B);

store acc -> C[BM][BN];
```

## 全局任务视图

```text
完整 C = M x N，被切成很多 BM x BN 的 tile

                 N direction
        C0,0     C0,1     C0,2     C0,3
      +--------+--------+--------+--------+
M     | CTA00  | CTA01  | CTA02  | CTA03  |
dir   +--------+--------+--------+--------+
      | CTA10  | CTA11  | CTA12  | CTA13  |
      +--------+--------+--------+--------+
      | CTA20  | CTA21  | CTA22  | CTA23  |
      +--------+--------+--------+--------+

当前计算：例如 CTA11
      A row panel: A[BM of row 1, K]
      B col panel: B[K, BN of col 1]
      C tile:      C[row 1, col 1]
```

换成矩阵视角：

```text
A: M x K                         B: K x N

      K                                N
+-------------+                 +-------------------+
|             |                 |   B0  B1  B2  B3 |
|   A0        |                 |                   |
+-------------+                 |                   |
|>>>A1<<<<<<< |  x              |>>>B1 column tile<<|
+-------------+                 |                   |
|   A2        |                 |                   |
+-------------+                 +-------------------+

当前 CTA11 计算：
A1 row tile  x  B1 column tile  ->  C11
```

## HBM 访存模型

单个 $C_{BM \times BN}$ tile 需要遍历整个 K。

单 CTA 近似读写：

$$\text{Bytes}_C
\approx
s(BM \cdot K + K \cdot BN + BM \cdot BN)$$

其中 (s) 是元素字节数，例如 FP16 为 2 bytes。

计算量：

$$2BM \cdot BN \cdot K$$

所以单 CTA 算术强度近似：

$$I_C
===

\frac{2BM \cdot BN \cdot K}
{s(BM \cdot K + K \cdot BN + BM \cdot BN)}$$

当 (K) 很大时：

$$I_C
\approx
\frac{2BM \cdot BN}
{s(BM + BN)}$$

如果 (BM=BN=128)，FP16 (s=2)：

$$I_C
\approx
\frac{2 \cdot 128 \cdot 128}{2(128+128)}
=64\ \text{FLOP/Byte}$$

这低于 H100 FP16 Tensor Core 的 HBM roofline 拐点 333 FLOP/Byte。所以如果只看“单 CTA 从 HBM 读 A/B”的模型，hold C 仍然像是 HBM-bound。

但真实高性能 GEMM 依赖更高层复用：

* A tile 会被多个不同 N 方向的 CTA 使用；
* B tile 会被多个不同 M 方向的 CTA 使用；
* L2 cache 会捕获跨 CTA 复用；
* shared memory 和 register file 会捕获 CTA 内复用；
* Tensor Core MMA 会把寄存器里的 fragment 重复使用。

所以 **hold C 是必要条件，但不是充分条件**。它解决的是 C partial sum 不落 HBM 的问题。

## 优点

Hold C 的最大优势是：$C_{ij}$的 partial sum 全程在寄存器中累加，最后写一次。

如果不 hold C，则每个 K tile 后都要把 partial C 写回 HBM，再读回来继续累加。假设 K 被切成 (K/BK) 段，则 C 的 HBM 流量会从：

$$O(MN)$$

变成：

$$O(MN \cdot K/BK)$$

这对 H100 这种 Tensor Core 算力极高的芯片非常致命。

## 缺点

主要问题是 accumulator 占寄存器。

例如一个 warp group 或 thread block 内部如果要同时 hold 很多 (C) fragment，会带来：

* register pressure；
* occupancy 下降；
* register spill 风险；
* tile shape 受限。

---

# 5. Hold A：A stationary

Hold A 的思路是：**让一个 A tile 在片上停留更久，用它去乘多个 B tile，生成多个 C tile。**

## 当前 tile 视图

```text
固定 A tile，横向扫多个 B tile

                   B0        B1        B2        B3
                +-------+ +-------+ +-------+ +-------+
                |       | |       | |       | |       |
                +-------+ +-------+ +-------+ +-------+
                    |         |         |         |
                    v         v         v         v

A tile        ->   C0        C1        C2        C3
+-------+        +-------+ +-------+ +-------+ +-------+
| hold  |        | acc0  | | acc1  | | acc2  | | acc3  |
|  A    |        +-------+ +-------+ +-------+ +-------+
+-------+
```

## 全局任务视图

```text
C 被切成 BM x BN tile。
hold A 相当于一个 A row panel 服务同一行上的多个 C tile。

                 N direction
        C0,0     C0,1     C0,2     C0,3
      +--------+--------+--------+--------+
row 0 |        |        |        |        |
      +--------+--------+--------+--------+
row 1 |  C10   |  C11   |  C12   |  C13   |  <<< 当前 A row panel 负责这一整排的一组 tile
      +--------+--------+--------+--------+
row 2 |        |        |        |        |
      +--------+--------+--------+--------+

当前驻留：
A[row 1, K block]

流入：
B[K block, col 0]
B[K block, col 1]
B[K block, col 2]
B[K block, col 3]

更新：
C10, C11, C12, C13
```

## HBM 访存模型

假设一次 hold A 服务 (P) 个 N 方向的 B/C tile。

也就是一个 (BM \times BK) 的 A tile，配合 (P) 个 (BK \times BN) 的 B tile。

计算量：

$$2BM \cdot (PBN) \cdot K$$

近似 HBM 字节数：

$$s(BM \cdot K + P \cdot K \cdot BN + P \cdot BM \cdot BN)$$

当 (K) 很大：

$$I_A
\approx
\frac{2BM \cdot PBN}
{s(BM + PBN)}$$

如果 (BM=BN=128)，FP16 (s=2)：

$$I_A
\approx
\frac{2 \cdot 128 \cdot P \cdot 128}
{2(128 + 128P)}
===============

\frac{128P}{1+P}$$

不同 (P) 下：

|  P | 含义                      |           近似 AI |
| -: | ----------------------- | --------------: |
|  1 | 退化成普通 hold C tile       |    64 FLOP/Byte |
|  2 | 一个 A tile 服务 2 个 B tile |  85.3 FLOP/Byte |
|  4 | 一个 A tile 服务 4 个 B tile | 102.4 FLOP/Byte |
|  8 | 一个 A tile 服务 8 个 B tile | 113.8 FLOP/Byte |
|  ∞ | 理论极限                    |   128 FLOP/Byte |

可以看到，**hold A 能提高 A 的复用，提升算术强度，但在固定 BM/BN 下提升有上限**。它不能无限提高 roofline 位置，因为 B 和 C 仍然要流动。

## 优点

适合 A 复用明显的场景，例如：

$$C_i = A \times B_i$$

也就是 A 固定，多个 B 不同。

典型场景：

* 一个输入激活 tile 复用到多个专家/多个 projection；
* 某些 batched GEMM 中 A 共享；
* N 方向很大，A row panel 可以横向复用。

## 缺点

问题在于 C accumulator 数量膨胀。

如果 hold A 同时更新 (P) 个 C tile，那么要么：

1. 同时 hold (P) 份 C accumulator，寄存器压力大；
2. 分批更新 C，导致 A 的驻留收益下降；
3. 把 partial C 写到 shared/global，再回来归约，增加访存。

所以 hold A 的收益通常受限于：

* register file 容量；
* shared memory 容量；
* warp/CTA scheduling；
* N 方向 tile group 的大小；
* C accumulator 是否能同时驻留。

---

# 6. Hold B：B stationary

Hold B 与 hold A 对称。它让一个 B tile 在片上停留更久，用多个 A tile 去乘它，生成多个 C tile。

## 当前 tile 视图

```text
固定 B tile，纵向扫多个 A tile

              B tile
            +---------+
            | hold B  |
            +---------+
                ^
                |
+-------+    +-------+
|  A0   | -> |  C0   |
+-------+    +-------+

+-------+    +-------+
|  A1   | -> |  C1   |
+-------+    +-------+

+-------+    +-------+
|  A2   | -> |  C2   |
+-------+    +-------+

+-------+    +-------+
|  A3   | -> |  C3   |
+-------+    +-------+
```

## 全局任务视图

```text
C 被切成 BM x BN tile。
hold B 相当于一个 B column panel 服务同一列上的多个 C tile。

                 N direction
        C0,0     C0,1     C0,2     C0,3
      +--------+--------+--------+--------+
row 0 |        |  C01   |        |        |
      +--------+--------+--------+--------+
row 1 |        |  C11   |        |        |
      +--------+--------+--------+--------+
row 2 |        |  C21   |        |        |
      +--------+--------+--------+--------+
row 3 |        |  C31   |        |        |
      +--------+--------+--------+--------+
                 ^ 
                 |
          当前 B column panel
          服务这一整列的一组 C tile
```

## HBM 访存模型

假设一次 hold B 服务 (Q) 个 M 方向的 A/C tile。

计算量：

$$2(QBM) \cdot BN \cdot K$$

近似 HBM 字节数：

$$s(Q \cdot BM \cdot K + K \cdot BN + Q \cdot BM \cdot BN)$$

当 (K) 很大：

$$I_B
\approx
\frac{2QBM \cdot BN}
{s(QBM + BN)}$$

如果 (BM=BN=128)，FP16：

$$I_B
\approx
\frac{128Q}{1+Q}$$

这和 hold A 对称。

|  Q | 含义                      |           近似 AI |
| -: | ----------------------- | --------------: |
|  1 | 退化成普通 hold C tile       |    64 FLOP/Byte |
|  2 | 一个 B tile 服务 2 个 A tile |  85.3 FLOP/Byte |
|  4 | 一个 B tile 服务 4 个 A tile | 102.4 FLOP/Byte |
|  8 | 一个 B tile 服务 8 个 A tile | 113.8 FLOP/Byte |
|  ∞ | 理论极限                    |   128 FLOP/Byte |

## 优点

适合 B 复用明显的场景，例如深度学习推理/训练中的权重矩阵：

$$C_i = A_i \times W$$

其中 (W) 相当于 B，多个输入 batch 或 token 共享同一个权重矩阵。

这种情况下 hold B 很有吸引力，因为权重 B 的复用率高。

## 缺点

和 hold A 类似，问题变成 C 的管理。

如果一个 B tile 服务多个 A tile，则会同时产生多个 M 方向上的 C tile。为了不增加 C 访存，需要 hold 更多 accumulator，导致：

* register pressure 增加；
* CTA/warp group 数量减少；
* occupancy 下降；
* M 方向任务划分更复杂。

---

# 7. 三者的 roofline 对比

以 FP16/BF16、(BM=BN=128)、H100 SXM dense Tensor Core 为例：

| 策略                | 片上驻留重点                |         理论 AI 变化 | 相对 H100 FP16 ridge point       |
| ----------------- | --------------------- | ---------------: | ------------------------------ |
| hold C            | C accumulator         |   约 64 FLOP/Byte | 低于 333，单 CTA 视角仍偏 memory-bound |
| hold A, P=4       | A tile 复用到 4 个 N tile |  约 102 FLOP/Byte | 仍低于 333                        |
| hold A, P→∞       | A 复用极限                |  约 128 FLOP/Byte | 仍低于 333                        |
| hold B, Q=4       | B tile 复用到 4 个 M tile |  约 102 FLOP/Byte | 仍低于 333                        |
| hold B, Q→∞       | B 复用极限                |  约 128 FLOP/Byte | 仍低于 333                        |
| 理想全局 GEMM, L=1024 | A/B/C 全局近似只搬一次        |  约 341 FLOP/Byte | 接近/略高于 333                     |
| 理想全局 GEMM, L=4096 | A/B/C 全局近似只搬一次        | 约 1365 FLOP/Byte | 明显 compute-bound               |

这个表的重点是：

**单个 CTA 的 hold A/B/C 只能解释局部数据流，不能单独解释 H100 上大 GEMM 为什么能接近 compute-bound。**

真正的大 GEMM 需要多层复用叠加：

```text
HBM
  ↓
L2 cache / cluster-level reuse
  ↓
shared memory tile
  ↓
warp-level fragment
  ↓
register accumulator C
  ↓
Tensor Core MMA
```

因此在 H100 上，实际高性能 GEMM 通常不是单纯 hold A、hold B 或 hold C，而是：

```text
register:      hold C accumulator
shared memory: reuse A and B tiles
L2/cluster:    improve A/B cross-CTA reuse
scheduler:     根据 M/N/K 形状选择偏 A-stationary 或 B-stationary 的任务映射
```

---

# 8. 三种策略的核心异同

## 相同点

三者都在计算同一个 GEMM：

$$C = A \times B$$

也都要做三件事：

1. 从 HBM 搬 A/B/C；
2. 在片上复用 A/B/C；
3. 用 Tensor Core 执行 MMA。

区别不是数学，而是数据驻留顺序。

---

## 不同点

### hold C

关注点是：

```text
不要让 partial C 落到 HBM
```

这是最基础、最通用、最重要的优化。

适合：

* 标准大 GEMM；
* cuBLAS/CUTLASS 常规 tile；
* K 很大，需要长链累加的情况。

主要瓶颈：

```text
register accumulator 太多
```

---

### hold A

关注点是：

```text
A 已经加载进来了，尽量多用几次
```

适合：

* A 固定、B 多变；
* N 方向很宽；
* 一个 A panel 可以服务多个 C column tile。

主要瓶颈：

```text
同时更新多个 C tile，accumulator 压力上升
```

---

### hold B

关注点是：

```text
B 已经加载进来了，尽量多用几次
```

适合：

* B 是权重矩阵；
* A 是多个 batch/token；
* M 方向很高；
* 推理或训练中多个输入共享同一组权重。

主要瓶颈：

```text
同时更新多个 M 方向 C tile，寄存器和调度压力上升
```

---

# 9. H100 上的实践判断

对 H100 这种 Tensor Core 算力极高、HBM 带宽也很高但相对算力仍然紧张的 GPU，dense FP16/BF16 GEMM 的优化核心是：

```text
尽可能把有效 AI 推到 333 FLOP/Byte 以上
```

所以：

## 小矩阵 / skinny GEMM

例如 (M) 很小或 (N) 很小：

* 理想 AI 本身不高；
* CTA 间复用不足；
* HBM/L2/TMA 开销占比高；
* hold A 或 hold B 的任务映射可能比单纯 hold C 更重要。

例如：

```text
M 小，N 大：偏 hold A
M 大，N 小：偏 hold B
```

## 大方阵 GEMM

例如 (M=N=K \ge 4096)：

* 理想 AI 很高；
* 只要实现足够好，通常更容易 compute-bound；
* hold C 是核心；
* A/B 的复用主要靠 shared memory、L2、cluster scheduling、TMA pipeline 完成。

## 权重固定的推理 GEMM

例如：

$$Y = XW$$

如果 W 在多个 batch/token 上复用，B-stationary 很有价值。

但 kernel 内部仍然通常会 hold C accumulator，因为 C partial sum 不落 HBM 是基本要求。

---

# 10. 更完整的“全局任务 + 当前计算”图示

你说得对，刚才的图只展示了当前计算视图，不够完整。下面这个图更适合看全局任务。

## Hold C 的完整视图

```text
Global GEMM:

A[M,K]                         B[K,N]
+-------------------+          +-----------------------------+
| A row tile 0      |          | B col tile 0 | B col tile 1 |
+-------------------+          |--------------+--------------|
| A row tile 1 >>>  |          | B col tile 2 | B col tile 3 |
+-------------------+          +-----------------------------+
| A row tile 2      |
+-------------------+

C[M,N]
+------------+------------+------------+------------+
| C00        | C01        | C02        | C03        |
+------------+------------+------------+------------+
| C10        | C11  <<<   | C12        | C13        |
+------------+------------+------------+------------+
| C20        | C21        | C22        | C23        |
+------------+------------+------------+------------+

当前 CTA:
C11 = A row tile 1 x B col tile 1

当前计算过程中:
A row tile 1 按 K 分块流入
B col tile 1 按 K 分块流入
C11 accumulator 一直 hold 在 register
```

---

## Hold A 的完整视图

```text
Global GEMM:

当前 hold 的 A panel:
A row tile 1
+-------------------+
|                   |
+-------------------+
| A row tile 1 <<<  |
+-------------------+
|                   |
+-------------------+

它横向服务多个 B column tile:

B[K,N]
+----------+----------+----------+----------+
| B0       | B1       | B2       | B3       |
| stream   | stream   | stream   | stream   |
+----------+----------+----------+----------+

更新 C 的一整行 tile group:

C[M,N]
+------------+------------+------------+------------+
| C00        | C01        | C02        | C03        |
+------------+------------+------------+------------+
| C10 <<<    | C11 <<<    | C12 <<<    | C13 <<<    |
+------------+------------+------------+------------+
| C20        | C21        | C22        | C23        |
+------------+------------+------------+------------+
```

直观理解：

```text
hold A = 固定 C 的某一行 tile group，横向扫 N
```

---

## Hold B 的完整视图

```text
Global GEMM:

当前 hold 的 B panel:
B column tile 1

B[K,N]
+----------+----------+----------+----------+
| B0       | B1 <<<   | B2       | B3       |
|          | hold B   |          |          |
+----------+----------+----------+----------+

它纵向服务多个 A row tile:

A[M,K]
+-------------------+
| A0 stream         |
+-------------------+
| A1 stream         |
+-------------------+
| A2 stream         |
+-------------------+
| A3 stream         |
+-------------------+

更新 C 的一整列 tile group:

C[M,N]
+------------+------------+------------+------------+
| C00        | C01 <<<    | C02        | C03        |
+------------+------------+------------+------------+
| C10        | C11 <<<    | C12        | C13        |
+------------+------------+------------+------------+
| C20        | C21 <<<    | C22        | C23        |
+------------+------------+------------+------------+
| C30        | C31 <<<    | C32        | C33        |
+------------+------------+------------+------------+
```

直观理解：

```text
hold B = 固定 C 的某一列 tile group，纵向扫 M
```

---

# 11. 一句话总结

在 H100 SXM 上，dense FP16/BF16 Tensor Core GEMM 的 HBM roofline 拐点大约是 **333 FLOP/Byte**。单纯一个 (128 \times 128) CTA tile 的 hold C 只有约 **64 FLOP/Byte**，hold A/B 通过扩大 A 或 B 的复用可以提高到约 **100–128 FLOP/Byte** 的量级，但仍不足以单独解释大 GEMM 的高性能。真正高性能实现依赖 **hold C accumulator + shared memory 复用 A/B + L2/cluster 跨 CTA 复用 + 合理任务映射**。

因此：

```text
hold C：最通用，解决 C partial sum 访存问题；
hold A：适合 A 复用强、横向 N tile 多的任务；
hold B：适合 B/权重复用强、纵向 M tile 多的任务；
H100 上实际 kernel 往往是三者分层组合，而不是三选一。
```

[1]: https://images.nvidia.cn/aem-dam/en-zz/Solutions/data-center/h100/nvidia-h100-datasheet-nvidia-a4-2287922-r7-zhCN.pdf "NVIDIA H100 | Tensor Core GPU"
