---
title: cuda_basics
date: 2026-05-27
tags:
  - #LLM
  - #cuda
  - #笔记
  - #from_me
  - #待整理
status: 更新中
---

下面我按 CUDA 小白视角，从“程序怎么跑”到“矩阵乘法怎么加速”系统讲一遍。你可以先建立一个总图：

```text
CUDA 程序
  ↓
Grid：一次 kernel 启动的全部任务
  ↓
CTA / Thread Block：一个线程块，调度到一个 SM 上执行
  ↓
Warp：32 个线程为一组执行
  ↓
Thread：单个 CUDA 线程

H100 硬件
  ↓
SM：流式多处理器，真正干活的计算单元集合
  ├── CUDA Core：通用标量 / 向量计算单元
  ├── Tensor Core：矩阵乘加专用单元
  ├── Shared Memory：片上共享内存
  ├── Register File：寄存器
  └── TMA：Hopper 上的数据搬运引擎
```

---

# 1. CTA 是什么？

CTA 的全称是 **Cooperative Thread Array**，在 CUDA 编程里通常就等价于 **thread block**，也就是线程块。

你写 CUDA kernel 时经常会看到：

```cpp
dim3 grid(...);
dim3 block(...);
kernel<<<grid, block>>>(...);
```

这里：

```text
grid  = 全局有多少个 CTA / block
block = 每个 CTA 里有多少个 thread
```

例如：

```cpp
dim3 block(256);
dim3 grid(1024);
```

表示启动：

```text
1024 个 CTA
每个 CTA 256 个 thread
总共 1024 × 256 个 CUDA thread
```

## CTA 在 GEMM 里干什么？

以矩阵乘法：

$$C = A \times B$$

为例，通常一个 CTA 负责计算 C 的一个小块。

比如：

```text
C 是 M × N 大矩阵
每个 CTA 计算一个 128 × 128 的 C tile
```

图示：

```text
完整 C 矩阵：

+---------+---------+---------+---------+
| CTA 0,0 | CTA 0,1 | CTA 0,2 | CTA 0,3 |
+---------+---------+---------+---------+
| CTA 1,0 | CTA 1,1 | CTA 1,2 | CTA 1,3 |
+---------+---------+---------+---------+
| CTA 2,0 | CTA 2,1 | CTA 2,2 | CTA 2,3 |
+---------+---------+---------+---------+
```

某个 CTA，比如 `CTA 1,2`，负责：

```text
C 的第 1 行 tile，第 2 列 tile
```

它会从 A 里取对应的 row panel，从 B 里取对应的 column panel，然后不断累加。

---

# 2. Warp 是什么？

一个 CTA 里有很多 thread，但 GPU 不是一个 thread 一个 thread 执行，而是以 **warp** 为基本执行单位。

```text
1 warp = 32 threads
```

例如一个 CTA 有 256 个线程：

```text
256 threads = 8 warps
```

CUDA Core、Tensor Core 的很多指令都是以 warp 或 warp group 为单位组织的。

在 GEMM 里：

```text
CTA 负责一个大的 C tile
warp 负责 CTA 内部的一个小 C tile
thread 负责更小的元素或 fragment
```

比如：

```text
CTA tile: 128 × 128
  ↓
warp tile: 64 × 64 或 64 × 32
  ↓
thread fragment: 若干寄存器中的小片段
```

---

# 3. SM 是什么？

SM 是 **Streaming Multiprocessor**，可以理解成 GPU 上真正执行 CUDA block 的计算岛。

一个 GPU 有很多 SM。H100 SXM 有很多个 SM，kernel 启动后，大量 CTA 会被分配到不同 SM 上执行。

简化理解：

```text
GPU = 很多个 SM
SM = 执行 CTA 的地方
CTA = 被调度到 SM 上运行的任务块
```

关系大概是：

```text
GPU
+------+ +------+ +------+ +------+
| SM 0 | | SM 1 | | SM 2 | | SM 3 |
+------+ +------+ +------+ +------+
   ↑       ↑        ↑        ↑
 CTA      CTA      CTA      CTA
```

一个 SM 上可以同时驻留多个 CTA，但数量受到资源限制，例如：

* 每个 CTA 用了多少 thread；
* 每个 CTA 用了多少 shared memory；
* 每个 thread 用了多少 register；
* CTA 内 warp 数量；
* 硬件最大 CTA 数限制。

这就引出一个常见概念：**occupancy**，即一个 SM 上能同时塞进多少 warp / CTA。

---

# 4. CUDA Core 是什么？

CUDA Core 是 GPU 上的通用计算核心。

它适合做：

* 普通 FP32 加减乘除；
* 整数计算；
* 地址计算；
* 分支判断；
* index 计算；
* 标量运算；
* 一些非矩阵密集型计算。

可以粗略理解成：

```text
CUDA Core ≈ 通用 ALU
Tensor Core ≈ 矩阵乘加专用加速器
```

在 GEMM 里，如果不用 Tensor Core，那么每个元素类似这样算：

```cpp
float acc = 0;
for (int k = 0; k < K; ++k) {
    acc += A[m][k] * B[k][n];
}
C[m][n] = acc;
```

这类标量 FMA 主要靠 CUDA Core 执行。

## CUDA Core 的特点

优点：

* 灵活；
* 支持很多通用计算；
* 适合不规则逻辑；
* 编程模型直接。

缺点：

* 做大规模矩阵乘法时吞吐远低于 Tensor Core；
* 对 FP16/BF16/FP8 GEMM 来说，CUDA Core 通常不是最优选择。

---

# 5. Tensor Core 是什么？

Tensor Core 是 NVIDIA GPU 上专门为矩阵乘加设计的硬件单元。

它执行的不是单个：

$$a \times b + c$$

而是一小块矩阵乘加：

$$D = A \times B + C$$

也就是 MMA：

```text
MMA = Matrix Multiply-Accumulate
```

你可以把 Tensor Core 理解成硬件版的小矩阵乘法机器。

例如某条 MMA 指令可能做类似：

```text
16 × 8 × 16
```

或其他形状的小矩阵乘加。程序员一般不会手写每一个乘法，而是让 warp 把 A/B fragment 喂给 Tensor Core。

## Tensor Core 在 GEMM 中的位置

```text
Global Memory
   ↓
Shared Memory
   ↓
Register fragment
   ↓
Tensor Core MMA
   ↓
Register accumulator C
   ↓
Global Memory
```

Tensor Core 不直接从 HBM/global memory 里拿大矩阵。数据要先经过：

1. global memory；
2. shared memory；
3. register fragment；
4. Tensor Core。

---

# 6. Tensor Core 和 CUDA Core 的区别

| 对比项     | CUDA Core             | Tensor Core                          |
| ------- | --------------------- | ------------------------------------ |
| 主要用途    | 通用计算                  | 矩阵乘加                                 |
| 执行粒度    | 标量 / 向量操作             | 小矩阵 MMA                              |
| 灵活性     | 高                     | 相对低                                  |
| GEMM 性能 | 较低                    | 极高                                   |
| 适合任务    | indexing、激活函数、规约、普通算术 | GEMM、卷积、attention、MLP                |
| 常见数据类型  | FP32、FP64、INT 等       | FP16、BF16、TF32、FP8、INT8、FP64 等，取决于架构 |

在深度学习里，大部分耗时在：

```text
矩阵乘法 / 卷积 / attention
```

这些都可以转成 GEMM 或 batched GEMM，所以 Tensor Core 非常重要。

---

# 7. TMA 是什么？

TMA 是 **Tensor Memory Accelerator**，Hopper 架构，也就是 H100 这一代引入的重要特性。

它的作用是：**高效搬运多维 tensor tile，尤其是在 global memory 和 shared memory 之间搬数据**。

NVIDIA 对 Hopper 的介绍中提到，TMA 可以在 global memory 和 shared memory 之间高效传输大块数据和多维 tensor，也支持 cluster 中 thread block 之间的异步拷贝；它的目标是更好地喂饱 H100 强大的 Tensor Core。([NVIDIA Developer][1])

## 没有 TMA 时

以前很多 kernel 里，数据搬运通常由一堆 thread 手动做：

```cpp
shared_A[tid] = global_A[index];
shared_B[tid] = global_B[index];
```

也就是说：

```text
很多 CUDA thread 既要负责计算，又要负责搬数据
```

这会带来几个问题：

* 地址计算复杂；
* thread 被搬运任务占用；
* 多维 tensor tile 搬运麻烦；
* shared memory layout 需要手工处理；
* 很难和 Tensor Core 计算完全重叠。

## 有 TMA 时

TMA 更像一个专门的数据搬运引擎：

```text
TMA：你告诉我 tensor 形状、stride、tile 位置
我帮你把一块多维数据搬到 shared memory
```

简化图：

```text
HBM / Global Memory
        |
        |  TMA 异步搬运
        v
Shared Memory
        |
        |  warp / Tensor Core 使用
        v
Register / Tensor Core
```

这样 CTA 内的一部分线程不需要手动搬大量数据，可以更专注于计算。

## TMA 在 GEMM 里的作用

H100 Tensor Core 很快，问题经常变成：

```text
数据来不来得及送到 Tensor Core？
```

TMA 的作用就是帮助完成：

```text
下一块 A/B tile 提前搬进 shared memory
当前 A/B tile 正在被 Tensor Core 计算
```

这叫 pipeline：

```text
时间线：

阶段 0: TMA 搬 A0/B0
阶段 1: Tensor Core 算 A0/B0，同时 TMA 搬 A1/B1
阶段 2: Tensor Core 算 A1/B1，同时 TMA 搬 A2/B2
阶段 3: Tensor Core 算 A2/B2，同时 TMA 搬 A3/B3
```

目标是隐藏访存延迟。

---

# 8. Shared Memory 是什么？

Shared memory 是 SM 内部的一块片上内存。

特点：

* 比 global memory 快得多；
* 由同一个 CTA 内的 thread 共享；
* 程序员可控；
* 容量有限；
* 常用于 tile 复用。

在 GEMM 中：

```text
A/B 从 global memory 读入 shared memory
CTA 内多个 warp 重复使用这些 A/B tile
```

示意：

```text
Global A/B
   ↓
Shared A/B tile
   ↓
多个 warp 反复读取
   ↓
Tensor Core MMA
```

如果没有 shared memory，A/B 可能会被反复从 HBM 读取，效率很低。

---

# 9. Register 是什么？

Register 是每个 thread 私有的最快存储。

在高性能 GEMM 中，最常见做法是：

```text
C accumulator 放在 register 里
```

也就是前面说的 hold C。

例如：

```cpp
float acc[8];  // 实际上每个 thread 持有若干 C fragment
for (...) {
    acc += mma(A_frag, B_frag);
}
store acc to C;
```

这样 C 的 partial sum 不需要每轮都写回 global memory。

但是 register 太多也有问题：

```text
每个 thread 用 register 越多
SM 上能同时驻留的 warp 越少
occupancy 下降
```

所以 GEMM kernel 要在：

```text
tile 大小、register 使用量、shared memory 使用量、occupancy
```

之间做平衡。

---

# 10. H100 上矩阵乘法的数据类型支持

H100 属于 Hopper 架构。NVIDIA H100 官方数据表列出了 H100 的 Tensor Core 能力，包括 FP64、TF32、FP16、BF16、FP8、INT8 等吞吐；同时数据表说明带星号的指标采用稀疏技术，不采用稀疏时约为一半。([NVIDIA][2])

下面按“矩阵乘法常见数据类型”解释。

---

## FP32 GEMM

### 含义

A、B、C 都是 32-bit float。

```text
float32 × float32 → float32
```

### 通常用什么硬件？

有两种路线：

1. CUDA Core FP32；
2. Tensor Core TF32。

在 NVIDIA Ampere/Hopper 上，很多深度学习框架默认会把 FP32 GEMM 用 TF32 Tensor Core 加速。

---

## TF32 GEMM

TF32 是 NVIDIA 为 Tensor Core 设计的一种格式。

它对程序员看起来像 FP32 输入，但 Tensor Core 内部用 TF32 精度做乘法，通常 FP32 累加。

```text
TF32 input × TF32 input → FP32 accumulate
```

特点：

* 比纯 FP32 CUDA Core 快很多；
* 精度低于严格 FP32；
* 深度学习训练中通常可接受；
* 科学计算中要谨慎。

在 PyTorch 里以前常见设置：

```python
torch.backends.cuda.matmul.allow_tf32 = True
```

新版本也有更细的 matmul precision 设置。

---

## FP16 GEMM

FP16 是 16-bit half。

常见 Tensor Core 模式：

```text
FP16 input × FP16 input → FP32 accumulate
```

或者某些场景：

```text
FP16 input × FP16 input → FP16 accumulate
```

深度学习中通常使用 FP32 accumulate，因为训练更稳定。

优点：

* 吞吐高；
* 显存占用小；
* 带宽压力低；
* Tensor Core 支持很好。

缺点：

* 动态范围小；
* 容易 overflow / underflow；
* 训练通常需要 loss scaling 或混合精度策略。

---

## BF16 GEMM

BF16 是 bfloat16。

它也是 16-bit，但和 FP16 不一样：

```text
BF16: 指数位多，尾数位少
FP16: 指数位少，尾数位多
```

直觉上：

```text
BF16 动态范围接近 FP32，但精度更粗
FP16 精度略细，但动态范围更小
```

常见 Tensor Core 模式：

```text
BF16 input × BF16 input → FP32 accumulate
```

优点：

* 比 FP16 更不容易 overflow；
* 训练更稳定；
* 大模型训练常用。

缺点：

* 尾数精度比 FP16 粗；
* 某些数值敏感任务要评估。

---

## FP8 GEMM

H100 的一个重要卖点就是 FP8 Tensor Core。Hopper 引入 FP8 Tensor Core，并支持 Transformer Engine，用于在 FP8 和高精度之间自动管理缩放与精度选择。([NVIDIA Developer][1])

FP8 常见有两种格式：

```text
E4M3：4 位指数，3 位尾数
E5M2：5 位指数，2 位尾数
```

一般直觉：

```text
E4M3：精度稍好，动态范围较小
E5M2：动态范围更大，精度更粗
```

常见模式：

```text
FP8 input × FP8 input → FP16 或 FP32 accumulate
```

FP8 的主要作用：

* 降低显存占用；
* 降低带宽压力；
* 提高 Tensor Core 吞吐；
* 加速大模型训练/推理。

但 FP8 不是简单把数据改成 8 位就行。它通常需要：

* scale factor；
* 动态缩放；
* per-tensor 或 per-channel scaling；
* 框架支持；
* Transformer Engine 或类似机制。

---

## INT8 GEMM

INT8 常用于推理。

常见模式：

```text
INT8 input × INT8 input → INT32 accumulate
```

最后再做：

```text
scale / dequantize / requantize
```

INT8 优点：

* 吞吐高；
* 显存和带宽压力低；
* 推理性价比高。

缺点：

* 需要量化；
* 精度依赖 calibration / QAT；
* 对 outlier 敏感；
* 训练中一般不用 INT8 GEMM 做主路径。

---

## FP64 GEMM

FP64 是 double。

H100 也支持 FP64，包括 FP64 Tensor Core 能力，适合 HPC 科学计算。H100 数据表中列出了 FP64 和 FP64 Tensor Core 的性能指标。([NVIDIA][2])

常见于：

* 数值模拟；
* 线性代数；
* 科学计算；
* 高精度求解器。

缺点：

* 对 AI workload 来说通常太贵；
* 吞吐低于低精度 Tensor Core。

---

# 11. 数据类型支持总表

下面是 CUDA 小白可以先记住的版本：

| 数据类型 | 常见用途          | 典型计算模式                | 主要硬件                    | 备注                        |
| ---- | ------------- | --------------------- | ----------------------- | ------------------------- |
| FP64 | HPC 科学计算      | FP64 × FP64 → FP64    | CUDA Core / Tensor Core | H100 支持 FP64 Tensor Core  |
| FP32 | 通用训练 / 科学计算   | FP32 × FP32 → FP32    | CUDA Core               | 精度高但慢                     |
| TF32 | FP32 深度学习加速   | TF32 × TF32 → FP32    | Tensor Core             | 输入像 FP32，用 Tensor Core 加速 |
| FP16 | 深度学习训练/推理     | FP16 × FP16 → FP32    | Tensor Core             | 混合精度常用                    |
| BF16 | 大模型训练         | BF16 × BF16 → FP32    | Tensor Core             | 动态范围更友好                   |
| FP8  | H100 大模型训练/推理 | FP8 × FP8 → FP16/FP32 | Tensor Core             | 需要 scaling / TE 支持        |
| INT8 | 推理量化          | INT8 × INT8 → INT32   | Tensor Core             | 推理常用                      |

---

# 12. GEMM 在 H100 上大概怎么跑？

以 BF16 GEMM 为例：

$$C = A \times B$$

大致流程是：

```text
1. Grid 切分整个 C 矩阵
   每个 CTA 负责一个 C tile

2. CTA 被调度到 SM 上

3. CTA 内部的 warp 协作

4. TMA 把 A/B tile 从 global memory 搬到 shared memory

5. warp 从 shared memory 取 A/B fragment 到 register

6. Tensor Core 执行 MMA

7. C accumulator 一直在 register 里累加

8. K 方向全部算完后，把 C 写回 global memory
```

图示：

```text
完整矩阵 C
+---------+---------+---------+
| CTA 0,0 | CTA 0,1 | CTA 0,2 |
+---------+---------+---------+
| CTA 1,0 | CTA 1,1 | CTA 1,2 |
+---------+---------+---------+

某个 CTA 内部：

Global Memory A/B
        |
        |  TMA / 异步 copy
        v
Shared Memory A/B tile
        |
        |  warp load fragment
        v
Register A/B fragments
        |
        |  Tensor Core MMA
        v
Register C accumulator
        |
        |  store
        v
Global Memory C
```

---

# 13. 为什么 GEMM 要分 tile？

因为完整矩阵太大，放不进片上资源。

比如：

```text
A: 4096 × 4096
B: 4096 × 4096
C: 4096 × 4096
```

不可能整个 A/B/C 都放进 shared memory 或 register。

所以要切块：

```text
C 被切成 128 × 128 tile
A/B 沿 K 方向切成 128 × 64、64 × 128 等 tile
每次只搬一小块进 shared memory
```

简化图：

```text
C tile = A row tile × B column tile

A[M,K]                     B[K,N]
+----------------+         +----------------+
|                |         |      B tile    |
|    A tile      |    ×    |      B tile    |
|                |         |      B tile    |
+----------------+         +----------------+

              ↓

          C tile
      +------------+
      |            |
      +------------+
```

---

# 14. 从 hold C / hold A / hold B 重新理解这些硬件

现在把前面的问题串起来。

## Hold C

```text
C accumulator 放 register
A/B 通过 shared memory 和 Tensor Core 流入
```

对应硬件：

```text
register 负责 hold C
Tensor Core 负责 MMA
TMA/shared memory 负责喂 A/B
```

这是最常见的高性能 GEMM 基础。

---

## Hold A

```text
A tile 尽量留在 shared memory / register
让它服务多个 B tile
```

对应硬件：

```text
shared memory 或 register 复用 A
多个 B tile 通过 TMA 流入
更新多个 C tile
```

适合 A 被重复使用的任务。

---

## Hold B

```text
B tile 尽量留在 shared memory / register
让它服务多个 A tile
```

对应硬件：

```text
shared memory 或 register 复用 B
多个 A tile 通过 TMA 流入
更新多个 C tile
```

适合 B 是权重矩阵、被多个输入复用的任务。

---

# 15. CUDA 小白最容易混淆的几个点

## 误区 1：CTA 是硬件吗？

不是。

```text
CTA / block 是 CUDA 编程模型里的任务单位
SM 是硬件执行单位
```

CTA 被调度到 SM 上执行。

---

## 误区 2：Tensor Core 会自动加速所有代码吗？

不会。

你写普通：

```cpp
for (...) {
    acc += a * b;
}
```

不一定会用 Tensor Core。

通常需要：

* 使用 cuBLAS / cuBLASLt；
* 使用 CUTLASS；
* 使用 WMMA / MMA intrinsic；
* 使用深度学习框架触发 Tensor Core kernel；
* 数据类型和矩阵 layout 满足要求。

---

## 误区 3：FP32 就一定用 CUDA Core？

不一定。

在现代 NVIDIA GPU 上，FP32 深度学习 GEMM 很多时候会走 TF32 Tensor Core。

也就是说：

```text
你以为是 FP32 GEMM
实际可能是 TF32 Tensor Core GEMM
```

这取决于框架设置和库配置。

---

## 误区 4：FP8 只是把数据存成 8 bit？

不是。

FP8 通常必须配合 scaling。

否则数值范围和精度都很容易出问题。

所以 FP8 更像是一套：

```text
数据格式 + 缩放策略 + Tensor Core 支持 + 框架策略
```

---

## 误区 5：TMA 是计算单元吗？

不是。

TMA 主要是数据搬运引擎，不是做矩阵乘法的单元。

它的作用是：

```text
更快、更省线程地把 tensor tile 搬到 shared memory
```

真正做矩阵乘加的是 Tensor Core 或 CUDA Core。

---

# 16. 学习顺序建议

你可以按这个顺序理解 CUDA GEMM：

```text
第一层：CUDA 编程模型
thread → warp → CTA/block → grid

第二层：GPU 硬件
SM → CUDA Core / Tensor Core / register / shared memory

第三层：矩阵乘法切块
C tile → A tile / B tile → K loop

第四层：数据流
global memory → shared memory → register → Tensor Core → register → global memory

第五层：性能模型
roofline → 算术强度 → 带宽瓶颈 / 计算瓶颈

第六层：高级优化
TMA → pipeline → warp specialization → cluster → persistent kernel
```

---

# 17. 最后用一句话串起来

在 H100 上做高性能 GEMM，可以这样理解：

```text
Grid 把整个 C 矩阵切给很多 CTA；
每个 CTA 在一个 SM 上计算一个 C tile；
TMA 把 A/B tile 从 HBM 搬到 shared memory；
warp 把 A/B fragment 喂给 Tensor Core；
Tensor Core 做高速矩阵乘加；
C 的 partial sum 放在 register 里 hold 住；
最后写回 global memory。
```

而 CUDA Core、Tensor Core、TMA 的分工是：

```text
CUDA Core：通用计算、地址、控制、普通算术
Tensor Core：矩阵乘加主力
TMA：高效搬运 tensor tile
Shared Memory：CTA 内复用 A/B
Register：保存 fragment 和 C accumulator
CTA：组织一群线程协作完成一个 tile
```

[1]: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/?utm_source=chatgpt.com "NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog"
[2]: https://resources.nvidia.com/en-us-hopper-architecture/nvidia-tensor-core-gpu-datasheet?utm_source=chatgpt.com "NVIDIA H100 GPU Datasheet"
