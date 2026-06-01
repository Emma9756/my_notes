---
title: "deepgemm_fp8_fp4"
date: 2026-05-28
tags:
  - #LLM
  - #算子
  - #gemm
  - #fp8
  - #fp4
  - #deepseek
  - #from_me
  - #待整理
status: 待整理
---

# `m_grouped_fp8_fp4_gemm_nt_contiguous` CUDA Walkthrough

这份文档面向 CUDA 初学者，目标不是把所有模板细节讲完，而是把这条调用链讲清楚：

1. Python 是怎么调用到它的
2. C++ 入口做了哪些检查和预处理
3. 为什么 scale 会被“变形”
4. JIT runtime 是怎么把配置编译成 CUDA kernel 的
5. kernel 在 GPU 上大致怎么跑
6. 用到了哪些底层硬件接口

如果你只想先抓住主线，可以先看第 1 节和第 8 节。

---

## 1. 一句话理解这条算子

`m_grouped_fp8_fp4_gemm_nt_contiguous` 做的事可以概括成：

- 把很多个 expert 的 token 行拼成一个大矩阵 `A`
- 每个 expert 有自己的一份权重 `B_i`
- 用 `grouped_layout` 告诉 kernel：
  “`A` 的哪一段行属于哪个 expert”
- 对每个 expert 分别做：

```text
D_i = A_i @ B_i^T
```

其中：

- `A_i` 是 FP8 激活
- `B_i` 是 FP4 权重
- `D_i` 是 BF16 输出

它的价值是：把“很多个小 GEMM”合并成“一次 grouped GEMM kernel”。

---

## 2. 入口在哪

从 `vllm` 视角，常见入口是：

- Python 包装：
  [`vllm/utils/deep_gemm.py:314`](</data/dnn/qinzq/repository/vllm/vllm/utils/deep_gemm.py:314>)
- 专家层调用点：
  [`deep_gemm_moe.py:508`](</data/dnn/qinzq/repository/vllm/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py:508>)
  [`deep_gemm_moe.py:528`](</data/dnn/qinzq/repository/vllm/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py:528>)

从 `DeepGEMM` 视角，真正的 C++ API 入口是：

- [`DeepGEMM/csrc/apis/gemm.hpp:144`](</data/dnn/qinzq/repository/DeepGEMM/csrc/apis/gemm.hpp:144>)

也就是：

```python
m_grouped_fp8_fp4_gemm_nt_contiguous(
    (a_data, a_scale),
    (b_data, b_scale),
    d,
    grouped_layout,
    ...
)
```

这里 `A` 和 `B` 都不是单个 tensor，而是 `(data, scale)` 二元组。

---

## 3. 输入长什么样

逻辑 shape 是：

```text
A: [M, K]
B: [G, N, K]
D: [M, N]
```

其中：

- `G` 是 group 数，MoE 里通常就是 local expert 数
- `A` 是把多个 group 的 token 行沿 `M` 维拼起来的结果
- `B[i]` 是第 `i` 个 group 的权重

更具体一点：

- `a_data`: `[M, K]`, `torch.float8_e4m3fn`
- `a_scale`: 逻辑上 `[M, ceil(K / gran_k_a)]`
- `b_data`: 逻辑上 `[G, N, K]`
  - 对 FP4 权重来说，物理存储通常是 `[G, N, K/2]`, `int8`
  - 因为两个 FP4 值打包到一个 byte
- `b_scale`: 逻辑上 `[G, N, ceil(K / gran_k_b)]`
- `d`: `[M, N]`, `torch.bfloat16`
- `grouped_layout`: `torch.int32`

### `grouped_layout` 有两种形式

`use_psum_layout=False` 时：

- `grouped_layout.shape == [M]`
- 每一行存一个 group id
- padding 行写 `-1`

`use_psum_layout=True` 时：

- `grouped_layout.shape == [G]`
- 每个元素表示该 group 的累计结束位置

可以粗略理解成：

```text
group 0 ends at grouped_layout[0]
group 1 ends at grouped_layout[1]
...
```

---

## 4. 从 Python 进 C++ 后，先做什么

在 [`gemm.hpp:144`](</data/dnn/qinzq/repository/DeepGEMM/csrc/apis/gemm.hpp:144>) 里，这个函数先做三件事。

### 4.1 检查 shape 和 dtype

它会确认：

- `A` 的 shape 是 `[M, K]`
- `B` 的逻辑 shape 是 `[G, N, K]`
- `D` 的 shape 是 `[M, N]`
- `D.dtype == bf16`
- `grouped_layout.dtype == int32`

并且会检查 memory layout 是否符合 kernel 要求。

### 4.2 判断当前 GPU 架构

它会读取：

```cpp
const auto arch_major = device_runtime->get_arch_major();
```

然后决定后面走哪条实现路径：

- `arch_major == 9`：SM90 路径
- `arch_major == 10`：SM100 路径

虽然函数名是 `fp8_fp4`，但在 SM90 上它会退化到 FP8 grouped kernel：

- SM90：`sm90_m_grouped_fp8_gemm_contiguous_1d2d(...)`
- SM100：`sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d(...)`

### 4.3 把 scale 变成 kernel 想要的样子

这一行很关键：

[`gemm.hpp:184`](</data/dnn/qinzq/repository/DeepGEMM/csrc/apis/gemm.hpp:184>)

```cpp
transform_sf_pair_into_required_layout(...)
```

它的作用不是“重新量化数据”，而是：

- 检查 scale 的量化粒度是否符合要求
- 把 scale 转成 kernel 更容易读取的 layout
- 必要时把 `float` scale 压成 `int` 表示

相关实现：

- [`layout.hpp:62`](</data/dnn/qinzq/repository/DeepGEMM/csrc/apis/layout.hpp:62>)

---

## 5. 为什么 scale 还要再转换一次

这是很多初学者最容易卡住的地方。

Python 侧量化函数通常产出的是“逻辑上正确”的 scale，例如：

- `A` 的 scale：每行每 `128` 列一个 scale
- `B` 的 scale：每行每 `32` 列一个 scale

但 kernel 真正关心的是：

- scale 在内存里是不是 **MN-major**
- shape 是否 **TMA 对齐**
- scale 是不是压成了更适合硬件加载的格式

`transform_sf_into_required_layout()` 就是在做这件事：

- SM90 下常见结果是 `float`
- SM100 下常见结果是打包后的 `int`

见：

- [`layout.hpp:14`](</data/dnn/qinzq/repository/DeepGEMM/csrc/apis/layout.hpp:14>)

文档化地说：

```text
原始 scale
    -> 检查量化粒度 recipe
    -> 转成 MN-major
    -> 做 TMA 对齐
    -> 必要时压成 packed UE8M0
    -> 交给 kernel
```

---

## 6. runtime 层在做什么

当 API 决定走 SM100 路径后，会进入：

- [`sm100_fp8_fp4_gemm_1d1d.hpp:158`](</data/dnn/qinzq/repository/DeepGEMM/csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:158>)

这层代码的主要职责不是“算 GEMM”，而是：

1. 根据 `m/n/k/group 数/数据类型` 选一个 kernel 配置
2. 为输入输出构造 TMA descriptor
3. 把这些配置塞进 `Args`
4. 生成 CUDA 源码
5. 交给 JIT 编译器编译
6. 最后 launch kernel

### 6.1 `GemmDesc` 和 `GemmConfig`

runtime 先构造 `GemmDesc`：

- 当前是什么 GEMM 类型
- `m/n/k` 是多少
- group 数是多少
- `A/B/D` 的 dtype 是什么
- major layout 是什么

然后通过 heuristic 选择 `GemmConfig`：

```cpp
const auto config = get_best_config<SM100ArchSpec>(desc);
```

你可以把它理解成：

- 自动选择 block size
- 自动选择 pipeline stage 数
- 自动选择 swizzle / cluster / 线程数

### 6.2 `Args`

SM100 runtime 最终把这些东西放进一个 `Args` 结构：

- `gemm_desc`
- `gemm_config`
- `gran_k_a`
- `gran_k_b`
- `grouped_layout`
- `tensor_map_a`
- `tensor_map_b`
- `tensor_map_sfa`
- `tensor_map_sfb`
- `tensor_map_cd`

见：

- [`sm100_fp8_fp4_gemm_1d1d.hpp:17`](</data/dnn/qinzq/repository/DeepGEMM/csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:17>)

---

## 7. 什么是 TMA descriptor

这是第一次接触 CUDA 硬件接口时最容易陌生的概念。

TMA 是 **Tensor Memory Accelerator**。  
它可以帮助 GPU 把“global memory 上的一块二维/三维 tensor”高效搬到 shared memory。

但 kernel 不能只拿一个裸指针就让 TMA 工作，它还需要一个描述对象：

```text
这个 tensor 的 dtype 是什么
全局内存里的 stride 是多少
tile 的大小是多少
shared memory 里希望怎么摆
要不要 swizzle
```

这个描述对象就是 `CUtensorMap`。

构造它的底层 CUDA Driver API 是：

```cpp
cuTensorMapEncodeTiled(...)
```

在代码里封装成了：

- `make_tma_a_desc(...)`
- `make_tma_b_desc(...)`
- `make_tma_sf_desc(...)`
- `make_tma_cd_desc(...)`

最终内部会调用：

- [`runtime_utils.hpp:109`](</data/dnn/qinzq/repository/DeepGEMM/csrc/jit_kernels/impls/runtime_utils.hpp:109>)

这是本函数和“普通 PyTorch matmul”最大的不同之一：

- 普通 matmul：多数细节由 cuBLAS 处理
- 这里：库自己构造 TMA descriptor，自己发起 tile copy

---

## 8. JIT 编译到底做了什么

DeepGEMM 不是把所有 kernel 预先编译死，而是会根据当前参数生成特化代码。

在 runtime 里你会看到：

```cpp
const auto code = SM100FP8FP4Gemm1D1DRuntime::generate(args);
const auto runtime = compiler->build("sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d", code);
```

见：

- [`sm100_fp8_fp4_gemm_1d1d.hpp:232`](</data/dnn/qinzq/repository/DeepGEMM/csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:232>)

`compiler->build(...)` 会：

1. 把生成出来的 `.cu` 代码写到缓存目录
2. 调用 NVCC 编译成 cubin
3. 把结果缓存到 `~/.deep_gemm`
4. 下次遇到相同签名时直接复用

相关实现：

- [`compiler.hpp:88`](</data/dnn/qinzq/repository/DeepGEMM/csrc/jit/compiler.hpp:88>)

所以你可以把它理解成：

```text
不是“调用一个固定 kernel”
而是“先按当前形状和配置生成一个专用 kernel，再运行它”
```

---

## 9. 真正的 CUDA kernel 在哪里

SM100 的设备实现入口在：

- [`sm100_fp8_fp4_gemm_1d1d.cuh:29`](</data/dnn/qinzq/repository/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh:29>)

函数原型大致是：

```cpp
sm100_fp8_fp4_gemm_1d1d_impl(
    grouped_layout,
    shape_m, shape_n, shape_k,
    tensor_map_a,
    tensor_map_b,
    tensor_map_sfa,
    tensor_map_sfb,
    tensor_map_cd)
```

注意这里已经看不到 Python 层的 tensor 了。  
kernel 真正收到的是：

- 一些整数参数
- `grouped_layout` 指针
- 一堆 TMA descriptor

这说明 host 侧已经把“怎么访问这些 tensor”的信息打包好了。

---

## 10. kernel 在 GPU 上大概怎么跑

把模板细节都忽略掉，它的主流程可以理解成下面几步。

### 10.1 用 `Scheduler` 决定当前 CTA 算哪一块

相关代码：

- [`scheduler/gemm.cuh:28`](</data/dnn/qinzq/repository/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/gemm.cuh:28>)

`Scheduler` 的职责是：

- 把大问题 `[M, N, K]` 切成 block
- 在 grouped 布局下，把每个 block 映射到正确的 group
- 根据 `grouped_layout` 算出：
  - 当前 block 对应的 `m_block_idx`
  - 当前 block 对应的 `n_block_idx`
  - 当前该读取哪个 group 的权重

你可以把它理解成“任务分发员”。

### 10.2 用 TMA 从 global memory 搬 tile 到 shared memory

SM100 kernel 里，`warp 0` 主要负责发 TMA load：

- 把 `A` 的一个 tile 搬进 shared memory
- 把 `B` 的一个 tile 搬进 shared memory
- 在某些 stage 同时搬 `SFA/SFB`

相关代码：

- [`sm100_fp8_fp4_gemm_1d1d.cuh:203`](</data/dnn/qinzq/repository/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh:203>)

### 10.3 用 barrier 保证“数据到了再算”

在 CUDA 里，load 和 compute 是并行流水的，所以必须有同步原语。

这里主要用的是：

- `ClusterTransactionBarrier`
- `cluster_sync()`
- 各种 full/empty barrier

初学者可以先把它理解成：

- producer：负责搬数据
- consumer：负责计算
- barrier：保证 consumer 不会读到“还没搬完”的 tile

### 10.4 用 UMMA 做矩阵乘

SM100 的核心计算不是普通 CUDA for-loop，而是 **UMMA** 指令路径。

你可以把 UMMA 粗略理解成：

- Blackwell 上更底层、更贴近硬件的 Tensor Core 矩阵乘接口

SM100 kernel 里专门有一个 warp 负责 issue UMMA：

- [`sm100_fp8_fp4_gemm_1d1d.cuh:271`](</data/dnn/qinzq/repository/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh:271>)

这一步做的事情可以概括成：

```text
从 shared memory 读入 A/B tile
结合 SFA/SFB 做 block-scaled dequant
在 Tensor Core 上执行 tile GEMM
把累加结果写到 tensor memory / 中间缓冲
```

### 10.5 epilogue 把结果写回 `D`

最后要把累加结果存回输出 `D:[M, N]`。

这一步通常会处理：

- dtype 转换
- tile store
- 边界 block 的裁剪

对这条路径来说，最终输出是 BF16。

---

## 11. “底层器件接口”都有哪些

如果你想知道“这条链路到底碰了哪些 CUDA 硬件接口”，可以重点看这几个概念。

### 11.1 Global Memory

就是 GPU 大显存。  
`A/B/scale/D` 的原始数据都先在这里。

### 11.2 Shared Memory

每个 SM 上的片上 scratchpad。  
kernel 先把 global memory 的 tile 搬进 shared memory，再喂给 Tensor Core。

### 11.3 TMA

Tensor Memory Accelerator。  
负责高效 tile 搬运。使用前需要 `CUtensorMap` descriptor。

### 11.4 `CUtensorMap` / `cuTensorMapEncodeTiled`

这是 CUDA Driver API 层的对象和编码接口。  
DeepGEMM 用它描述 A/B/scale/D 的 tile 访问方式。

### 11.5 Barrier

用于 producer/consumer 同步。  
典型对象是 `ClusterTransactionBarrier`。

### 11.6 UMMA

Blackwell 上的矩阵乘硬件接口。  
这是 SM100 路径的核心计算接口。

### 11.7 TMEM

Tensor Memory。  
SM100 kernel 会用它存一些累加/scale 相关中间状态，再配合 epilogue 写回。

---

## 12. 如果当前机器不是 SM100，会发生什么

这个函数虽然叫 `fp8_fp4`，但 host 入口是“按架构分发”的：

- SM90：走 `sm90_m_grouped_fp8_gemm_contiguous_1d2d`
- SM100：走 `sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d`

为什么会这样：

- SM90 没有 SM100 那套 FP8/FP4 block-scaled UMMA 路径
- 所以只能退到旧一代 grouped FP8 实现

这也是为什么你在 C++ 入口会看到：

```cpp
if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) { ... }
else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) { ... }
```

含义是：

- SM90 偏好 float scale layout
- SM100 偏好 packed int scale layout

---

## 13. 一条完整调用链总结

把前面的内容压缩成一条链，就是：

```text
vllm Python 调用
    -> deep_gemm Python wrapper
    -> DeepGEMM C++ API
    -> 检查 shape / dtype / layout
    -> transform_sf_pair_into_required_layout
    -> 选择 SM90 或 SM100 runtime
    -> 构造 GemmDesc / GemmConfig
    -> make_tma_*_desc 生成 CUtensorMap
    -> generate(args) 生成特化 CUDA 源码
    -> compiler->build(...) JIT 编译
    -> launch kernel
    -> Scheduler 决定 block 属于哪个 group
    -> TMA 搬 A/B/scale tile 到 shared memory
    -> barrier 同步
    -> UMMA/WGMMA 做 tile GEMM
    -> epilogue 写回 BF16 输出 D
```

---

## 14. 给 CUDA 小白的阅读建议

如果你打算顺着源码继续看，建议按这个顺序：

1. 先看 [`gemm.hpp:144`](</data/dnn/qinzq/repository/DeepGEMM/csrc/apis/gemm.hpp:144>)
   只理解“输入检查 + 分发”
2. 再看 [`layout.hpp:62`](</data/dnn/qinzq/repository/DeepGEMM/csrc/apis/layout.hpp:62>)
   只理解“为什么 scale 要变形”
3. 再看 [`sm100_fp8_fp4_gemm_1d1d.hpp:158`](</data/dnn/qinzq/repository/DeepGEMM/csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:158>)
   只理解“runtime 怎么组织参数”
4. 再看 [`runtime_utils.hpp`](</data/dnn/qinzq/repository/DeepGEMM/csrc/jit_kernels/impls/runtime_utils.hpp:1>)
   只理解“CUtensorMap/TMA descriptor 是什么”
5. 最后看 [`sm100_fp8_fp4_gemm_1d1d.cuh:29`](</data/dnn/qinzq/repository/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh:29>)
   只抓主流程，不要一开始陷进模板细节

如果你在第 5 步卡住，优先只找这些关键词：

- `Scheduler`
- `tma::copy`
- `Barrier`
- `UMMA`
- `store`

先理解“角色分工”，再看模板参数。
