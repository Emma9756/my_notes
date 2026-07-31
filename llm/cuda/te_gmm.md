# Transformer Engine PyTorch `te_general_grouped_gemm` 接口解析

本文分析：

```text
transformer_engine/pytorch/csrc/extensions/gemm.cpp:444
```

对应函数：

```cpp
std::optional<std::vector<at::Tensor>> te_general_grouped_gemm(
    std::vector<py::handle> A,
    bool transa,
    std::vector<py::handle> B,
    bool transb,
    std::optional<std::vector<at::Tensor>> D,
    DType D_type,
    std::vector<int64_t> m_splits,
    std::vector<at::Tensor> bias,
    DType bias_type,
    bool single_output,
    std::vector<at::Tensor> pre_gelu_out,
    bool grad,
    std::vector<at::Tensor> workspace,
    size_t workspaceSize,
    bool accumulate,
    bool use_split_accumulator,
    int math_sm_count);
```

它是 Transformer Engine（下文简称 TE）PyTorch 扩展中的“多矩阵 GEMM”入口：一次接收多组 A/B/D/bias，统一完成 Tensor 包装、输出组织、低精度 scale 布局处理，再把有效的各组 GEMM 提交给 `nvte_multi_tensor_gemm`。

这里的 grouped 不是传统“同形状 batch GEMM”的同义词。每组矩阵可以有不同的 M/N/K；默认后端把不同组分配到多个 CUDA compute stream 上，每组仍调用一次 cuBLASLt GEMM。满足特定条件并显式开启环境变量时，Hopper 上也可以走 CUTLASS grouped GEMM 快速路径。

## 1. 调用链

正常调用链：

```text
Python:
transformer_engine/pytorch/cpp_extensions/gemm.py
::general_grouped_gemm(...)
    |
    | 准备 bias/dbias、pre-GELU、多个 workspace、transpose 标志
    v
PyBind:
transformer_engine/pytorch/csrc/extensions/pybind.cpp
::te_general_grouped_gemm
    |
    v
C++ PyTorch wrapper:
transformer_engine/pytorch/csrc/extensions/gemm.cpp
::te_general_grouped_gemm(...)
    |
    | 包装多组 Tensor、组织单输出/多输出、处理 scale
    v
TE C API:
transformer_engine/common/gemm/cublaslt_gemm.cu
::nvte_multi_tensor_gemm(...)
    |
    +--> 默认/回退：multi_stream_cublas_gemm(...)
    |       每组调用 nvte_cublas_gemm_v2(...)
    |
    +--> 可选：Hopper CUTLASS grouped GEMM
```

PyBind 这里只注册函数，没有逐参数命名：

```cpp
m.def(
    "te_general_grouped_gemm",
    &transformer_engine::pytorch::te_general_grouped_gemm,
    "Grouped GEMM");
```

所以直接调用扩展时是位置参数接口。正常应通过 Python 的 `general_grouped_gemm(...)` 调用，以获得列表构造、workspace 申请和 dtype 转换等上层准备。

## 2. 功能概览

设共有 `Z` 组 GEMM：

```text
A = [A0, A1, ..., A(Z-1)]
B = [B0, B1, ..., B(Z-1)]
D = [D0, D1, ..., D(Z-1)]
```

它计算：

```text
Di = GEMM(Ai, Bi), i = 0 ... Z-1
```

并可对每一组融合：

- bias；
- GELU；
- dGELU；
- bias gradient；
- 累加到已有 D；
- FP8 等低精度 GEMM。

接口还支持两种输出组织：

```text
single_output = false:
    D = [D0, D1, ..., D(Z-1)]

single_output = true:
    D = [D_all]
    D_all 的连续存储依次包含 D0, D1, ..., D(Z-1)
```

第二种适合 MoE/grouped linear 一类场景：输入按 token group 切分进行多次 GEMM，但最终输出仍希望是一个连续 Tensor。

## 3. 数学与布局语义

### 3.1 每组的数学式

默认 cuBLAS 多流路径对每组最终调用：

```cpp
const float alpha = 1.f;
const float beta = accumulate ? 1.f : 0.f;

nvte_cublas_gemm_v2(
    transa,
    transb,
    &alpha,
    A[i],
    B[i],
    &beta,
    D[i],  // C
    D[i],  // D
    ...);
```

因此每组的底层形式是：

```text
D_i,new = op(A_i) * op(B_i) + beta * D_i,old
```

但从 PyTorch Tensor 的行主序观察方式来看，与单 GEMM 一样，更实用的理解是：

```text
D_i = op_py(B_i) @ op_py(A_i) + beta * D_i,old
```

其中：

```text
op_py(A) = A.T if transa else A
op_py(B) = B.T if transb else B
```

Python 上层根据 `layout` 设置：

```python
transa = layout[0] == "T"
transb = layout[1] == "T"
```

常用布局：

| layout | `transa` | `transb` | PyTorch 张量视角 |
|---|---:|---:|---|
| `TN` | true | false | `B_i @ A_i.T` |
| `NN` | false | false | `B_i @ A_i` |
| `NT` | false | true | `B_i.T @ A_i` |

与单 GEMM 的 Python `general_gemm` 不同，`general_grouped_gemm` 本身没有显式 `assert layout in (...)`，但仓库的测试和调用点使用的仍是上述三种布局；调用者应避免未经底层验证的 layout 字符串。

### 3.2 三种典型训练计算

假设第 i 个专家/组：

```text
weight_i: [N, K]
input_i:  [M_i, K]
dY_i:     [M_i, N]
```

#### 前向：TN

```text
D_i[M_i, N] = input_i[M_i, K] @ weight_i[N, K].T
```

输入：

```text
A_i = weight_i [N, K]
B_i = input_i  [M_i, K]
```

#### 输入梯度：NN

```text
dX_i[M_i, K] = dY_i[M_i, N] @ weight_i[N, K]
```

输入：

```text
A_i = weight_i [N, K]
B_i = dY_i     [M_i, N]
```

#### 权重梯度：NT

```text
dW_i[N, K] = dY_i[M_i, N].T @ input_i[M_i, K]
```

输入：

```text
A_i = input_i [M_i, K]
B_i = dY_i    [M_i, N]
```

仓库 `test_grouped_gemm` 正是按这三种关系构造测试，并逐组调用单 GEMM 作为参考结果。

### 3.3 每组输出形状

函数复用单 GEMM 的：

```cpp
getGemmOutputShape(te_A.shape(), transa, te_B.shape(), transb)
```

若将每个输入的前导维压平：

```text
A_i -> [A0_i, A1_i]
B_i -> [B0_i, B1_i]
```

则内维必须满足：

```text
(transa ? A1_i : A0_i) == (transb ? B0_i : B1_i)
```

输出形状：

- `transb=false` 时保留 `B_i.shape[:-1]`；
- `transb=true` 时输出首维为 `B1_i`；
- 最后一维在 `transa=true` 时为 `A0_i`，否则为 `A1_i`。

默认 TN 的常见形式：

```text
A_i: [N_i, K_i]
B_i: [M_i, K_i]
D_i: [M_i, N_i]
```

## 4. 与单 GEMM、batched GEMM 和传统公式的异同

### 4.1 与传统 `d = a @ b + c`

相同点：

- 核心都是矩阵乘；
- `accumulate=true` 时支持把结果加到旧输出。

不同点：

1. 一次计算多组 `(A_i, B_i)`；
2. 不提供任意独立矩阵 C，C 与 D 使用同一存储；
3. 没有公开 `alpha`/`beta` 参数：
   - `alpha` 固定为 1；
   - `beta` 只能由 `accumulate` 选择 0 或 1；
4. `bias_i` 是广播 bias，不是完整矩阵 C；
5. 可融合 GELU/dGELU/dbias；
6. 支持 TE 量化 Tensor 和 scale swizzle；
7. 可跨多个 CUDA stream 并发提交不同组。

每组普通路径的核心式：

```text
accumulate=false:
    D_i = matmul_i

accumulate=true:
    D_i = matmul_i + D_i_old
```

加 bias/GELU 后，前向近似为：

```text
pre_gelu_i = matmul_i + D_i_old(if accumulate) + bias_i
D_i = GELU(pre_gelu_i)   # 若请求 GELU
```

### 4.2 与单 GEMM `gemm.cpp:127`

共同点：

- 使用相同的输出形状推导；
- 都支持 A/B transpose；
- 都支持 bias、GELU/dGELU/dbias、accumulate、split accumulator；
- 都处理低精度 scale swizzle；
- 最终都可落到 `nvte_cublas_gemm_v2`。

主要差异：

| 项目 | 单 GEMM `gemm` | `te_general_grouped_gemm` |
|---|---|---|
| 输入 | 一组 A/B | A/B 列表 |
| 输出量化器 | 有 `quantizer` | C++ 接口无输出 quantizer |
| 输出分配 | 可由 quantizer 创建 | 按 `D_type` 创建普通 ATen Tensor |
| alpha/beta | 可指定 alpha，beta 可选 | alpha=1；beta 仅为 0/1 |
| 通信重叠 | 支持 UB overlap | 不支持该组参数 |
| 输出组织 | 单 Tensor | 多 Tensor 或一个连续 Tensor |
| 执行 | 当前 stream 单 GEMM | 多 compute stream / 可选 CUTLASS grouped |

### 4.3 与传统 strided batched GEMM

传统 strided batched GEMM 通常要求：

- 每个 batch 的 M/N/K 相同；
- Tensor 通过固定 stride 排列；
- 所有 batch 使用同一 GEMM 配置。

该接口则允许每组拥有独立指针和不同形状：

```text
组0: [M0, K0] @ [K0, N0]
组1: [M1, K1] @ [K1, N1]
...
```

因此它更接近 pointer-array grouped GEMM / multi-tensor GEMM，而不是固定尺寸的 strided batch。

但所有组共用：

- `transa`；
- `transb`；
- `D_type`；
- `bias_type`；
- `grad`；
- `accumulate`；
- `use_split_accumulator`；
- `math_sm_count`。

## 5. 每个参数的含义、作用和要求

函数共有 17 个参数。

### 5.1 `A`

```cpp
std::vector<py::handle> A
```

含义：

- A 操作数列表；
- `A[i]` 与 `B[i]` 组成第 i 个 GEMM；
- 元素可以是普通 PyTorch Tensor 或 TE 支持的量化 Tensor。

作用：

- 每个元素由 `makeTransformerEngineTensor(A[i], none)` 包装；
- wrapper 保存数据、shape、dtype、scale/scale inverse 和 scaling mode；
- `A.size()` 被当作总 GEMM 数量。

要求：

- 列表必须非空，否则 `workspace[0]`、上层的 `A[0]` 等都会越界；
- `B`、bias、pre-GELU 等按组列表至少要能索引到 `A.size()-1`；
- 每个 A 至少应是一维且能按 GEMM 规则压平成二维；
- 每组 A/B 必须满足内维匹配；
- 所有 Tensor 应位于同一 CUDA device；
- 各组可以有不同形状，但 transpose 模式相同。

代码没有集中验证所有列表长度，因此这些是调用者必须保证的前置条件。

### 5.2 `transa`

```cpp
bool transa
```

含义：

- 所有组共同的 A transpose 标志。

作用：

- 参与每组输出形状推导；
- 传给底层多 GEMM；
- 决定 A 的量化 scale swizzle 方向；
- Blackwell FP8 block-scaling 转换后，内部可能统一改为 true。

要求：

- 所有组必须能使用同一个 transa；
- Python 上层通常由 layout 第一个字符产生。

### 5.3 `B`

```cpp
std::vector<py::handle> B
```

含义：

- B 操作数列表；
- 第 i 组使用 `A[i]` 和 `B[i]`。

要求：

- 实际长度必须等于 A 长度；
- 每个元素必须与对应 A 满足矩阵内维关系；
- device、量化元数据和生命周期必须有效；
- 可有不同的 M，但所有组共用 transb。

函数体没有显式检查 `B.size() == A.size()`，长度不足会产生越界访问。

### 5.4 `transb`

```cpp
bool transb
```

含义：

- 所有组共同的 B transpose 标志。

作用和要求：

- 用于每组形状推导、scale swizzle 和底层 GEMM；
- Python 上层通常由 layout 第二个字符产生；
- Blackwell FP8 block-scaling 兼容路径可能在内部改为 false。

### 5.5 `D`

```cpp
std::optional<std::vector<at::Tensor>> D
```

含义：

- 可选输出 Tensor 列表；
- 也是 `accumulate=true` 时旧输出的来源。

`single_output=false`：

- D 为空：C++ 为每个有效组创建一个输出，并放入 `D_vectors` 返回；
- D 非空：直接使用 `(*D)[i]`；
- 正常 Python 包装总会传入 `out`，所以常见路径是复用调用者输出。

`single_output=true`：

- D 必须非空；
- `D[0]` 是总输出；
- C++ 在其连续存储中依次创建每组的非 owning view。

要求：

- `single_output=true` 时必须至少有 `D[0]`；
- `single_output=false && D有值` 时长度必须至少等于 A 长度；
- 每个 D 的 dtype 应与 `D_type` 一致；
- 每组 D 形状必须与推导结果一致；
- 单输出的存储必须 contiguous，容量必须覆盖所有组输出；
- accumulate 模式下 D 必须预先含有有效旧值。

值得注意：与单 GEMM wrapper 不同，此函数对调用者提供的 D 没有显式执行 shape/dtype/contiguous 检查，错误通常会更晚在底层暴露，或导致错误的 view/内存解释。因此上层必须严格保证这些条件。

### 5.6 `D_type`

```cpp
DType D_type
```

含义：

- 新建输出 Tensor 时使用的 TE dtype；
- 被转换为 ATen dtype：

```cpp
auto dtype = GetATenDType(D_type);
```

Python 上层的实际来源：

```python
out_dtype = TE_DType[out[0].dtype] if D_dtype is None else D_dtype
```

注意 Python 函数虽然还有名为 `out_dtype` 的必选参数，但当前代码随后用 `out[0].dtype` 或 `D_dtype` 覆盖了局部变量。因此传入 C++ 的 D_type 实际以已有输出 dtype 为默认依据。

要求：

- 必须是 TE 可映射到 ATen 的 dtype；
- 应与调用者提供的 D 实际 dtype 一致；
- 当前 C++ 接口没有输出 quantizer，不能仅靠 D_type 创建 TE 量化对象。

### 5.7 `m_splits`

```cpp
std::vector<int64_t> m_splits
```

含义：

- 从 Python API 看，它描述每组沿 M/token 维的大小；
- 典型要求：

```text
len(m_splits) == num_gemms
sum(m_splits) == 总输入/输出的 M
```

典型例子：

```text
m_splits = [10, 3, 0, 7]
```

表示四组分别处理 10、3、0、7 个 token。

非常重要的实现事实：

- 在 `te_general_grouped_gemm` 的 C++ 函数体中，`m_splits` 完全没有被读取；
- 单输出 C++ view 的偏移量是依据每组推导出的 `D_shape[0] * D_shape[1]` 计算，而不是依据 `m_splits`；
- DebugQuantizer 的 Python 回退路径才显式使用 `m_splits` 从总输出切片。

因此，在正常 C++ 路径中，真正决定每组大小的是 `A[i]`、`B[i]` 的 shape；`m_splits` 是上层分组契约的一部分，但不是此函数内的计算依据。

要求：

- 应与输入实际切分和输出布局一致；
- `single_output=true` 时每项通常应等于对应输出的 M；
- 即使 C++ 当前未使用，也不能随意传错误值，因为 DebugQuantizer 回退路径及更高层 grouped linear 逻辑会使用它。

### 5.8 `bias`

```cpp
std::vector<at::Tensor> bias
```

含义：

- 每组的 bias 或 bias gradient 输出；
- 未启用 bias 时，上层传入一组 empty Tensor 占位。

前向 `grad=false`：

- `bias[i]` 被配置为第 i 组的广播 bias。

反向 `grad=true`：

- 上层创建 `grad_bias[i]`；
- 底层把 `bias[i]` 作为 dbias 输出 Tensor。

空输入组：

```cpp
if (bias[i].numel() != 0 && grad) {
    bias[i].zero_();
}
```

要求：

- 长度必须至少等于 A 长度；
- 启用前向 bias 时，每项通常是一维、长度等于对应输出最后一维；
- grad 模式的 dbias 存储必须可写；
- dtype 应与 `bias_type` 一致；
- 不启用时也要传可安全索引的 empty Tensor 列表，不能简单传空 vector。

### 5.9 `bias_type`

```cpp
DType bias_type
```

含义：

- 所有组共同的 bias/dbias 和 pre-GELU dtype。

Python 上层：

```python
if use_bias:
    bias_dtype = (
        TE_DType[grad_bias[0].dtype]
        if grad
        else TE_DType[bias[0].dtype]
    )
else:
    bias_dtype = TE_DType[torch.bfloat16]
```

C++ 还固定：

```cpp
DType gelu_type = bias_type;
```

要求：

- 所有组 bias/dbias 应能共享该 dtype；
- pre-GELU Tensor 也应与它一致；
- 没有 bias/GELU 时上层仍传默认 BF16 枚举，但 empty Tensor 不产生实际计算。

### 5.10 `single_output`

```cpp
bool single_output
```

含义：

- 是否让多组 GEMM 写入一个连续总输出。

`false`：

```text
D[0] -> group 0 output
D[1] -> group 1 output
...
```

`true`：

```text
D[0] storage:
+----------+----------+-----+
| group 0  | group 1  | ... |
+----------+----------+-----+
```

实现：

```cpp
output_data_ptr = (*D)[0].data_ptr();

out_tensor = at::from_blob(output_data_ptr, D_shape, opts);

output_data_ptr +=
    D_shape[0] * D_shape[1] * (*D)[0].element_size();
```

要求：

- D 不能是 `nullopt`，否则立即报错；
- D 至少包含 `D[0]`；
- `D[0]` 必须是连续、容量足够的二维兼容存储；
- 每组输出在存储中必须按组序紧密排列；
- 当前偏移计算只使用 `D_shape[0] * D_shape[1]`，因此实际上假设每组输出是二维；
- `D_type` 必须匹配 `D[0]` 的元素大小和解释方式。

空组处理：

- 如果某组输出任一维为 0，不对该组调用 `at::from_blob`，避免指针已经位于分配末端时触发 PyTorch blob 边界错误；
- 仍按零元素大小推进指针，并把空 Tensor 放入内部列表。

### 5.11 `pre_gelu_out`

```cpp
std::vector<at::Tensor> pre_gelu_out
```

含义：

- 每组 GELU 前的辅助 Tensor；
- 前向保存 `matmul + bias`；
- 反向提供 dGELU 输入；
- 不启用 GELU 时，上层传 empty Tensor 占位。

上层 `gelu=true` 时：

```python
gelu_input = [
    torch.empty_like(
        o,
        dtype=bias_dtype,
        memory_format=torch.contiguous_format,
    )
    for o in out
]
```

要求：

- 列表长度必须至少为 A.size()；
- 有效项的形状应与对应 D 一致；
- dtype 应为 `bias_type`；
- 不启用时也需传可索引的 empty Tensor 列表；
- `single_output=true` 时，上层注释明确指出当前 `gelu_input` 处理“应当有所不同”：它按 `out` 列表创建，而 `out` 往往只有一个总 Tensor。这说明 single-output + GELU 组合在这层接口中并未被完整组织，使用前应核查具体调用路径，不能假定已支持每组 auxiliary buffer。

空输入组会将非空 pre-GELU Tensor 清零。

### 5.12 `grad`

```cpp
bool grad
```

含义：

- 在底层为所有组统一选择前向 epilogue 或梯度 epilogue。

底层多流配置：

```cpp
if (grad) {
    config.dbias_tensor = bias[i];
    config.with_dgelu_epilogue = has_pre_gelu;
} else {
    config.bias_tensor = bias[i];
    config.with_gelu_epilogue = has_pre_gelu;
}
```

要求：

- 它不自动推导 wgrad/dgrad 的矩阵布局；
- 调用者仍需正确选择 A、B、layout；
- 所有组共用同一个 grad 值。

### 5.13 `workspace`

```cpp
std::vector<at::Tensor> workspace
```

含义：

- 多 compute stream 的 cuBLASLt workspace 列表。

Python 上层由：

```python
get_cublas_workspace(device, ub=False, grouped_gemm=True)
```

创建，数量等于：

```python
tex.get_num_cublas_streams()
```

C++ 为每项创建 byte TensorWrapper。默认多流路径按：

```cpp
workspace[i % num_streams]
```

复用。

要求：

- vector 必须非空，因为函数入口使用 `workspace[0].device()`；
- 对默认多流后端，数量应至少覆盖 `nvte_get_num_compute_streams()` 所需索引；
- 每项必须是 CUDA Tensor；
- 所有项与 A/B/D 位于同一设备；
- 每项实际字节容量至少为 workspaceSize；
- workspace 生命周期必须覆盖异步多流执行。

### 5.14 `workspaceSize`

```cpp
size_t workspaceSize
```

含义：

- 每一个 workspace Tensor 可用的字节数。

包装方式：

```cpp
makeTransformerEngineTensor(
    workspace[i].data_ptr(),
    {workspaceSize},
    DType::kByte);
```

Python 上层传：

```python
workspaces[0].shape[0]
```

由于 workspace 是一维 uint8 Tensor，元素数等于字节数。

要求：

- 单位为 bytes；
- 所有 workspace 被统一解释为相同大小；
- 不得超过任何一项实际分配的容量。

### 5.15 `accumulate`

```cpp
bool accumulate
```

含义：

- 是否把每组 GEMM 结果累加到该组已有 D。

底层固定：

```text
accumulate=false -> beta=0
accumulate=true  -> beta=1
```

要求：

- 为 true 时 D 必须已经分配并包含调用者希望保留的旧值；
- 单输出模式下每个内部 view 都从总输出对应片段读取旧值；
- 空输入组且 `accumulate=false` 时输出清零；
- 空输入组且 `accumulate=true` 时保留旧输出。

与单 GEMM 不同，这里不能指定 `alpha=2` 或 `beta=0.5` 一类任意缩放。

### 5.16 `use_split_accumulator`

```cpp
bool use_split_accumulator
```

含义：

- 是否对低精度 GEMM 使用 split accumulator 策略。

作用：

- 所有组共用；
- 传入每一组的 `MatmulConfig`；
- 影响 FP8 GEMM 的性能/数值策略；
- CUTLASS FP16/BF16 grouped 快速路径会忽略该参数。

要求：

- 应由量化 recipe、硬件架构和精度要求决定；
- 它不是把一组 GEMM 切成多个输出。

### 5.17 `math_sm_count`

```cpp
int math_sm_count
```

含义：

- GEMM 数学 kernel 可使用的 SM 数量提示/限制；
- 0 表示交给 cuBLAS heuristics。

Python 上层计算：

```python
sm_count - int(os.getenv("NVTE_EXT_MARGIN_SM", str(sm_count)))
```

也就是环境变量 `NVTE_EXT_MARGIN_SM` 表示希望为其他工作预留的 SM 数；默认预留值为全部 SM，所得 `math_sm_count=0`，由底层启发式决定。

要求：

- 应在 `[0, device_sm_count]` 的合理范围内；
- 所有组共用；
- 多 stream 并不意味着每个 stream 都独占这么多个 SM，这只是传给各 GEMM/后端的配置。

## 6. 参数列表之间的长度契约

令：

```text
Z = A.size()
```

调用者应保证：

| 参数 | 需要的长度 |
|---|---:|
| A | Z，且 Z > 0 |
| B | Z |
| bias | Z，包括不用 bias 时的 empty 占位 |
| pre_gelu_out | Z，包括不用 GELU 时的 empty 占位 |
| m_splits | 通常为 Z |
| D，single_output=false | Z |
| D，single_output=true | 至少 1，总输出为 D[0] |
| workspace | 通常为 compute stream 数，且至少非空 |

C++ wrapper 没有统一检查这些长度。它以 `A.size()` 为循环上界，对 B、bias、pre-GELU 直接使用 `[i]`；错误长度可能导致越界，而不一定得到友好的异常信息。

Python 上层也没有完整执行所有长度、shape、device 检查，因此这是实质性的调用契约，不只是建议。

## 7. 输出组织的实现

### 7.1 多输出模式

`single_output=false`：

```cpp
if (D == std::nullopt) {
    out_tensor = at::empty(D_shape, opts);
    D_vectors.emplace_back(out_tensor);
} else {
    out_tensor = (*D)[i];
}
```

若 C++ 自行分配输出，函数返回这些 Tensor：

```cpp
return D_vectors;
```

但函数最后实际写的是：

```cpp
return bias;
```

这里需要结合真实签名理解：源码中函数返回 `std::optional<std::vector<at::Tensor>>`，末尾返回的是 `bias`，而不是 `D_vectors`。因此：

- 调用者传入的 D 是原地输出；
- Python wrapper 不使用 C++ 返回值来获得 D；
- 返回值被 Python 当作 bias/dbias 列表；
- `D == nullopt` 时虽然构造了 `D_vectors` 以维持新输出生命周期，但这些新输出没有作为函数返回值交给 Python。

因此正常 Python API 必须传入 `out`。C++ 的 `D == nullopt` 分支并不是一个完整、可供 Python 获得新 D 的独立输出分配接口。

### 7.2 单输出模式

设：

```text
D[0].shape = [sum(M_i), N]
```

C++ 从首地址开始为每组建立 view：

```text
view_0 -> D[0] 的前 M_0*N 个元素
view_1 -> 接下来的 M_1*N 个元素
...
```

这些 view 使用 `at::from_blob`，本身不拥有内存；真正 owner 是调用者传入的 `D[0]`。函数执行期间 `D` 参数和上层 `out` 保持 owner 存活。

偏移量按：

```text
D_shape[0] * D_shape[1] * D[0].element_size()
```

计算。因此单输出模式的可靠使用条件是：

- 每组输出为二维；
- dtype 相同；
- 输出紧密连续；
- 总容量恰好或至少容纳各组元素数之和。

## 8. 空组处理

MoE 场景可能存在某个专家收到 0 个 token：

```text
M_i = 0
```

函数不会把该组加入实际 GEMM wrapper vector：

```cpp
if (te_A.numel() == 0 || te_B.numel() == 0) {
    if (out_tensor.numel() != 0 && !accumulate)
        out_tensor.zero_();
    if (bias[i].numel() != 0 && grad)
        bias[i].zero_();
    if (pre_gelu_out[i].numel() != 0)
        pre_gelu_out[i].zero_();
    continue;
}
```

结果：

- 空组不提交 GEMM；
- 非累加模式的非空输出被清零；
- dbias 和 pre-GELU 被清零；
- 实际传给底层的 `num_gemms` 是“非空组数量”，不是原始 A.size()。

注意一个隐含后果：空组被过滤后，底层 A/B/D/bias/pre-GELU vector 仍保持彼此对齐，但它们的下标不再等同于原始 group id。该函数只需执行计算，因此不额外返回这个映射。

## 9. 量化和 FP8 block scaling

### 9.1 scale swizzle

所有非空组被包装后，统一处理 scale inverse：

```cpp
multi_tensor_swizzle_scales_for_gemm(
    te_A_wrappers, transa, !transa);

multi_tensor_swizzle_scales_for_gemm(
    te_B_wrappers, !transb, transb);
```

返回的临时 Tensor 保存于：

```cpp
swizzled_scale_inverses_list
```

以保证 GEMM 提交期间生命周期有效。

### 9.2 Blackwell 上的 block-scaling 兼容路径

SM100 及以上架构中，如果检测到 FP8 block scaling：

1. 要求所有有效 A/B Tensor 要么全部采用 block scaling，要么全部不采用；
2. 不能混合 block-scaling 与非 block-scaling Tensor；
3. 将每个 A/B 转换为 MXFP8；
4. swizzle 相应 scale；
5. 把执行布局统一改为 TN：

```cpp
transa = true;
transb = false;
```

这样避免实际转置数据，并绕过 cuBLASLt 对该 FP8 block scaling 模式缺乏原生支持的问题。

这里检查的是过滤掉空组后的 TensorWrapper 列表。

### 9.3 `quantization_params` 为什么不在 C++ 签名中

Python `general_grouped_gemm` 接收：

```python
quantization_params: List[Optional[Quantizer]]
```

但正常 C++ 调用没有把该列表作为参数传下去。原因是输入 A/B 如果已经是量化 Tensor，其量化元数据包含在各自 Python 对象及 `TensorWrapper` 中。

当前 Python 代码直接读取 `quantization_params[0]` 来判断 DebugQuantizer 回退：

- 若第一项是 DebugQuantizer，逐组调用单 GEMM；
- 否则进入本 C++ multi-tensor 路径。

因此调用者仍需提供长度至少为 1 的 quantization_params 列表；通常长度应与组数一致。

## 10. 底层并发执行

### 10.1 默认 multi-stream cuBLASLt

默认执行：

```cpp
multi_stream_cublas_gemm(...)
```

流程：

1. 在主/current stream 上记录 event；
2. 每个 compute stream 等待该 event；
3. 第 i 组提交到：

```text
compute_stream[i % num_streams]
```

4. workspace 同样使用：

```text
workspace[i % num_streams]
```

5. 各 compute stream 完成后记录 event；
6. 主 stream 等待所有被使用的 compute stream。

因此从调用者当前 stream 的依赖视角看：

```text
此前主流工作
    -> 多个 compute stream 并发 GEMM
    -> 后续主流工作
```

函数释放 Python GIL 后提交这些操作，但并不执行全设备同步。

### 10.2 可选 CUTLASS grouped GEMM

环境变量：

```text
NVTE_USE_CUTLASS_GROUPED_GEMM=1
```

会请求 CUTLASS 路径，但当前只在 Hopper（SM90）并满足全部条件时使用：

- bias 数组全部为空；
- pre-GELU 数组全部为空；
- A/B/D dtype 相同；
- dtype 为 FP16 或 BF16；
- 所有组的 K 相同；
- K 是 128 的倍数。

否则回退到 multi-stream cuBLASLt。若设置：

```text
NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK=1
```

回退时会打印警告。

注意：这条 `nvte_multi_tensor_gemm` 内部的 Hopper CUTLASS 路径，与 `gemm.cpp` 后续 `te_general_grouped_gemm_for_grouped_tensor` 所调用的、要求 Blackwell/cuBLAS 13.2 的另一套 grouped API 不是同一个入口，不应混为一谈。

## 11. 返回值

C++ 返回类型：

```cpp
std::optional<std::vector<at::Tensor>>
```

实际末尾：

```cpp
return bias;
```

因此返回的是传入的 bias/dbias vector：

- `grad=false`：通常返回 bias 列表或 empty 占位列表；
- `grad=true`：返回底层写入后的 grad_bias 列表。

Python wrapper：

```python
bias = tex.te_general_grouped_gemm(...)
return out, bias, gelu_input
```

最终 Python 返回三项：

```text
(out, bias_or_grad_bias, gelu_input)
```

其中：

1. `out`
   - 由调用者传入，原地写入；
   - single_output 时通常是只含总输出 Tensor 的列表。

2. `bias_or_grad_bias`
   - 前向是原 bias/empty 占位；
   - 反向且 use_bias 时是计算出的每组 bias gradient。

3. `gelu_input`
   - 启用 GELU 时为 pre-GELU Tensor 列表；
   - 否则是 empty Tensor 占位列表。

## 12. 主要实现步骤

### 12.1 检查单输出模式

```cpp
if (single_output && D == std::nullopt) {
    NVTE_ERROR(
        "not implemented, D should be allocated "
        "for single output case.");
}
```

### 12.2 固定 CUDA device

```cpp
at::cuda::CUDAGuard device_guard(workspace[0].device());
```

代码假定所有 Tensor 位于同一 device。

### 12.3 遍历原始 group

对每组：

1. 包装 A/B；
2. 推导 D shape；
3. 创建/选择输出；
4. 处理空组；
5. 包装 D、bias、pre-GELU；
6. 将有效组加入 wrapper vectors。

### 12.4 处理量化 scale

- multi-tensor scale swizzle；
- Blackwell block scaling 到 MXFP8 的转换；
- 保存临时 scale Tensor 生命周期。

### 12.5 构造 C API 指针数组

将 wrapper 转成：

```cpp
std::vector<NVTETensor>
```

包括：

- A vector；
- B vector；
- D vector；
- bias vector；
- pre-GELU vector；
- workspace vector。

### 12.6 调用底层并释放 GIL

```cpp
NVTE_SCOPED_GIL_RELEASE({
    nvte_multi_tensor_gemm(...);
});
```

### 12.7 返回 bias/dbias

输出 D 已原地写入，返回值只传回 bias/dbias 列表。

## 13. 典型用法

### 13.1 多专家前向、一个连续输出

```python
# 每个专家有独立权重
A = [
    weight_0,  # [N, K]
    weight_1,  # [N, K]
    weight_2,  # [N, K]
]

# 输入按专家收到的 token 数切分
m_splits = [10, 3, 7]
B = list(torch.split(input, m_splits))  # 每项 [M_i, K]

out = torch.empty(
    [sum(m_splits), N],
    dtype=input.dtype,
    device="cuda",
)

general_grouped_gemm(
    A,
    B,
    [out],
    [None] * len(A),
    input.dtype,
    layout="TN",
    m_splits=m_splits,
    single_output=True,
)
```

数学式：

```text
out segment i = B_i @ A_i.T
```

### 13.2 每组独立输出

```python
out = [
    torch.empty([M_i, N_i], device="cuda", dtype=dtype)
    for M_i, N_i in group_shapes
]

general_grouped_gemm(
    A,
    B,
    out,
    [None] * len(A),
    dtype,
    layout="TN",
    m_splits=m_splits,
    single_output=False,
)
```

适合每组 N 不同或调用者本来就需要离散结果的情况。

### 13.3 累加模式

```python
general_grouped_gemm(
    A,
    B,
    out,
    [None] * len(A),
    dtype,
    layout="NT",
    m_splits=m_splits,
    grad=True,
    accumulate=True,
)
```

每组：

```text
out_i[:] = B_i.T @ A_i + out_i_old
```

常用于分批累加权重梯度。

## 14. 最容易踩坑的地方

1. **Python 张量视角下默认 TN 是 `B_i @ A_i.T`。**

2. **A.size() 是组数的唯一主循环依据。**  
   B、bias、pre-GELU 等长度不会被统一检查。

3. **`m_splits` 在本 C++ 函数中未使用。**  
   实际 group shape 来自 A/B；但上层和 DebugQuantizer 路径仍依赖它。

4. **正常 Python API应预分配 out。**  
   C++ 的 `D == nullopt` 分支不会把新建 D 作为返回值返回。

5. **single_output 要求连续二维布局。**  
   偏移量硬编码为 `D_shape[0] * D_shape[1] * element_size`。

6. **没有独立 C，也没有任意 alpha/beta。**  
   只能选择覆盖或 `+ D_old`。

7. **unused bias/GELU 不是空 vector，而是等长 empty Tensor 列表。**

8. **single_output + GELU 的上层组织存在明确 TODO 式注释。**  
   不应未经验证直接组合使用。

9. **所有 Tensor 和 workspace 应在同一 CUDA device。**

10. **grouped 不代表所有组同形状。**  
    不同 M/N/K 是允许的，但共享 transpose、dtype 和执行策略。

11. **默认实现是 multi-stream cuBLASLt。**  
    CUTLASS 只在 Hopper、显式开启且满足严格条件时采用。

12. **CUDA 执行仍是异步的。**  
    主 stream 通过 event 与内部 compute streams 建立依赖，但函数返回不等于设备同步。

## 15. 一句话总结

`te_general_grouped_gemm` 是一个面向 grouped linear/MoE 等场景的多 Tensor GEMM 编排器：它让多组不同形状的 `B_i @ A_i.T`（或 NN/NT 变体）共享一次 Python/C++ 调用，可将结果写入离散输出或一个连续总输出，并支持 bias、GELU/dGELU、dbias、FP8 scale 和梯度累加；默认执行方式是多 CUDA stream 上的多次 cuBLASLt GEMM，而不是简单的等形状 batched GEMM。
