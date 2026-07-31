# Transformer Engine PyTorch `gemm` 接口解析

本文分析的是：

```text
transformer_engine/pytorch/csrc/extensions/gemm.cpp:127
```

对应函数：

```cpp
std::vector<py::object> gemm(
    py::handle A,
    bool transa,
    py::handle B,
    bool transb,
    py::object D,
    py::handle quantizer,
    std::optional<DType> out_dtype,
    MaybeTensor bias,
    DType bias_type,
    bool gelu,
    MaybeTensor gelu_in,
    bool grad,
    at::Tensor workspace,
    size_t workspaceSize,
    bool accumulate,
    bool use_split_accumulator,
    CommOverlapCore* comm_overlap,
    std::optional<CommOverlapType> comm_type,
    MaybeTensor extra_output,
    bool bulk_overlap,
    float alpha,
    std::optional<float> beta);
```

它不是一个只计算 `A @ B` 的薄包装，而是 Transformer Engine（下文简称 TE）PyTorch 扩展中的通用单 GEMM 入口。它负责：

1. 接收普通 PyTorch Tensor 或 TE 量化 Tensor；
2. 检查并推导矩阵形状；
3. 创建或复用输出；
4. 设置 `alpha`、`beta` 和累加语义；
5. 可选融合 bias、GELU、dGELU 和 bias gradient；
6. 支持 FP8 等低精度输入、输出量化和 scale 布局转换；
7. 可选将 GEMM 与 All-Gather / Reduce-Scatter 通信重叠；
8. 最终调用 `nvte_cublas_gemm_v2`，后者再通过 cuBLASLt 执行矩阵乘。

## 1. 从 Python 到 CUDA 的调用链

正常调用链如下：

```text
Python:
transformer_engine/pytorch/cpp_extensions/gemm.py::general_gemm
    |
    | 根据 layout 生成 transa/transb，申请 workspace
    v
PyBind:
transformer_engine/pytorch/csrc/extensions/pybind.cpp
    tex.generic_gemm(...)
    |
    v
C++ PyTorch wrapper:
transformer_engine/pytorch/csrc/extensions/gemm.cpp::gemm
    |
    | 包装 Tensor、推导输出、配置量化/epilogue/通信
    v
TE C API:
transformer_engine/common/gemm/cublaslt_gemm.cu::nvte_cublas_gemm_v2
    |
    v
TE cuBLASLt wrapper:
cublas_gemm(...)
    |
    v
cuBLASLt / CUDA kernel
```

PyBind 注册名是 `generic_gemm`：

```cpp
m.def("generic_gemm", transformer_engine::pytorch::gemm,
      "Compute GEMM (matrix-matrix multiply)",
      py::arg("A"), py::arg("transA"),
      py::arg("B"), py::arg("transB"),
      py::arg("D"), py::arg("quantizer"),
      py::arg("output_dtype"), py::arg("bias"),
      py::arg("bias_type"), py::arg("gelu"),
      py::arg("gelu_in"), py::arg("grad"),
      py::arg("workspace"), py::arg("workspace_size"),
      py::arg("accumulate"), py::arg("use_split_accumulator"),
      py::arg("comm_overlap") = nullptr,
      py::arg("comm_type") = std::nullopt,
      py::arg("extra_output") = std::nullopt,
      py::arg("bulk_overlap") = false,
      py::arg("alpha") = 1.0f,
      py::arg("beta") = std::nullopt);
```

用户通常不会直接调用它，而是调用 Python 的 `general_gemm(...)`。后者只允许 `"TN"`、`"NN"`、`"NT"` 三种 layout，准备 workspace 后调用 `tex.generic_gemm`。

## 2. 数学语义：不要把形参顺序直接理解成 PyTorch 的 `A @ B`

### 2.1 底层 C API 的定义

`nvte_cublas_gemm_v2` 的接口文档把运算写成：

```text
D = alpha * op(A) * op(B) + beta * C
```

其中 `op(X)` 由对应的 transpose 标志决定。

但是，TE 的 PyTorch 包装层与 cuBLASLt 的布局约定使得“Python 张量视角”的结果顺序看起来是反过来的。对 PyTorch 用户，更实用的理解是：

```text
D = alpha * op_py(B) @ op_py(A) + beta * D_old
```

这里：

```text
op_py(A) = A.T if transa else A
op_py(B) = B.T if transb else B
```

更准确地说，若先把输入的前若干维压平成二维：

```text
A2.shape = [A0, A1]
B2.shape = [B0, B1]
```

则三个 Python layout 的常见含义是：

| layout | `transa` | `transb` | Python 张量视角 |
|---|---:|---:|---|
| `TN` | true | false | `B2 @ A2.T` |
| `NN` | false | false | `B2 @ A2` |
| `NT` | false | true | `B2.T @ A2` |

`TT` 没有被 Python `general_gemm` 接口接受。

最典型的线性层前向是：

```python
# weight: [out_features, in_features]
# inp:    [tokens, in_features]
out, *_ = general_gemm(weight, inp)  # 默认 layout="TN"
```

仓库 `linear.py` 也明确注释：

```python
# Forward GEMM
# Note: y = x * w^T
gemm_out, *_, reduce_scatter_out = general_gemm(
    weightmat,
    inputmat_total,
    ...
)
```

因此这里实际得到：

```text
out = input @ weight.T
shape: [tokens, in_features] @ [in_features, out_features]
    -> [tokens, out_features]
```

### 2.2 `getGemmOutputShape` 的作用

函数定义位于 `gemm.cpp:41`：

```cpp
std::vector<size_t> getGemmOutputShape(
    const NVTEShape& A_shape,
    const bool transa,
    const NVTEShape& B_shape,
    const bool transb) {
  // Flatten outer dims to get 2D matrices
  const size_t A0 =
      A_shape.ndim > 0
          ? product(A_shape, 0, A_shape.ndim - 1)
          : 1;
  const size_t A1 =
      A_shape.ndim > 0
          ? A_shape.data[A_shape.ndim - 1]
          : 1;
  const size_t B0 =
      B_shape.ndim > 0
          ? product(B_shape, 0, B_shape.ndim - 1)
          : 1;
  const size_t B1 =
      B_shape.ndim > 0
          ? B_shape.data[B_shape.ndim - 1]
          : 1;

  // Check matrix dims
  NVTE_CHECK(
      (transa ? A1 : A0) == (transb ? B0 : B1),
      "Invalid matrix dimensions for GEMM (A=(",
      A0, ",", A1, "), transa=", transa,
      ", B=(", B0, ",", B1,
      "), transb=", transb, ")");

  // Construct output dims
  std::vector<size_t> ret;
  if (transb) {
    ret.emplace_back(B1);
  } else {
    // Unflatten B0
    for (size_t i = 0; i < B_shape.ndim - 1; ++i) {
      ret.emplace_back(B_shape.data[i]);
    }
  }
  if (transa) {
    ret.emplace_back(A0);
  } else {
    ret.emplace_back(A1);
  }
  return ret;
}
```

它有两个职责：

1. 将可能为多维的 A、B 按 TE GEMM 规则解释成两个二维矩阵，并检查矩阵乘的归约维 K 是否匹配；
2. 在不真正执行 GEMM 的情况下，返回 Python/ATen 视角下输出 D 的完整 shape。

它只处理 shape，不会：

- 分配输出；
- 转置或复制输入数据；
- 检查 dtype、device、stride、contiguous 或量化 scale；
- 执行 GEMM。

单 GEMM 在 `gemm.cpp:158` 使用它创建或检查 D；旧式 list-based grouped GEMM 在 `gemm.cpp:475` 对每组调用它，以确定每组输出或单个连续总输出中的 view 大小。

### 2.3 第一步：把多维输入逻辑压平成二维

设：

```text
A.shape = [a0, a1, ..., a(r-2), a(r-1)]
B.shape = [b0, b1, ..., b(s-2), b(s-1)]
```

函数定义：

```text
A0 = a0 * a1 * ... * a(r-2)
A1 = a(r-1)

B0 = b0 * b1 * ... * b(s-2)
B1 = b(s-1)
```

即：

```text
A2.shape = [A0, A1]
B2.shape = [B0, B1]
```

这里的“flatten”是形状推导中的逻辑解释。此函数本身没有调用 `reshape`、没有分配新 Tensor，也没有移动数据。

例如：

```text
A.shape = [2, 3, 4]
```

会被视为：

```text
A2.shape = [2*3, 4] = [6, 4]
```

而：

```text
B.shape = [5, 7, 4]
```

会被视为：

```text
B2.shape = [5*7, 4] = [35, 4]
```

若采用 TN，则核心二维乘法是：

```text
B2[35, 4] @ A2.T[4, 6] -> D2[35, 6]
```

函数返回的不是扁平 `[35, 6]`，而是把 B 的外维恢复后得到：

```text
D.shape = [5, 7, 6]
```

这正是注释 `Unflatten B0` 的含义。

### 2.4 第二步：检查 GEMM 的归约维

Python Tensor 视角的核心乘法是：

```text
op_py(B2) @ op_py(A2)
```

各操作数的逻辑形状：

| 标志 | `op_py(A2)` | `op_py(B2)` |
|---|---|---|
| `transa=false` | `[A0, A1]` | — |
| `transa=true` | `[A1, A0]` | — |
| `transb=false` | — | `[B0, B1]` |
| `transb=true` | — | `[B1, B0]` |

矩阵乘 `op_py(B2) @ op_py(A2)` 要求：

```text
op_py(B2) 的最后一维
    ==
op_py(A2) 的第一维
```

写成代码正是：

```cpp
(transa ? A1 : A0) == (transb ? B0 : B1)
```

分四种组合展开：

| `transa` | `transb` | 核心乘法 | 合法性条件 |
|---:|---:|---|---|
| false | false | `[B0,B1] @ [A0,A1]` | `B1 == A0` |
| true | false | `[B0,B1] @ [A1,A0]` | `B1 == A1` |
| false | true | `[B1,B0] @ [A0,A1]` | `B0 == A0` |
| true | true | `[B1,B0] @ [A1,A0]` | `B0 == A1` |

虽然函数能计算 TT 的形状，但 Python `general_gemm` 当前只接受 TN、NN、NT，不接受 TT。

维度不匹配时，`NVTE_CHECK` 立即报错，错误信息给出逻辑压平后的 A/B 二维形状及两个 transpose 标志，例如：

```text
Invalid matrix dimensions for GEMM
(A=(N,K), transa=1, B=(M,K2), transb=0)
```

这意味着应先检查 `K == K2`，而不是只比较原始 Tensor 的 rank。

### 2.5 第三步：构造输出 shape

矩阵乘结果的二维形状是：

```text
[op_py(B2) 的行数, op_py(A2) 的列数]
```

因此：

```text
输出行数 = transb ? B1 : B0
输出列数 = transa ? A0 : A1
```

但当 `transb=false` 时，B0 原本是由 B 的所有外维相乘得到的。函数会恢复这些外维：

```cpp
if (transb) {
  ret.emplace_back(B1);
} else {
  for (size_t i = 0; i < B_shape.ndim - 1; ++i) {
    ret.emplace_back(B_shape.data[i]);
  }
}
```

最后追加 A 提供的输出列维：

```cpp
ret.emplace_back(transa ? A0 : A1);
```

完整规则：

```text
transb=false:
    D.shape = [B.shape[0], ..., B.shape[-2],
               transa ? A0 : A1]

transb=true:
    D.shape = [B1, transa ? A0 : A1]
```

注意不对称性：

- 只有 `transb=false` 时才恢复 B 的外维；
- A 的外维永远不会逐维恢复，只会以乘积 A0 的形式作为最后一维；
- `transb=true` 时 B 的原外维同样只以乘积 B0 参与归约，不会保留在输出中。

这是因为 Python 视角下 B 是矩阵乘的左操作数，A 是右操作数。

### 2.6 四种 transpose 组合的输出

| layout | `transa` | `transb` | 合法性条件 | 输出 shape |
|---|---:|---:|---|---|
| NN | false | false | `B1 == A0` | `[*B_outer, A1]` |
| TN | true | false | `B1 == A1` | `[*B_outer, A0]` |
| NT | false | true | `B0 == A0` | `[B1, A1]` |
| TT | true | true | `B0 == A1` | `[B1, A0]` |

这里：

```text
A0 = product(A.shape[:-1])
A1 = A.shape[-1]
B0 = product(B.shape[:-1])
B1 = B.shape[-1]
```

其中 `*B_outer` 表示原样保留 `B.shape[:-1]` 的每一维。

### 2.7 具体例子

#### 例 1：默认 TN 线性层

```text
A = weight.shape = [N, K]
B = input.shape  = [M, K]
transa = true
transb = false
```

压平：

```text
A0=N, A1=K
B0=M, B1=K
```

检查：

```text
A1 == B1 -> K == K
```

输出：

```text
D.shape = [M, A0] = [M, N]
```

对应：

```text
D = input @ weight.T
```

#### 例 2：多维激活的 TN 线性层

```text
A.shape = [N, K]
B.shape = [batch, sequence, K]
```

逻辑二维矩阵：

```text
A2 = [N, K]
B2 = [batch*sequence, K]
```

函数恢复 B 的外维，得到：

```text
D.shape = [batch, sequence, N]
```

#### 例 3：NN 输入梯度

```text
A.shape = [N, K]   # weight
B.shape = [M, N]   # dY
transa = false
transb = false
```

检查：

```text
A0 == B1 -> N == N
```

输出：

```text
D.shape = [M, K]
```

对应：

```text
dX = dY @ weight
```

#### 例 4：NT 权重梯度

```text
A.shape = [M, K]   # input
B.shape = [M, N]   # dY
transa = false
transb = true
```

检查：

```text
A0 == B0 -> M == M
```

输出：

```text
D.shape = [N, K]
```

对应：

```text
dW = dY.T @ input
```

#### 例 5：维度不匹配

```text
A.shape = [128, 64]
B.shape = [32, 63]
layout = TN
```

TN 要求：

```text
A1 == B1
64 == 63  # false
```

因此函数在输出分配和 GEMM 执行前报错。

### 2.8 一维、零维和零长度维的边界行为

#### 一维 Tensor

若：

```text
A.shape = [K]
```

则：

```text
A0 = product(A_shape, 0, 0) = 1
A1 = K
```

`product` 的实现把返回值初始化为 1，并遍历半开区间 `[begin,end)`，所以空区间的乘积明确为 1。B 同理。形状函数因此能为一维输入推导结果；不过实际 GEMM 是否接受相应的向量式组合，还要满足底层 Tensor/GEMM 的其他约束，正常线性层仍主要使用二维或可压平的多维 Tensor。

#### 零维 Tensor

代码为 `ndim == 0` 写了防越界默认值：

```text
A0=A1=1 或 B0=B1=1
```

但这不表示标量是受支持的 GEMM 输入。单 GEMM 主函数在调用 `getGemmOutputShape` 之后仍明确要求：

```cpp
NVTE_CHECK(A_shape.ndim >= 1,
           "Tensor A needs to have at least 1 dimension");
NVTE_CHECK(B_shape.ndim >= 1,
           "Tensor B needs to have at least 1 dimension");
```

因此零维保护主要避免函数自身无条件访问 `shape[-1]`，最终接口仍拒绝标量。

#### 包含 0 的维度

若外维乘积或最后一维为 0，A0/A1/B0/B1 可能为 0。只要归约维检查成立，函数仍能返回包含 0 的输出 shape。

后续 `gemm` 会检测：

```cpp
if (A_tensor.numel() != 0 && B_tensor.numel() != 0) {
    // launch GEMM
} else {
    // zero output when needed
}
```

所以 `getGemmOutputShape` 负责推导空输出的形状，是否跳过 kernel、是否清零则由调用者处理。

### 2.9 与 `checkGemmShape` 的配合

紧随其后的：

```cpp
bool checkGemmShape(
    const std::vector<size_t>& expected,
    const NVTEShape& actual) {
  if (expected.size() != actual.ndim) return false;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != actual.data[i]) return false;
  }
  return true;
}
```

职责不同：

```text
getGemmOutputShape:
    根据 A/B/transpose 推导 expected D shape，
    同时验证 GEMM 归约维。

checkGemmShape:
    将 expected 与调用者提供的实际 D shape
    做逐维严格比较。
```

单 GEMM 中：

```cpp
const auto& D_shape =
    getGemmOutputShape(A_shape, transa, B_shape, transb);

if (D.is_none()) {
  // 按 D_shape 创建输出
} else {
  NVTE_CHECK(
      checkGemmShape(D_shape, D_tensor.shape()),
      "GEMM output has invalid dims ...");
}
```

因此错误分成两类：

1. A/B 归约维不匹配：由 `getGemmOutputShape` 报错；
2. A/B 可以相乘，但已有 D 的 rank 或任一维错误：由 `checkGemmShape` 报错。

### 2.10 设计要点总结

`getGemmOutputShape` 的核心可压缩成：

```text
1. A -> [A0, A1]
2. B -> [B0, B1]
3. 检查 op_py(B) @ op_py(A) 的 K 相同
4. 输出行来自 op_py(B)
5. 输出列来自 op_py(A)
6. transb=false 时恢复 B 的外维
```

默认 TN：

```text
A.shape = [...A_outer, K]，压平后为 [N, K]
B.shape = [...B_outer, K]，压平后为 [M, K]
D.shape = [...B_outer, N]
```

也就是：

```text
D = B @ A.T
```

最值得记住的是：这个函数不是按形参书写顺序推导 `A @ B`，而是按 TE PyTorch 的逻辑布局推导 `op_py(B) @ op_py(A)`；这也是它的内维检查和输出维选择看起来“反向”的原因。

## 3. 与传统 `d = a @ b + c` 的异同

### 相同点

二者核心都是 GEMM，并且都可以表达矩阵乘、缩放和累加：

```text
传统写法: d = a @ b + c
BLAS 写法: D = alpha * op(A) @ op(B) + beta * C
```

若只看数学能力，选择适当的转置标志并令 `alpha=1`、`beta=1`，TE 接口也能表达乘积加旧矩阵。

### 不同点

#### 1. PyTorch 层的 A/B 观察顺序不同

TE 的常见 Python 调用是：

```text
general_gemm(weight, input, layout="TN")
```

结果是：

```text
input @ weight.T
```

而不是按形参顺序写成 `weight.T @ input`。

#### 2. 这里没有独立的任意矩阵 `C`

普通路径最终这样调用：

```cpp
nvte_cublas_gemm_v2(
    transa, transb,
    &alpha,
    A_tensor.data(),
    B_tensor.data(),
    &beta.value(),
    out_tensor.data(),  // C
    out_tensor.data(),  // D
    ...
);
```

也就是说底层 `C` 和 `D` 指向同一个输出 Tensor。底层实现还显式检查：

```cpp
NVTE_CHECK(C_tensor == D_tensor,
           "Currently nvte_cublas_gemm_v2 does not support different C and D tensors.");
```

因此这个包装接口能直接表达的是：

```text
D_new = alpha * matmul + beta * D_old
```

而不是传入一个独立、任意的矩阵 `C` 再生成另一个 `D`。

#### 3. `bias` 不等同于公式里的矩阵 C

`bias` 是 cuBLASLt epilogue 的广播向量，典型形状为输出最后一维，语义类似：

```text
D = matmul + bias
```

它不是完整的 `[M, N]` 累加矩阵。接口可以同时存在：

```text
alpha * matmul + beta * D_old + bias
```

并且还可继续融合 GELU。

#### 4. 支持低精度 Tensor 和输出量化

传统 `a @ b + c` 通常只描述数值运算；本接口还携带 FP8/NVFP4 等量化 Tensor 的 scale、scale inverse、amax 和 scaling mode，并能将结果量化输出。

#### 5. 支持融合 epilogue

前向可融合：

```text
matmul -> bias -> GELU
```

反向可融合 dGELU 和 bias gradient。这样可以减少独立 kernel launch 和中间显存读写。

#### 6. 支持通信与计算重叠

通过 `comm_overlap`、`comm_type`、`extra_output` 和 `bulk_overlap`，同一入口可走 All-Gather/GEMM 或 GEMM/Reduce-Scatter 的重叠路径。这不是传统单 GEMM 接口的一部分。

#### 7. workspace、SM 数量和累加器策略可控

接口显式接收 cuBLASLt workspace，可限制参与计算的 SM 数量，并能为低精度 GEMM 选择 split accumulator。

## 4. 每个参数的含义、作用和要求

### 4.1 `A`

```cpp
py::handle A
```

含义：

- GEMM 的第一个底层操作数；
- 在常见 `TN` 线性层中通常是权重 `weight[out_features, in_features]`；
- 可以是普通 PyTorch Tensor，也可以是 TE 支持的量化 Tensor。

作用：

- 由 `makeTransformerEngineTensor(A, none)` 转成 `TensorWrapper`；
- Wrapper 既保存数据指针和 dtype，也携带量化 scale、amax、scaling mode 等元数据。

要求：

- 不能是 `None`，否则报错 `Tensor A has not been provided`；
- 至少一维；
- 必须与 `B` 满足转置后的内维匹配；
- 正常路径假定所有 Tensor 位于同一 CUDA device；
- 量化类型必须具有与其 Tensor 类型相匹配的量化元数据。

### 4.2 `transa`

```cpp
bool transa
```

含义：

- 是否按转置形式使用底层 A。

来源：

```python
transa = layout[0] == "T"
```

作用：

- 参与内维检查和输出形状推导；
- 传递给 cuBLASLt；
- 决定 FP8 scale 如何 swizzle；
- 对 Blackwell 上的 FP8 block scaling 仿真路径，代码可能在内部将它改成 `true`。

要求：

- 必须与 A/B 的实际形状匹配；
- 不应脱离 layout 和 B 的形状单独理解。

### 4.3 `B`

```cpp
py::handle B
```

含义：

- GEMM 的第二个底层操作数；
- 在默认 `TN` 线性层前向中通常是输入 `input[tokens, in_features]`。

作用和要求与 A 类似：

- 不能是 `None`；
- 至少一维；
- 必须与 A 在转置后满足内维匹配；
- 应与 workspace、A、D、bias 等位于同一 device。

### 4.4 `transb`

```cpp
bool transb
```

含义：

- 是否按转置形式使用底层 B。

来源：

```python
transb = layout[1] == "T"
```

作用：

- 参与形状检查、输出形状推导、scale swizzle 和最终 GEMM 配置。

要求：

- Python `general_gemm` 当前只允许 `TN`、`NN`、`NT`，所以不会从该上层入口得到 `transa=true, transb=true`。

### 4.5 `D`

```cpp
py::object D
```

含义：

- 输出 Tensor；
- 也充当 `beta * C` 中的旧值 C，因为底层调用把 C 和 D 设为同一个 Tensor。

行为：

- 若为 `None`，用 `quantizer` 和 `out_dtype` 创建输出；
- 若已提供，则复用其存储并检查输出形状；
- 若同时提供 `out_dtype`，还会检查 D 的 dtype 与其一致；
- Python 层额外要求用户提供的 `out` 必须 contiguous。

要求：

- 形状必须严格等于 `getGemmOutputShape(...)` 的结果；
- 若 `accumulate=true`，D 必须包含调用者希望累加的旧值；
- 若输出需要非融合量化，内部会先创建未量化临时输出，再量化写入 D。

### 4.6 `quantizer`

```cpp
py::handle quantizer
```

含义：

- 输出量化器；不量化输出时为 `None`。

作用：

- D 未提供时，由它创建输出对象；
- D 已提供时，参与把 Python 对象包装成 TE Tensor；
- 决定输出量化能否与 GEMM 融合；
- 不能融合时，先生成未量化 D，再调用 `quantize(...)`。

当前代码中的融合判断：

- 没有 quantizer：不需要额外量化；
- 对低精度输入，只有“delayed scaling 的 Float8 quantizer + per-tensor scaling 输入”的特定组合可直接融合；
- BF16 输入到 FP8 输出、或其他 FP8 输出量化方式通常走未融合后量化；
- 自定义 Tensor 在 Python 层可能提前分派到 `custom_gemm`，根本不进入本 C++ 函数。

要求：

- 必须是 TE 能由 `convert_quantizer` 识别的对象或 `None`；
- 应与目标输出格式、dtype 和 scaling recipe 一致。

### 4.7 `out_dtype`

```cpp
std::optional<DType> out_dtype
```

含义：

- 期望的逻辑输出 dtype。

行为：

```cpp
DType output_dtype = out_dtype ? *out_dtype : A_tensor.dtype();
```

- 未指定时默认采用 A 的 dtype；
- D 已存在且显式指定时，检查两者一致；
- 若使用非融合量化，临时未量化输出也以此 dtype 创建。

要求：

- Python 层由 `TE_DType[out_dtype]` 转成 TE 枚举；
- 必须是 TE 和当前 GEMM/量化路径支持的 dtype；
- 不要把它误解为一定等于最终 Python 对象的物理存储格式：量化 Tensor 可能有逻辑 dtype 与量化存储。

### 4.8 `bias`

```cpp
MaybeTensor bias  // std::optional<at::Tensor>
```

含义：

- 前向时是融合 bias；
- `grad=true` 时，它更像“是否请求并承载 bias-gradient 语义”的开关：函数会新建 `bias_grad`，而不是把传入 bias 当作前向 bias 使用。

前向行为：

- 非 contiguous bias 会先转换为 contiguous；
- 写入 `MatmulConfig::bias_tensor`；
- 由 cuBLASLt epilogue 加到 GEMM 输出。

反向行为：

- 分配长度为 `B_shape` 最后一维的 `bias_grad`；
- 写入 `MatmulConfig::dbias_tensor`；
- 返回值第二项为该梯度。

要求：

- 可为 `None`；
- 前向 bias 的长度和 dtype 必须适合输出 epilogue；
- 所有 Tensor 应在相同 CUDA device；
- `grad=true` 时传入值本身的数据不会作为前向 bias 加到输出。

### 4.9 `bias_type`

```cpp
DType bias_type
```

含义：

- bias / GELU 辅助 Tensor 使用的 TE dtype 信息。

Python 层默认：

```python
bias_dtype = TE_DType[torch.bfloat16 if bias is None else bias.dtype]
```

作用：

- 对低精度输入，`gelu_type` 取 `bias_type`；
- 非低精度输入时，`gelu_type` 取输出 dtype。

要求：

- 有 bias 时应与 bias 实际 dtype 一致；
- 即使 bias 为 `None`，上层仍传一个默认 BF16 枚举。

### 4.10 `gelu`

```cpp
bool gelu
```

含义：

- 是否启用融合 GELU（前向）或 dGELU（反向）epilogue。

前向 `grad=false`：

- 分配与 D 同形状的 `pre_gelu_out`；
- 配置 `with_gelu_epilogue=true`；
- D 是 GELU 后结果；
- 第三个返回值保存 GELU 前的值，供反向使用。

反向 `grad=true`：

- 配置 `with_dgelu_epilogue=true`；
- 若提供 `gelu_in`，把它作为 epilogue auxiliary input。

要求：

- 反向 dGELU 实际需要有效的前向 GELU 输入，即通常应同时提供 `gelu_in`；
- 若请求 GELU/dGELU，底层配置要求 auxiliary tensor 存在。

### 4.11 `gelu_in`

```cpp
MaybeTensor gelu_in
```

含义：

- dGELU 所需的前向 pre-GELU 值。

作用：

- 仅在 `gelu=true && grad=true` 时采用；
- 前向路径不会读取它，而是自行分配并返回 pre-GELU 输出。

要求：

- 做融合 dGELU 时应提供；
- 形状应与 GEMM 输出一致；
- dtype 应与 `gelu_type` 相容。

### 4.12 `grad`

```cpp
bool grad
```

含义：

- 选择普通前向 epilogue，还是梯度计算 epilogue。

| `grad` | bias 配置 | GELU 配置 |
|---:|---|---|
| false | `bias_tensor` | `with_gelu_epilogue` |
| true | `dbias_tensor` | `with_dgelu_epilogue` |

要求：

- 它不会自动判断这是哪一种矩阵梯度；A、B、transpose 仍由调用者正确组织；
- 它只切换融合 epilogue 的前向/反向语义。

### 4.13 `workspace`

```cpp
at::Tensor workspace
```

含义：

- cuBLASLt 算法选择和执行所用的临时显存。

作用：

- `CUDAGuard` 使用它的 device，确保 cuBLASLt handle 在正确 GPU 上创建；
- 其数据指针被包装成 `DType::kByte` 的 TE Tensor；
- 最终传给普通 GEMM 或通信重叠实现。

Python 层默认大小：

- Hopper 及更新架构：约 32 MiB，另加 1024 bytes；
- 其他架构：4 MiB；
- Userbuffers 路径会为多个 stream 申请更大的区域。

要求：

- 必须是 CUDA Tensor；
- 必须和输入、输出在同一 device；
- 生命周期必须覆盖异步 kernel 的提交过程；
- 实际可用字节数至少应覆盖 `workspaceSize`。

### 4.14 `workspaceSize`

```cpp
size_t workspaceSize
```

含义：

- 告诉 TE workspace 有多少字节可用。

作用：

```cpp
makeTransformerEngineTensor(
    workspace.data_ptr(),
    std::vector<size_t>{workspaceSize},
    DType::kByte);
```

要求：

- 单位是 bytes，不是元素个数的一般含义；
- Python 上层传 `workspace.shape[0]`，因为 workspace 是一维 `uint8` Tensor，此时元素数正好等于字节数；
- 不得大于实际分配空间。

### 4.15 `accumulate`

```cpp
bool accumulate
```

含义：

- 是否把 GEMM 结果累加到已有 D。

它与 beta 的组合规则是：

| `accumulate` | `beta` 未提供时 | 显式 beta 要求 | 结果 |
|---:|---:|---|---|
| false | 0 | 必须为 0 | 覆盖 D |
| true | 1 | 可由调用者给出 | 缩放并累加旧 D |

代码：

```cpp
if (accumulate) {
    if (!beta) beta = 1.0f;
} else {
    if (!beta) beta = 0.0f;
    NVTE_CHECK(beta == 0.0, ...);
}
```

要求：

- 若为 true，应提供已经分配且包含有效旧值的 D；
- 若输入为空且 `accumulate=false`，代码会把非空输出清零；
- Python `general_gemm` 会先验证 beta：不累加时只能是 0 或 `None`。

### 4.16 `use_split_accumulator`

```cpp
bool use_split_accumulator
```

含义：

- 是否为低精度 GEMM 使用 split accumulator 策略。

作用：

- 写入 `MatmulConfig`；
- 影响 cuBLASLt/TE 选择的累加方式，主要用于低精度计算的性能与数值行为折中；
- FP8 block-scaling 输入在 Python 层会强制设为 true；
- 仓库配置说明该选项只在 Hopper 上生效。

要求：

- 应由量化 recipe 或硬件/精度策略决定；
- 它不是把输出拆分成多个 Tensor，也不是 grouped GEMM。

### 4.17 `comm_overlap`

```cpp
CommOverlapCore* comm_overlap
```

含义：

- 通信与 GEMM 重叠执行的控制对象；
- Python 参数名是 `ub`，通常对应 Userbuffers 通信器。

行为：

- 为 null：走普通 `nvte_cublas_gemm_v2`；
- 非 null：不直接调用普通 GEMM，而由 overlap 对象选择 bulk、atomic 或 split overlap 实现。

要求：

- 若提供它，Python 层要求同时提供 `comm_type`；
- 对象、缓冲区、通信组和输入切分必须已由上层正确配置。

### 4.18 `comm_type`

```cpp
std::optional<CommOverlapType> comm_type
```

含义：

- 通信重叠类型，主要是：
  - `AG`：All-Gather 与 GEMM 重叠；
  - `RS`：GEMM 与 Reduce-Scatter 重叠。

要求：

- 使用 `comm_overlap` 时必须有值，因为 C++ 路径直接调用 `comm_type.value()`；
- 不使用通信重叠时通常为 `None`；
- Python 层会交叉检查 `ub` 与 `ub_type`。

### 4.19 `extra_output`

```cpp
MaybeTensor extra_output
```

含义：

- 通信重叠路径需要的附加输出缓冲区；
- 常用于 Reduce-Scatter 的结果。

作用：

- 有值时包装成 TE Tensor传入 overlap 实现；
- 也作为函数第四个返回值返回；
- 没有时传一个空的 byte Tensor wrapper。

要求：

- 对多数 GEMM+RS overlap 路径，Python 层要求它非空；
- bulk overlap 且 userbuffer 非 FP8 的特定情况可例外；
- 形状、dtype 和存储空间必须符合相应通信实现的约定。

### 4.20 `bulk_overlap`

```cpp
bool bulk_overlap
```

含义：

- 是否选择 bulk overlap 路径。

行为：

```cpp
if (bulk_overlap) {
    comm_overlap->bulk_overlap(...);
} else if (comm_type == AG) {
    atomic_gemm_overlap_ag(...) 或 split_overlap_ag(...);
} else {
    atomic_gemm_overlap_rs(...) 或 split_overlap_rs(...);
}
```

要求：

- 只有 `comm_overlap` 非空时才有实际作用；
- 必须与通信器的类型和缓冲区设置一致。

### 4.21 `alpha`

```cpp
float alpha
```

含义：

- GEMM 乘积项的缩放系数。

数学作用：

```text
D_new = alpha * matmul + beta * D_old
```

默认值：

```text
1.0
```

要求：

- Python 层保证 `None` 时替换为 1；
- 当前接口类型是 host float，调用底层时传其地址；
- bias/GELU 属于 epilogue，不应把 alpha 理解为统一缩放所有 epilogue 输出。

### 4.22 `beta`

```cpp
std::optional<float> beta
```

含义：

- 旧 D（底层 C 项）的缩放系数。

默认规则：

- `accumulate=false`：默认 0，且禁止非零；
- `accumulate=true`：默认 1，也可显式指定其他 float。

要求：

- 想使用非零 beta 必须同时设置 `accumulate=true`；
- 当前包装没有独立 C，因此 beta 只能作用于 D 原有内容。

## 5. 主要实现步骤

### 5.1 固定 CUDA device

```cpp
at::cuda::CUDAGuard device_guard(workspace.device());
```

这是为了避免用户侧 `torch.cuda.set_device` 的当前状态与实际 Tensor device 不一致，确保 cuBLASLt handle 在 workspace 所属设备上创建。代码明确假定所有传入 Tensor 都在同一设备。

### 5.2 包装输入并识别低精度/缩放模式

```cpp
TensorWrapper A_tensor = makeTransformerEngineTensor(A, none);
TensorWrapper B_tensor = makeTransformerEngineTensor(B, none);

const bool low_precision =
    is_low_precision(A_tensor.dtype()) ||
    is_low_precision(B_tensor.dtype());
```

同时检查输入是否采用 1D/2D FP8 block scaling，以决定后续 scale swizzle 和 Blackwell 兼容路径。

### 5.3 推导输出并处理 alpha/beta

函数依据 A/B 形状和转置标志推导 D 形状，然后规范化 beta：

```text
accumulate=false -> beta=0
accumulate=true  -> beta=1（若调用者未指定）
```

### 5.4 创建或复用输出

- D 为空：由 quantizer 创建；
- D 已有：检查形状和可选 dtype；
- 若 GEMM 不能直接生成所需量化格式：创建未量化临时输出。

### 5.5 配置 bias 和 GELU/dGELU

前向配置：

```cpp
config.set_bias_tensor(...);
config.set_with_gelu_epilogue(gelu);
```

反向配置：

```cpp
config.set_dbias_tensor(...);
config.set_with_dgelu_epilogue(gelu);
```

两者共用：

```cpp
config.set_epilogue_aux_tensor(...);
```

### 5.6 配置 workspace 和可用 SM

```cpp
const int sm_count = transformer_engine::cuda::sm_count(device_id);
int num_math_sms =
    sm_count - getenv<int>("NVTE_EXT_MARGIN_SM", sm_count);
config.set_sm_count(num_math_sms);
```

`NVTE_EXT_MARGIN_SM` 用来给数据并行通信预留 SM。按这段公式，该环境变量表达“预留的 SM 数”，默认值等于全部 SM，因此默认计算出的 `num_math_sms` 是 0；底层以 0 表示交给 cuBLAS heuristics 决定。

### 5.7 处理量化 scale

普通非空输入会先按 GEMM 方向 swizzle scale：

```cpp
swizzle_scales_for_gemm(A_tensor, transa, !transa);
swizzle_scales_for_gemm(B_tensor, !transb, transb);
```

这些临时 Tensor 被保存在 vector 中，以保证异步 GEMM 提交期间生命周期有效。

对 Blackwell（SM100 及以上）且使用 FP8 block scaling 的输入，代码把它转换成 MXFP8 来模拟，因为 cuBLASLt 不原生支持该模式，并将 GEMM 统一改为 TN 形式以避免转置实际数据：

```cpp
convert_block_scaling_to_mxfp8_tensor(...);
transa = true;
transb = false;
```

### 5.8 执行普通 GEMM 或通信重叠 GEMM

普通路径：

```cpp
nvte_cublas_gemm_v2(
    transa, transb,
    &alpha,
    A_tensor.data(),
    B_tensor.data(),
    &beta.value(),
    out_tensor.data(),  // C
    out_tensor.data(),  // D
    te_workspace.data(),
    config,
    main_stream);
```

调用被 `NVTE_SCOPED_GIL_RELEASE` 包围，执行期间释放 Python GIL。

通信路径则由 `comm_overlap` 根据 bulk / AG / RS 和 atomic / split 选择专门实现。

### 5.9 空输入处理

若 A 或 B 的元素数为 0，不启动 GEMM：

- 非空 D 且不累加时，将 D 清零；
- 请求 bias grad 时将其清零；
- 累加模式下保留 D 原值。

### 5.10 必要时量化输出并打包返回

若不能在 GEMM 内融合目标量化：

```cpp
my_quantizer->quantize(unquantized_D_tensor, D_tensor);
```

最终固定返回四项：

```text
[D, bias_grad, pre_gelu_out, extra_output]
```

## 6. 返回值

函数 C++ 类型是：

```cpp
std::vector<py::object>
```

Python `general_gemm` 将其解包为：

```python
out, bias_grad, gelu_input, extra_output
```

各项含义：

1. `out`
   - GEMM 最终输出；
   - 可能是普通 Tensor，也可能是 quantizer 创建的量化对象。

2. `bias_grad`
   - 仅在提供 bias 且 `grad=true` 时生成；
   - 其他情况为 `None`。

3. `gelu_input`
   - 仅在 `gelu=true && grad=false` 时返回前向 pre-GELU 值；
   - 用于后续 dGELU；
   - 其他情况为 `None`。

4. `extra_output`
   - 通信重叠路径的附加输出；
   - 未提供时为 `None`。

## 7. 常见用法对应关系

### 7.1 普通线性层前向

```python
out, _, _, _ = general_gemm(
    A=weight,              # [N, K]
    B=input,               # [M, K]
    layout="TN",
    out_dtype=input.dtype,
)
```

等价核心数学式：

```text
out[M, N] = input[M, K] @ weight[N, K].T
```

### 7.2 融合 bias

```python
out, _, _, _ = general_gemm(
    weight,
    input,
    bias=bias,             # [N]
)
```

近似语义：

```text
out = input @ weight.T + bias
```

其中 bias 沿输出的前导维广播。

### 7.3 融合 bias 和 GELU

```python
out, _, pre_gelu, _ = general_gemm(
    weight,
    input,
    bias=bias,
    gelu=True,
)
```

近似语义：

```text
pre_gelu = input @ weight.T + bias
out      = GELU(pre_gelu)
```

### 7.4 累加到既有输出

```python
out, *_ = general_gemm(
    weight,
    input,
    out=old_out,
    accumulate=True,
    alpha=2.0,
    beta=0.5,
)
```

核心语义：

```text
old_out[:] = 2.0 * (input @ weight.T) + 0.5 * old_out
```

这里必须注意原地别名：作为 C 的旧值和作为 D 的新值是同一个 `out` 存储。

## 8. 使用时最值得注意的约束

1. **默认 `TN` 是 `B @ A.T`。**  
   这是阅读调用点时最重要的约定。

2. **没有独立 C 输入。**  
   beta 作用于 D 的旧内容；`bias` 也不是完整矩阵 C。

3. **所有 Tensor 应在同一 CUDA device。**  
   device guard 以 workspace.device 为准。

4. **已有输出必须 contiguous 且形状严格匹配。**

5. **非累加模式禁止非零 beta。**

6. **融合 dGELU 时需要有效的 `gelu_in`。**

7. **通信重叠参数必须成套出现。**  
   `comm_overlap` 与 `comm_type` 相互依赖，RS 通常还要求 `extra_output`。

8. **量化输出不保证总能与 GEMM 融合。**  
   不满足特定条件时会先写未量化临时 Tensor，再执行量化。

9. **接口是异步 CUDA 提交。**  
   函数使用当前 CUDA stream；返回不代表 GPU 已同步完成。

10. **`use_split_accumulator` 是数值/性能策略，不是输出切分。**

## 9. 一句话总结

`gemm.cpp:127` 的 `gemm` 是 TE PyTorch 层的“通用矩阵乘执行编排器”：它以常见 `B @ A.T` 的线性层布局为中心，把输出复用、`alpha/beta` 累加、bias/GELU/dGELU 融合、FP8 等量化、cuBLASLt workspace、SM 配额以及通信重叠统一封装到一次接口调用中；与简单的 `d = a @ b + c` 相比，它的核心矩阵乘相同，但张量观察顺序、C/D 原地别名、广播 bias、量化和融合/通信能力都更复杂。
