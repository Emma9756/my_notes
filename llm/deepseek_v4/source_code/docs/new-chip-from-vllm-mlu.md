---
title: "new-chip-from-vllm-mlu"
date: 2026-05-28
tags:
  - #LLM
  - #推理
  - #deepseek
  - #from_me
  - #待整理
status: 待整理
---

# 从 vllm-mlu 出发，为新芯片搭建推理框架

## 1. 文档目标

本文不是单纯介绍 vLLM，也不是单纯介绍 `vllm-mlu` 的代码细节，而是回答一个更工程化的问题：

**如果现在有一款全新的芯片，希望参考 `vllm-mlu`，从 0 搭建自己的推理框架，应该如何系统、有条理地推进？**

本文给出的路线分成两段：

1. **先接入 vLLM**  
   以最小代价获得一个“能运行、能加载模型、能调度请求、能做服务化”的基础推理系统。

2. **再逐步演化成独立框架**  
   当芯片特性越来越多，vLLM 的通用抽象已经不够承载时，再把关键层逐步抽离，形成自己的推理栈。

核心思想是：

- 不要一开始就重写整个框架。
- 先借助 vLLM 完成通用能力。
- 再把真正芯片特化的层一点点替换掉。
- `vllm-mlu` 就是一份非常典型的“平台适配 + 局部重写”的参考样板。

---

## 2. 先建立整体认知：vLLM 与 vllm-mlu 的关系

在工程上，`vllm-mlu` 不是“另起炉灶重写了一套完整推理框架”，而是：

- 复用 `vllm/` 主仓库的大部分通用能力
- 通过 **插件注册 + monkey patch + 平台特化实现**
- 把底层执行从 CUDA/GPU 迁移到 MLU

你可以把它理解成三层：

### 2.1 顶层：vLLM 通用框架

主仓库 `/data/users/tudj/qemu/vllm/vllm/` 提供：

- 请求生命周期管理
- LLM / Engine / Scheduler 抽象
- KV cache 抽象
- 模型注册与加载机制
- 服务入口
- 并行策略
- attention backend 抽象
- 采样、spec decode、多模态、量化等通用框架

### 2.2 中层：芯片平台适配层

`vllm-mlu` 提供：

- 新平台的 `Platform` 实现
- Worker / ModelRunner / Executor 的芯片化版本
- 新 attention backend
- 新算子绑定
- 新模型实现
- 新配置项与环境变量
- profiling、benchmark、graph、quant 等平台增强

### 2.3 底层：芯片 runtime / driver / kernel

例如：

- `torch.mlu`
- `cncl`
- 自定义 `mlu_ops`
- 图执行 runtime
- 芯片 SDK
- 自定义 kernel 库

所以，**新芯片框架最合理的第一步，不是重写 LLM 框架，而是先实现“中层”和“底层的接入”**。

---

## 3. 推荐路线：先接入 vLLM，再演进为独立框架

建议分成 4 个阶段。

## Phase A：最小可运行接入

目标：

- 单卡推理跑通
- 一个模型跑通
- 基础生成接口跑通
- 基础 KV cache 跑通
- attention 跑通

输出物：

- 一个类似 `vllm_<chip>/` 的插件仓库
- 一个可运行的 `examples/offline_inference.py`

## Phase B：平台增强接入

目标：

- 多卡推理
- 分布式通信
- 自定义 attention backend
- 图模式、编译、profiling
- 自定义量化
- MoE / 多模态 / speculative decode

输出物：

- 平台专属 `worker / runner / backend / ops / models`
- 可对标 `vllm-mlu` 的 benchmark 能力

## Phase C：深度特化

目标：

- 芯片专属调度
- 芯片专属 KV cache 布局
- 芯片专属图执行与编译流水线
- 芯片专属模型结构重写

输出物：

- 平台对关键路径拥有完全控制权

## Phase D：演化为独立推理框架

目标：

- 当 vLLM 抽象无法继续承载需求时
- 将部分或全部执行核心抽离成独立推理引擎

输出物：

- 自有 runtime 层
- 自有 scheduler / executor / graph / kernel 框架
- 对上保留兼容 API，或自定义新的 serving API

---

## 4. 从 0 搭建时，先做什么，不先做什么

### 4.1 第一优先级

先做这些：

- 平台识别
- 设备切换
- 显存查询
- 最小 Worker
- 最小 ModelRunner
- 最小模型加载
- 最小 attention 执行
- 最小 KV cache
- 单卡离线生成

### 4.2 暂时不要先做

一开始不要先做这些：

- 自己的完整服务框架
- 自己的完整调度系统
- 复杂多模型管理
- 复杂多租户
- 全套 benchmark 平台
- 全套量化支持
- 全套多模态支持
- 全套 speculative decode

原因很简单：

**没有最小闭环，后面所有高级优化都会变成无源之水。**

---

## 5. 建议的新仓库骨架

建议不要直接在 `vllm/` 主仓库里大改，而是先单独建一个插件仓库，比如：

```text
vllm-mychip/
├── README.md
├── setup.py
├── examples/
│   └── offline_inference/
│       └── offline_inference.py
├── csrc/
│   ├── ops/
│   └── bindings/
├── vllm_mychip/
│   ├── __init__.py
│   ├── _chip_utils.py
│   ├── _chip_ops.py
│   ├── logger.py
│   ├── chip_hijack.py
│   ├── chip_hijack_utils.py
│   ├── platforms/
│   │   └── mychip.py
│   ├── entrypoints/
│   │   ├── llm.py
│   │   └── openai/
│   │       └── api_server.py
│   ├── engine/
│   │   └── arg_utils.py
│   ├── config/
│   │   ├── model.py
│   │   ├── scheduler.py
│   │   └── vllm.py
│   ├── distributed/
│   │   ├── parallel_state.py
│   │   ├── device_communicators/
│   │   │   └── mychip_communicator.py
│   │   └── kv_transfer/
│   │       └── kv_connector/
│   ├── attention/
│   │   └── layer.py
│   ├── compilation/
│   │   ├── mychip_graph.py
│   │   └── fix_functionalization.py
│   ├── model_executor/
│   │   ├── parameter.py
│   │   ├── model_loader/
│   │   ├── layers/
│   │   └── models/
│   ├── v1/
│   │   ├── engine/
│   │   ├── executor/
│   │   ├── core/
│   │   │   └── sched/
│   │   ├── attention/
│   │   │   └── backends/
│   │   └── worker/
│   ├── profiler/
│   ├── benchmarks/
│   └── lora/
└── tests/
```

这个结构几乎可以直接参考 `vllm-mlu`。

---

## 6. 第一步：平台注册与插件机制

这是最先要建立的东西，因为没有它，vLLM 根本不知道你的芯片存在。

### 6.1 参考位置

参考 `vllm-mlu`：

- `vllm_mlu/__init__.py`
- `setup.py`
- `vllm_mlu/platforms/mlu.py`

### 6.2 你需要实现什么

至少需要：

- 平台注册函数
- hijack 注册函数
- `setup.py` 里的 `entry_points`
- 平台类 `MyChipPlatform`

### 6.3 代码骨架

`vllm_mychip/__init__.py`

```python
def register_mychip_platform():
    return "vllm_mychip.platforms.mychip.MyChipPlatform"


def register_mychip_hijack():
    from vllm_mychip import chip_hijack
    from vllm_mychip.model_executor.models import register_model
    register_model()
```

`setup.py` 需要：

```python
entry_points={
    "vllm.platform_plugins": [
        "mychip = vllm_mychip:register_mychip_platform",
    ],
    "vllm.general_plugins": [
        "mychip_hijack = vllm_mychip:register_mychip_hijack",
    ],
}
```

### 6.4 为什么这一层重要

它决定了两件事：

1. vLLM 会不会识别你的平台
2. 你的补丁逻辑会不会在启动时生效

---

## 7. 第二步：实现平台类 `MyChipPlatform`

这是最核心的入口层。

### 7.1 参考位置

参考：

- `vllm_mlu/platforms/mlu.py`

### 7.2 平台类的职责

平台类不是用来直接算模型的，它负责：

- 声明平台身份
- 指定设备类型
- 指定分布式通信后端
- 指定 worker 类
- 指定 attention backend
- 指定 graph wrapper
- 查询设备属性
- 修正平台配置

### 7.3 推荐最小实现骨架

`vllm_mychip/platforms/mychip.py`

```python
import os
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum

logger = init_logger(__name__)


envs.environment_variables.update({
    "MYCHIP_VISIBLE_DEVICES": lambda: os.environ.get("MYCHIP_VISIBLE_DEVICES"),
})


class MyChipPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name = "mychip"
    device_type = "mychip"
    dispatch_key = "MYCHIP"
    ray_device_key = "GPU"
    device_control_env_var = "MYCHIP_VISIBLE_DEVICES"
    dist_backend = "mycccl"
    simple_compile_backend = "inductor"

    supported_quantization = []
    additional_env_vars = [
        "MYCHIP_GRAPH_CAPTURE_LIST",
        "VLLM_MYCHIP_DEBUG",
    ]

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        major, minor = torch.mychip.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.mychip.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return torch.mychip.get_device_properties(device_id).total_memory

    @classmethod
    def set_device(cls, device: torch.device):
        torch.mychip.set_device(device)

    @classmethod
    def empty_cache(cls):
        torch.mychip.empty_cache()

    @classmethod
    def synchronize(cls):
        torch.mychip.synchronize()

    @classmethod
    def mem_get_info(cls):
        return torch.mychip.mem_get_info()

    @classmethod
    def get_attn_backend_cls(cls, *args, **kwargs) -> str:
        return "vllm_mychip.v1.attention.backends.flash_attn.MyChipFlashAttentionBackend"

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm_mychip.compilation.mychip_graph.MyChipGraphWrapper"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_mychip.distributed.device_communicators.mychip_communicator.MyChipCommunicator"

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        if vllm_config.parallel_config.worker_cls == "auto":
            vllm_config.parallel_config.worker_cls = (
                "vllm_mychip.v1.worker.gpu_worker.MyChipWorker"
            )
```

### 7.4 初始阶段建议

初始阶段不要把 `check_and_update_config()` 写得过重，只做：

- 选 worker
- 选 backend
- 选 graph wrapper
- 给少量必要默认值

---

## 8. 第三步：实现 hijack 机制

`vllm-mlu` 的一个核心经验是：

**新芯片通常不可能只靠平台注册就完全接入，必须对 vLLM 的一些类做定点替换。**

### 8.1 参考位置

参考：

- `vllm_mlu/mlu_hijack.py`
- `vllm_mlu/mlu_hijack_utils.py`

### 8.2 你需要的最小能力

- 能替换类方法
- 能注册新方法
- 能按模块导入触发替换

### 8.3 代码骨架

`vllm_mychip/chip_hijack_utils.py`

```python
class ChipHijackObject:
    hijack_objs = []

    @classmethod
    def apply_hijack(cls, obj, org_func, hijack_func):
        cls.hijack_objs.append((obj, org_func, hijack_func))
        if isinstance(org_func, str):
            name = org_func
        else:
            name = org_func.__name__.split("__")[-1]
        setattr(obj, name, hijack_func)
```

`vllm_mychip/chip_hijack.py`

```python
from vllm_mychip.logger import logger

logger.info("[MYCHIP] Apply monkey patch.")

import vllm_mychip.engine.arg_utils
import vllm_mychip.config.vllm
import vllm_mychip.entrypoints.llm
import vllm_mychip.v1.engine.llm_engine
import vllm_mychip.v1.engine.core
import vllm_mychip.v1.executor.abstract
```

### 8.4 为什么必须有这一层

因为平台适配不是只改一个文件，而是会分散到：

- 参数默认值
- 配置约束
- engine 行为
- worker 行为
- attention 行为
- 模型加载行为

---

## 9. 第四步：搭最小运行例子

在正式深入 worker 和算子之前，先准备一个最小离线推理例子。

### 9.1 推荐文件

`examples/offline_inference/offline_inference.py`

### 9.2 代码骨架

```python
import sys
from vllm import LLM, SamplingParams


def main(model_path: str):
    prompts = [
        "Hello, mychip.",
        "The future of inference is",
    ]

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=16,
    )

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_num_seqs=len(prompts),
        max_model_len=2048,
    )

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(output.prompt, output.outputs[0].text)


if __name__ == "__main__":
    main(sys.argv[1])
```

### 9.3 这个例子的意义

它不是示例而已，它是你的最小验收闭环：

- 插件是否注册成功
- 平台是否识别成功
- worker 是否启动成功
- 模型是否加载成功
- attention / KV cache 是否能运行

---

## 10. 第五步：实现 Worker

这是平台真正接到设备上的关键层。

### 10.1 参考位置

参考：

- `vllm_mlu/v1/worker/gpu_worker.py`

### 10.2 Worker 的职责

Worker 负责：

- 初始化设备
- 初始化分布式环境
- 检查 dtype
- 做显存 profiling
- 创建 model runner
- 初始化 KV cache
- 执行模型 step

### 10.3 推荐最小骨架

`vllm_mychip/v1/worker/gpu_worker.py`

```python
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu_worker import Worker

from vllm_mychip.v1.worker.gpu_model_runner import MyChipModelRunner


class MyChipWorker(Worker):
    def init_device(self):
        self.device = torch.device(f"mychip:{self.local_rank}")
        current_platform.set_device(self.device)
        current_platform.check_if_supports_dtype(self.model_config.dtype)

        self.model_runner = MyChipModelRunner(self.vllm_config, self.device)

    def execute_model(self, scheduler_output):
        return self.model_runner.execute_model(scheduler_output)
```

### 10.4 最初版本要优先实现什么

优先把以下链路打通：

- 单卡
- eager 模式
- 基础 forward
- 基础采样

图模式、异步、复杂 pipeline 先别上。

---

## 11. 第六步：实现 ModelRunner

`ModelRunner` 是真正“组织输入并调用模型”的地方。

### 11.1 参考位置

参考：

- `vllm_mlu/v1/worker/gpu_model_runner.py`

### 11.2 ModelRunner 的职责

- 保存各类 config
- 创建输入 buffer
- 加载模型
- 初始化 KV cache
- 准备 attention metadata
- 把调度结果整理成张量
- 调用 `self.model(...)`
- 处理 logits / sampling

### 11.3 推荐最小骨架

`vllm_mychip/v1/worker/gpu_model_runner.py`

```python
import torch

from vllm.model_executor.model_loader import get_model_loader
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class MyChipModelRunner(GPUModelRunner):
    def __init__(self, vllm_config, device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device = device
        self.model = None

    def load_model(self):
        model_loader = get_model_loader(self.vllm_config.load_config)
        self.model = model_loader.load_model(
            vllm_config=self.vllm_config,
            model_config=self.model_config,
        )

    def initialize_kv_cache(self, kv_cache_config):
        self.kv_cache_config = kv_cache_config

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        input_ids = scheduler_output.input_ids
        positions = scheduler_output.positions
        return self.model(input_ids=input_ids, positions=positions)
```

### 11.4 初期阶段的策略

一开始不要试图完全复刻 `gpu_model_runner.py` 的几千行逻辑。  
先从一个极简 runner 开始，只承载：

- 单模型
- 单卡
- decode-only 或简单 prefill+decode
- 基础 logits 输出

然后逐步补：

- KV cache
- paged attention
- async scheduling
- graph capture
- multimodal

---

## 12. 第七步：模型注册、模型类、权重加载

这是从“框架能跑”到“模型能正确跑”的关键。

### 12.1 参考位置

参考：

- `vllm_mlu/model_executor/models/__init__.py`
- `vllm_mlu/model_executor/models/registry.py`
- `vllm_mlu/model_executor/models/deepseek_v4.py`

### 12.2 你要做的三件事

1. 注册你的模型类
2. 实现模型结构
3. 实现权重映射与加载

### 12.3 注册代码骨架

`vllm_mychip/model_executor/models/__init__.py`

```python
from vllm import ModelRegistry


def register_model():
    ModelRegistry.register_model(
        "MyModelForCausalLM",
        "vllm_mychip.model_executor.models.my_model:MyChipModelForCausalLM",
    )
```

### 12.4 模型类骨架

`vllm_mychip/model_executor/models/my_model.py`

```python
import torch
from torch import nn


class MyChipDecoderLayer(nn.Module):
    def __init__(self, config, prefix: str):
        super().__init__()
        self.self_attn = nn.Identity()
        self.mlp = nn.Identity()

    def forward(self, hidden_states, positions, **kwargs):
        hidden_states = self.self_attn(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states


class MyChipModel(nn.Module):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            MyChipDecoderLayer(config, f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids, positions, **kwargs):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, **kwargs)
        return self.norm(hidden_states)


class MyChipModelForCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.model = MyChipModel(vllm_config=vllm_config, prefix=f"{prefix}.model")
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, positions, **kwargs):
        hidden_states = self.model(input_ids, positions, **kwargs)
        return hidden_states

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            if name in params:
                params[name].data.copy_(tensor)
                loaded.add(name)
        return loaded
```

### 12.5 权重加载原则

你需要重点设计：

- checkpoint 名称和当前模型参数名的映射
- TP/EP/PP 分片规则
- 是否存在 fused 参数
- 是否需要自定义量化权重格式

这部分非常容易成为平台落地的第一个大坑。

---

## 13. 第八步：attention backend 与模型 attention 的关系

这个是最容易混淆的地方。

### 13.1 两层结构

需要明确区分：

1. **模型里的 Attention 层**
   例如 `MyChipAttention`
2. **底层 Attention Backend**
   例如 `MyChipFlashAttentionBackend`

### 13.2 模型层在做什么

模型 attention 层负责：

- q/k/v/o projection
- RoPE
- local window / compress / cache 策略
- 调整输入输出 shape
- 选择调用 backend 或直接调用自定义 op

### 13.3 backend 层在做什么

backend 负责：

- 定义 metadata
- 组织 kernel 所需张量
- 对接 prefill / decode 两种模式
- 处理 paged cache 布局
- 调用底层 attention kernel

### 13.4 参考位置

参考：

- `vllm_mlu/platforms/mlu.py`
- `vllm_mlu/v1/attention/backends/flash_attn.py`
- `vllm_mlu/v1/attention/backends/mla/flashmla.py`
- `vllm_mlu/model_executor/models/deepseek_v4.py`

### 13.5 推荐最小 backend 骨架

`vllm_mychip/v1/attention/backends/flash_attn.py`

```python
class MyChipFlashAttentionBackend:
    @classmethod
    def get_name(cls) -> str:
        return "mychip_flash_attn"
```

初始阶段 backend 可以非常薄，只做最少的 metadata 对接。  
真正复杂的优化，可以后面逐步下沉进去。

---

## 14. 第九步：KV cache 与 paged attention

任何高吞吐推理框架，KV cache 都是重中之重。

### 14.1 初期目标

先回答这几个问题：

- 你的芯片上 KV cache 放在哪里
- block size 是多少
- page layout 是什么
- decode 时如何读写
- prefix cache 是否支持

### 14.2 最小策略

初期建议：

- 先支持单一 dtype
- 先支持统一 block size
- 先支持最基础 paged cache
- prefix cache 先不开
- compression 先不开

### 14.3 后续再补

- 多 dtype
- prefix cache
- hybrid kv cache
- shared kv cache
- sparse kv cache
- disaggregated kv transfer

### 14.4 参考位置

参考：

- `vllm_mlu/v1/core/kv_cache_utils.py`
- `vllm_mlu/v1/core/kv_cache_manager.py`
- `vllm_mlu/v1/worker/block_table.py`
- `vllm_mlu/v1/worker/gpu_model_runner.py`

---

## 15. 第十步：接入自定义算子

这一步才是真正体现芯片价值的地方。

### 15.1 典型自定义算子类型

- attention kernel
- paged attention kernel
- rotary embedding
- layernorm / rmsnorm
- fused MLP
- fused MoE
- quant matmul
- cache reshape / scatter / gather

### 15.2 推荐目录

```text
vllm_mychip/
├── _chip_ops.py
└── csrc/
    ├── ops/
    ├── kernels/
    └── bindings/
```

### 15.3 Python 绑定层骨架

`vllm_mychip/_chip_ops.py`

```python
import torch


def single_query_cached_kv_attn(*args, **kwargs):
    return torch.ops.mychip.single_query_cached_kv_attn(*args, **kwargs)


def reshape_paged_cache(*args, **kwargs):
    return torch.ops.mychip.reshape_paged_cache(*args, **kwargs)
```

### 15.4 原则

建议把 **算子绑定层** 和 **模型逻辑层** 分开：

- 模型逻辑层知道什么时候调用哪个算子
- 算子绑定层只负责稳定接口

不要把太多分支逻辑直接写死在 C++ binding 层。

---

## 16. 第十一步：分布式通信与并行

当单卡跑通后，再进入多卡。

### 16.1 最先补什么

优先补：

- TP
- 基础 all-reduce / all-gather
- 基础通信组初始化

### 16.2 后补什么

后续再补：

- PP
- EP
- DP
- DCP
- KV transfer

### 16.3 参考位置

参考：

- `vllm_mlu/distributed/parallel_state.py`
- `vllm_mlu/distributed/device_communicators/`
- `vllm_mlu/v1/executor/ray_executor.py`
- `vllm_mlu/v1/executor/multiproc_executor.py`

### 16.4 通信层建议

新芯片通常需要自己的：

- collective library
- process group 初始化逻辑
- tensor shard / gather 实现

如果芯片生态已有 PyTorch extension，优先对齐 PyTorch distributed 风格；  
如果没有，需要自己补足 communicator abstraction。

---

## 17. 第十二步：配置系统与环境变量

新芯片适配一定会引入大量平台专属配置。

### 17.1 参考位置

参考：

- `vllm_mlu/_mlu_utils.py`
- `vllm_mlu/engine/arg_utils.py`
- `vllm_mlu/config/model.py`
- `vllm_mlu/config/scheduler.py`
- `vllm_mlu/config/vllm.py`

### 17.2 你要管理的配置类型

- 平台可见设备
- graph capture 开关
- attention backend 选择
- quant 配置
- cache 配置
- benchmark / profiler 配置
- 特定模型的功能兼容开关

### 17.3 原则

配置要分层：

- **用户显式参数**
- **平台默认值**
- **平台强制修正**
- **实验特性环境变量**

不要把所有特殊逻辑都塞进一个大环境变量文件里。

---

## 18. 第十三步：图模式、编译、性能优化

这一层建议在功能稳定后再做。

### 18.1 参考位置

参考：

- `vllm_mlu/compilation/mlu_graph.py`
- `vllm_mlu/compilation/fix_functionalization.py`
- `vllm_mlu/config/vllm.py`
- `vllm_mlu/platforms/mlu.py`

### 18.2 推荐演进顺序

1. 先 eager 跑通
2. 再固定 batch / shape 做 warmup
3. 再做 graph capture
4. 再做 runtime graph dispatch
5. 再做复杂编译 pass 和 functionalization 修复

### 18.3 不要过早做的事

不要在以下事情没稳定前就做 graph：

- 模型结构不稳定
- KV cache 布局不稳定
- attention kernel 不稳定
- 输入 shape 还在频繁变化

---

## 19. 第十四步：可观测性、benchmark、profiling

这一步不是锦上添花，而是平台落地必需品。

### 19.1 参考位置

参考：

- `vllm_mlu/entrypoints/llm.py`
- `vllm_mlu/v1/engine/core.py`
- `vllm_mlu/v1/core/sched/scheduler.py`
- `vllm_mlu/profiler/`
- `vllm_mlu/benchmarks/`

### 19.2 最少需要的指标

- step latency
- model forward latency
- tokens/s
- memory usage
- kv cache 使用量
- scheduler batch usage
- attention kernel latency

### 19.3 为什么必须尽早补

没有这些指标，你根本不知道问题在：

- 调度
- 模型
- attention
- 通信
- graph
- 内存

---

## 20. 第十五步：逐步补高级特性

等基础框架稳定后，再逐步支持：

- 量化
- LoRA
- MoE
- speculative decoding
- 多模态
- disaggregated prefill/decode
- prefix cache
- sequence parallel
- expert parallel

建议顺序：

1. LoRA
2. 量化
3. MoE
4. speculative decode
5. 多模态

因为它们对主干侵入程度逐步增加。

---

## 21. 如果要从“接入 vLLM”演化到“自建独立框架”，如何做

这是本文最关键的第二部分。

### 21.1 先明确哪些层适合继续复用 vLLM

可以长期复用的层通常包括：

- OpenAI-compatible API
- sampling 参数结构
- 一部分模型注册机制
- 一部分 tokenizer / 输入输出协议
- benchmark 和测试框架的一部分

### 21.2 哪些层通常会逐渐独立

最容易逐步独立的层：

- runtime / graph executor
- device memory manager
- KV cache manager
- attention backend
- model runner
- executor
- scheduler
- distributed communicator

### 21.3 推荐的演化顺序

#### 阶段 1：插件化

仍然完全运行在 vLLM 框架内。

#### 阶段 2：执行核心独立

保留 vLLM 上层入口，但把：

- worker
- runner
- graph
- kernels

变成你自己的执行核心。

#### 阶段 3：调度核心独立

当你的芯片需要非常不同的：

- prefill/decode 流水线
- cache 策略
- 长序列策略
- 多流调度策略

就要开始独立 scheduler。

#### 阶段 4：完整 runtime 独立

最后再决定是否保留 vLLM 外层接口，或者形成完全独立的 serving stack。

### 21.4 为什么不推荐一步到位全独立

因为一开始最难的是：

- 算子正确性
- KV cache 正确性
- 权重加载正确性
- 多卡一致性

这些问题和“是否自己写服务框架”没有直接关系。  
先借助 vLLM 的通用外壳，更容易聚焦真正的芯片核心问题。

---

## 22. 推荐的开发顺序清单

下面给出一个可执行的顺序。

### Milestone 0：仓库初始化

- 建 `vllm_mychip/`
- 配 `setup.py`
- 配平台插件
- 配 general plugin

### Milestone 1：单卡最小跑通

- `MyChipPlatform`
- `MyChipWorker`
- `MyChipModelRunner`
- 一个最小模型类
- 一个最小 attention kernel
- 一个离线推理例子

验收：

- 单卡生成成功

### Milestone 2：正确性稳定

- 正确权重加载
- 基础 KV cache
- 基础 prefill + decode
- 基础 logits / sampling

验收：

- 与参考框架在小模型上对齐输出

### Milestone 3：多卡

- TP 通信
- all-reduce / all-gather
- 多卡模型分片加载

验收：

- TP=2/4 可运行

### Milestone 4：性能优化

- attention kernel 优化
- KV cache 布局优化
- graph capture
- warmup / compile

验收：

- 达到目标 tokens/s

### Milestone 5：高级特性

- quant
- LoRA
- MoE
- 多模态
- speculative decode

### Milestone 6：框架演化

- 将运行时、调度器、内存管理器抽离
- 逐步形成独立 runtime

---

## 23. 各目录建议职责

为了后续工程不失控，建议从第一天开始就把职责分清楚。

### `platforms/`

只做平台声明和配置修正，不做复杂执行逻辑。

### `engine/`

只做参数和引擎层 patch，不做底层 kernel。

### `config/`

只做配置结构、默认值、合法性检查。

### `v1/worker/`

做 worker、runner、输入整理、输出整理。

### `v1/attention/backends/`

做底层 attention 执行抽象与 metadata。

### `model_executor/models/`

做具体模型结构和权重加载。

### `model_executor/layers/`

做芯片专属层，如：

- rotary
- layernorm
- moe
- fused linear
- compressor
- indexer

### `distributed/`

做通信组、communicator、kv transfer。

### `compilation/`

做图执行和编译适配。

### `profiler/` / `benchmarks/`

做性能分析，不要和主执行逻辑纠缠。

---

## 24. 常见坑位

### 24.1 一开始就改太多层

建议先只改：

- platform
- worker
- runner
- model
- attention kernel

不要首版就把 scheduler、sampling、multimodal、spec decode 全改了。

### 24.2 模型类和 backend 职责混乱

要分清：

- 模型 attention 负责“结构和拼装”
- backend 负责“执行与 metadata”

### 24.3 权重加载映射不系统

建议单独维护：

- 参数命名映射规则
- 分片规则
- fused 参数规则

否则后面模型一多就会崩。

### 24.4 过早绑定复杂 graph

graph 一定要后置。

### 24.5 没有 benchmark 与 profiling

没有可观测性，就没有优化闭环。

### 24.6 不区分“兼容层”和“核心层”

建议始终把代码分成两类：

- **兼容 vLLM 的接口层**
- **芯片独有的核心实现层**

这样将来抽离成独立框架时最轻松。

---

## 25. 最后给新芯片团队的建议

如果你今天真的要基于 `vllm-mlu` 启动一个新芯片推理框架项目，最推荐的做法是：

1. 先 fork 一份 `vllm-mlu` 的目录结构思路，但不要直接复制所有实现。
2. 先最小化实现：
   - `Platform`
   - `Worker`
   - `ModelRunner`
   - 一个模型
   - 一个 attention kernel
3. 先跑通离线单卡。
4. 再补多卡。
5. 再补性能优化。
6. 最后再考虑是否独立出自己的 runtime 和 scheduler。

一句话总结：

**把 vLLM 当成外壳，把新芯片的价值集中放在执行路径、算子、KV cache、通信和图模式上；等这些层成熟后，再决定是否把框架整体独立出去。**

---

## 26. 推荐阅读顺序

如果你要结合源码进一步学习，建议按这个顺序看：

### vLLM 主仓库

1. `vllm/entrypoints/llm.py`
2. `vllm/engine/arg_utils.py`
3. `vllm/platforms/`
4. `vllm/v1/engine/`
5. `vllm/v1/worker/`
6. `vllm/v1/attention/backends/`
7. `vllm/model_executor/model_loader/`
8. `vllm/model_executor/models/`

### vllm-mlu

1. `examples/offline_inference/offline_inference.py`
2. `vllm_mlu/__init__.py`
3. `vllm_mlu/mlu_hijack.py`
4. `vllm_mlu/platforms/mlu.py`
5. `vllm_mlu/engine/arg_utils.py`
6. `vllm_mlu/v1/worker/gpu_worker.py`
7. `vllm_mlu/v1/worker/gpu_model_runner.py`
8. `vllm_mlu/v1/attention/backends/`
9. `vllm_mlu/model_executor/models/`
10. `vllm_mlu/compilation/`
11. `vllm_mlu/distributed/`

按这个顺序，你会最容易把“从例子到平台、从平台到执行、从执行到模型、从模型到底层 kernel”的整条链路串起来。

---

## 27. 分阶段实施清单与验收项

这一节把前面的路线进一步压缩成真正可执行的项目清单。  
建议新芯片团队把它当成研发里程碑表来用。

## Stage 0：项目立项与环境摸底

### 目标

明确芯片软件栈是否具备接入 vLLM 的基本条件。

### 需要确认的事情

- 是否存在 `torch.<device>` 风格的 PyTorch 设备扩展
- 是否支持基础张量算子
- 是否支持 autograd 之外的 inference-only runtime
- 是否有分布式通信库
- 是否有自定义 C++/CUDA-like kernel 注册机制
- 是否支持 basic profiler / event / stream
- 是否支持 device memory query

### 输出物

- 一份芯片运行时能力表
- 一份与 CUDA/ROCm/MLU 的差异清单
- 一份最小算子支持列表

### 验收项

- 能在 Python 中执行 `torch.<device>.set_device()`
- 能创建 device tensor
- 能完成一次简单 matmul
- 能读到显存信息

## Stage 1：最小平台接入

### 目标

让 vLLM 能识别你的芯片平台。

### 任务清单

- 新建 `vllm_<chip>/`
- 写 `setup.py` 插件入口
- 写 `__init__.py` 注册函数
- 写 `platforms/<chip>.py`
- 配置 `MYCHIP_VISIBLE_DEVICES`

### 输出物

- 插件仓库可安装
- vLLM 启动时可识别 `MyChipPlatform`

### 验收项

- 打印日志时可看到平台插件被加载
- `current_platform.device_type` 为你的设备类型
- `current_platform.get_device_name()` 可返回芯片名

## Stage 2：单卡最小推理闭环

### 目标

跑通一次最简单的离线生成。

### 任务清单

- 实现 `MyChipWorker`
- 实现 `MyChipModelRunner`
- 实现一个最小 `MyChipModelForCausalLM`
- 实现最基础 `load_weights`
- 写 `examples/offline_inference/offline_inference.py`

### 输出物

- 可运行的离线推理例子

### 验收项

- 单卡可加载模型
- 单条 prompt 可返回文本
- 输出不报 device / dtype / shape 错误

## Stage 3：基础正确性

### 目标

确认不是“碰巧能跑”，而是真的逻辑正确。

### 任务清单

- 对齐 tokenizer 与输入格式
- 对齐 logits 输出
- 对齐采样流程
- 验证不同 batch size
- 验证 prefill + decode

### 输出物

- 一组 correctness test

### 验收项

- 小模型上与参考实现输出接近
- greedy decoding 结果稳定
- batch=1 与 batch>1 行为一致

## Stage 4：KV cache 与 attention 正式接入

### 目标

让 decode 不再依赖重复全量计算，而是进入真正的推理框架模式。

### 任务清单

- 实现最小 KV cache layout
- 实现 block table
- 实现 paged attention 或等价机制
- 实现基础 attention backend
- 在 runner 中接入 attn metadata

### 输出物

- decode 阶段使用 KV cache
- attention backend 可运行

### 验收项

- 长输出时 latency 明显低于“每步全量重算”
- 连续 decode 不发生 cache 越界
- block size 改动后行为可解释

## Stage 5：多卡与并行

### 目标

让模型能在多卡上运行。

### 任务清单

- 实现 communicator
- 接入 TP all-reduce / all-gather
- 验证分片权重加载
- 支持多进程 worker

### 输出物

- TP=2/4 运行能力

### 验收项

- 多卡输出正确
- 多卡吞吐高于单卡
- 通信错误可观测

## Stage 6：性能优化第一轮

### 目标

让框架从“能跑”进入“可用”。

### 任务清单

- attention kernel 优化
- layernorm / rope / mlp 融合
- buffer 预分配
- 减少 host-device copy
- 增加 profiler 指标

### 输出物

- 一轮性能报告

### 验收项

- 能稳定输出 tokens/s
- 能定位瓶颈在 attention / 通信 / 采样中的哪一层
- 主要热点算子已经替换为芯片自定义实现

## Stage 7：图模式与编译

### 目标

进一步降低运行时调度开销。

### 任务清单

- 设计 graph capture 条件
- 增加 graph wrapper
- 支持 warmup
- 支持 graph batch size 列表
- 处理 compile / functionalization 问题

### 输出物

- 图模式可选开启

### 验收项

- 固定 batch 场景性能提升明显
- graph 模式与 eager 模式结果一致
- 图捕获失败时可自动回退

## Stage 8：高级特性扩展

### 目标

逐步逼近成熟推理框架能力。

### 任务清单

- LoRA
- quantization
- MoE
- speculative decode
- multimodal
- prefix caching

### 输出物

- 平台高级特性矩阵

### 验收项

- 每个特性都有独立开关
- 不支持的组合能明确报错
- 已支持组合有 regression test

## Stage 9：平台工程化

### 目标

把平台变成可维护、可交付的产品级项目。

### 任务清单

- benchmark 脚本
- accuracy regression
- perf regression
- CI
- 文档
- 版本兼容说明

### 输出物

- 平台发布基线

### 验收项

- 新版本芯片 runtime 升级后可快速回归
- 新模型接入有固定 checklist
- 问题定位有明确日志与指标

## Stage 10：从插件式适配演化到独立框架

### 目标

在不推倒重来的前提下形成自己的推理栈。

### 任务清单

- 识别哪些模块已严重偏离 vLLM 原抽象
- 抽离 runtime / graph / memory manager
- 抽离 scheduler / executor
- 设计兼容层或统一 API

### 输出物

- 独立 runtime 架构设计
- 迁移边界定义

### 验收项

- 独立运行时可单独测试
- 上层服务 API 不必随底层重构大幅波动
- vLLM 兼容层与自研核心层边界清晰

---

## 28. 各阶段推荐测试策略

不同阶段的测试重点不同，建议不要一套测试打天下。

## 28.1 Stage 1-2

重点：

- 插件是否加载
- 平台是否识别
- 单卡是否能完成一次生成

建议测试：

- smoke test
- 最小模型加载测试
- 最小生成测试

## 28.2 Stage 3-4

重点：

- logits 正确性
- attention 正确性
- KV cache 正确性

建议测试：

- greedy 对齐测试
- 不同 prompt 长度测试
- block table 边界测试
- decode 多步一致性测试

## 28.3 Stage 5-6

重点：

- 多卡一致性
- 通信正确性
- 性能退化监控

建议测试：

- TP 对齐测试
- all-reduce/all-gather 单测
- 多卡长序列测试
- tokens/s 基线测试

## 28.4 Stage 7+

重点：

- graph/eager 一致性
- 高级特性组合兼容性
- regression 稳定性

建议测试：

- graph capture 回退测试
- quant + lora 组合测试
- MoE + TP/EP 测试
- nightly perf regression

---

## 29. 新芯片项目启动时的角色分工建议

如果团队人数允许，建议按职责拆分，而不是所有人都同时改所有层。

### 角色 A：平台与运行时

负责：

- `platforms/`
- `distributed/`
- `compilation/`
- runtime / stream / event / memory

### 角色 B：执行链路

负责：

- `worker/`
- `executor/`
- `model_runner/`
- KV cache 与输入整理

### 角色 C：模型与权重

负责：

- `model_executor/models/`
- `model_loader/`
- 权重映射
- TP/EP 分片加载

### 角色 D：kernel 与性能

负责：

- `csrc/`
- `_chip_ops.py`
- attention / rope / norm / moe 等热点算子

### 角色 E：验证与工程化

负责：

- `tests/`
- `benchmarks/`
- profiler
- regression 与文档

这种拆法最接近 `vllm-mlu` 的真实复杂度，也最适合新芯片平台团队协同。

---

## 30. 回到 vLLM 主仓库后，应该怎么读源码

如果目标是“参考 `vllm-mlu`，为自己的新芯片搭平台”，那么回到主仓库 `/data/users/tudj/qemu/vllm/` 后，不建议一上来就把所有目录都翻一遍。

更合理的方式是：

**沿着一次离线推理调用链，从入口一路追到平台、配置、引擎、执行、模型。**

推荐顺序：

1. `vllm/entrypoints/llm.py`
2. `vllm/engine/arg_utils.py`
3. `vllm/platforms/__init__.py`
4. `vllm/platforms/interface.py`
5. `vllm/v1/engine/llm_engine.py`
6. `vllm/v1/engine/core.py`
7. `vllm/v1/worker/`
8. `vllm/v1/attention/backends/`
9. `vllm/model_executor/model_loader/`
10. `vllm/model_executor/models/`

你可以把这 10 步理解成：

- 先看“用户怎么进来”
- 再看“配置怎么成型”
- 再看“平台怎么被识别”
- 再看“引擎怎么启动”
- 再看“模型怎么执行”

---

## 31. 第一层：`vllm/entrypoints/llm.py`

### 31.1 这一层是做什么的

这个文件是离线推理最靠近用户的入口层。  
当你写：

```python
from vllm import LLM

llm = LLM(model=..., tensor_parallel_size=..., ...)
outputs = llm.generate(...)
```

本质上就是在走这里。

### 31.2 为什么新芯片团队必须先看它

因为它回答了三个最关键的问题：

1. 用户参数从哪里进入系统
2. 参数什么时候被包装成 `EngineArgs`
3. `LLMEngine` 是什么时候被创建的

### 31.3 关键源码位置

在 `LLM.__init__()` 中，最核心的是这两步：

1. 创建 `EngineArgs`  
   见 [entrypoints/llm.py](/data/users/tudj/qemu/vllm/vllm/entrypoints/llm.py:340)

2. 创建 `LLMEngine`  
   见 [entrypoints/llm.py](/data/users/tudj/qemu/vllm/vllm/entrypoints/llm.py:381)

### 31.4 对应到新芯片接入时意味着什么

这一层通常不需要重写整个 `LLM`，但经常需要：

- 给 `LLM._run_engine()` 增加平台指标
- 给 `LLM` 增加平台专属观测接口
- 给离线 benchmark 增加统计逻辑

这也是 `vllm-mlu` 在 `vllm_mlu/entrypoints/llm.py` 中所做的事情。

### 31.5 新芯片团队在这一层该怎么做

最推荐策略：

- **第一阶段**：完全复用原始 `LLM`
- **第二阶段**：仅 patch `_run_engine()` 和少量 metric 接口
- **第三阶段**：如果要做平台 benchmark，再加平台专属方法

换句话说：

**入口层不是第一批大改对象，它主要用来理解参数与引擎的连接方式。**

---

## 32. 第二层：`vllm/engine/arg_utils.py`

### 32.1 这一层是做什么的

这是 vLLM 的配置中枢。  
用户传进来的零散参数，最后都会在这里整理成完整 `VllmConfig`。

### 32.2 为什么它是新芯片接入的第一核心层

因为几乎所有平台适配，最终都要落实成：

- 新默认值
- 新合法性约束
- 新环境变量
- 新配置项
- 新平台相关修正

这些事情最自然的落点就是 `EngineArgs.create_engine_config()` 及其周边逻辑。

### 32.3 关键源码位置

最关键位置是：

- `load_general_plugins()`  
  见 [engine/arg_utils.py](/data/users/tudj/qemu/vllm/vllm/engine/arg_utils.py:712)

- `create_engine_config()`  
  见 [engine/arg_utils.py](/data/users/tudj/qemu/vllm/vllm/engine/arg_utils.py:1594)

- `current_platform.pre_register_and_update()`  
  见 [engine/arg_utils.py](/data/users/tudj/qemu/vllm/vllm/engine/arg_utils.py:1604)

### 32.4 为什么 `pre_register_and_update()` 特别重要

这一步意味着：

- 配置正式构造之前
- 平台插件已经有机会提前做准备
- 例如：
  - 注册量化方法
  - 注册模型
  - 修补 parser / 默认值

这正是 `vllm-mlu` 在 `MLUPlatform.pre_register_and_update()` 中注册量化方法的原因。

### 32.5 对应到新芯片接入时意味着什么

新芯片团队最容易在这一层加的内容包括：

- 平台专属 `EngineArgs` patch
- 平台专属 `VllmConfig` 子配置
- attention backend 的默认选择
- graph / compile / cache / worker 默认值
- 某些不兼容组合的报错

### 32.6 新芯片团队的建议动作

应该单独建立：

- `vllm_<chip>/engine/arg_utils.py`
- `vllm_<chip>/config/model.py`
- `vllm_<chip>/config/scheduler.py`
- `vllm_<chip>/config/vllm.py`

把平台规则拆散，不要全部堆进一个大 patch 文件。

---

## 33. 第三层：`vllm/platforms/__init__.py`

### 33.1 这一层是做什么的

它负责**平台发现与平台激活**。

关键逻辑是：

- 内建平台探测
- 外部平台插件加载
- 最终决定 `current_platform`

### 33.2 关键源码位置

最关键的是：

- `load_plugins_by_group(PLATFORM_PLUGINS_GROUP)`  
  见 [platforms/__init__.py](/data/users/tudj/qemu/vllm/vllm/platforms/__init__.py:213)

- `resolve_current_platform_cls_qualname()`  
  见 [platforms/__init__.py](/data/users/tudj/qemu/vllm/vllm/platforms/__init__.py:212)

### 33.3 为什么这一层对新芯片最关键

因为你的新芯片平台能不能被 vLLM 识别，第一步就取决于这里。

换句话说，平台接入不是从 worker 开始的，而是从：

- 插件能不能被发现
- 平台类能不能被激活

开始的。

### 33.4 对应到 `vllm-mlu`

`vllm-mlu` 正是通过：

- `setup.py` 中的 `vllm.platform_plugins`
- `vllm_mlu:register_mlu_platform`

把自己的平台类插进来的。

### 33.5 新芯片团队在这一层要学到什么

你应该先把“平台注册机制”理解透，再去写任何底层执行代码。  
否则就会出现：

- 算子写好了
- worker 写好了
- 但 vLLM 根本没有走到你的平台分支

---

## 34. 第四层：`vllm/platforms/interface.py`

### 34.1 这一层是做什么的

这里定义了 `Platform` 抽象基类。  
新芯片平台类本质上都要继承它。

### 34.2 为什么这是新芯片平台最重要的抽象层

因为它定义了：

- 平台是什么
- 平台必须告诉上层什么
- 上层会向平台询问哪些能力

### 34.3 关键能力清单

`Platform` 这层主要要求平台提供：

- `device_name`
- `device_type`
- `dispatch_key`
- `dist_backend`
- `device_control_env_var`
- `supported_quantization`
- `get_device_capability()`
- `get_device_name()`
- `get_device_total_memory()`
- `set_device()`
- `empty_cache()`
- `synchronize()`
- `get_attn_backend_cls()`
- `check_and_update_config()`

### 34.4 为什么说“Platform 层是新芯片接入的第一站”

因为它处在所有后续模块之前：

- EngineArgs 需要它来修配置
- Worker 需要它来切设备
- attention selector 需要它来选 backend
- distributed 层需要它来知道通信后端

如果这一层没定义好，后面所有层都会混乱。

### 34.5 对新芯片团队的建议

一开始只实现最小接口：

- 设备名
- 设备切换
- 显存查询
- attention backend 返回值
- worker 默认类

其他复杂功能后面逐步补。

---

## 35. 第五层：参考内建平台实现，例如 `vllm/platforms/cuda.py`

### 35.1 为什么建议看 `cuda.py`

因为它是主仓库中最完整、最成熟的平台实现样板。  
虽然新芯片不一定具备 CUDA 的所有能力，但它最能展示：

- 一个成熟平台该提供哪些信息
- backend 选择如何做
- dtype 能力如何表达
- 设备能力如何参与策略选择

### 35.2 关键源码位置

例如：

- `CudaPlatformBase.device_name / device_type / dist_backend`
  见 [platforms/cuda.py](/data/users/tudj/qemu/vllm/vllm/platforms/cuda.py:158)

- `supported_dtypes`
  见 [platforms/cuda.py](/data/users/tudj/qemu/vllm/vllm/platforms/cuda.py:170)

- `set_device()`
  见 [platforms/cuda.py](/data/users/tudj/qemu/vllm/vllm/platforms/cuda.py:182)

### 35.3 新芯片团队应该借鉴什么

不要照搬 CUDA 的实现细节，但要借鉴它的设计风格：

- 平台能力要集中表达
- 与设备能力相关的逻辑要在平台层完成
- backend 选择尽量依赖平台抽象，而不是散落在模型代码里

---

## 36. 第六层：`vllm/v1/engine/llm_engine.py`

### 36.1 这一层是做什么的

`LLMEngine` 是“离线入口”和“执行核心”之间的桥。

它负责：

- 根据 `EngineArgs` 生成 `VllmConfig`
- 选择 `Executor`
- 创建 `EngineCoreClient`
- 建立输入处理器和输出处理器

### 36.2 关键源码位置

最核心的是：

- `LLMEngine.from_engine_args()`  
  见 [v1/engine/llm_engine.py](/data/users/tudj/qemu/vllm/vllm/v1/engine/llm_engine.py:151)

其中两行最关键：

- `vllm_config = engine_args.create_engine_config(...)`
- `executor_class = Executor.get_class(vllm_config)`

### 36.3 这一层对新芯片意味着什么

它说明平台接入要同时回答两个问题：

1. 配置构造完后，系统长什么样
2. 最终执行器选哪一个类

因此新芯片团队通常要在以下位置介入：

- `Executor.get_class()` 的条件分支
- 平台选择出来的 `worker_cls`
- 平台选择出来的 `executor`

### 36.4 为什么 `vllm-mlu` 会 patch engine/core/executor

因为平台光有 `Platform` 还不够。  
当执行核心：

- 需要平台专属指标
- 需要平台专属 batch 组织方式
- 需要平台专属 graph / kv cache / scheduler 协同

就必须 patch `engine` 和 `executor`。

---

## 37. 把这几层串起来理解

到这里，新的芯片团队应该形成这样一条主线：

1. 用户调用 `LLM(...)`
2. `LLM.__init__()` 把参数打包成 `EngineArgs`
3. `EngineArgs.create_engine_config()` 构造 `VllmConfig`
4. 在这个过程中，平台插件已经被加载，`current_platform` 已经确定
5. `current_platform.pre_register_and_update()` 和 `check_and_update_config()` 开始介入
6. `LLMEngine.from_engine_args()` 创建引擎
7. 引擎再去选 `Executor`、`Worker`、`ModelRunner`
8. 最终才进入模型加载和 step 执行

所以新芯片接入的最自然顺序就是：

- 先入口理解
- 再配置理解
- 再平台理解
- 再引擎理解
- 最后才去做执行和模型

这也是为什么本文前面建议你：

**不要一上来就扎进 `worker` 或 kernel，而是先把 `entrypoints -> arg_utils -> platforms -> engine` 这条主线建立起来。**

---

## 38. 基于主仓库源码的接入顺序建议

如果现在以主仓库为老师、以 `vllm-mlu` 为参考，去搭一个新芯片平台，那么建议按下面顺序真正开始开发：

### Step 1：先写平台注册

原因：

- 这是整个系统进入你平台分支的入口

对应主仓库理解点：

- `vllm/platforms/__init__.py`
- `vllm/platforms/interface.py`

### Step 2：再写配置 patch

原因：

- 这是平台策略真正落地的位置

对应主仓库理解点：

- `vllm/engine/arg_utils.py`

### Step 3：再写 worker / runner

原因：

- 这是设备执行真正发生的地方

对应主仓库理解点：

- `vllm/v1/worker/`

### Step 4：再写 attention backend 与 KV cache

原因：

- 这是性能和正确性的主战场

对应主仓库理解点：

- `vllm/v1/attention/backends/`
- `vllm/v1/core/`

### Step 5：最后写模型特化和高级功能

原因：

- 这是在最小闭环跑通之后再逐步增强

对应主仓库理解点：

- `vllm/model_executor/models/`
- `vllm/model_executor/model_loader/`

---

## 39. 主仓库源码解读时的学习目标

建议每读一层时都问自己 3 个问题：

1. **这一层在 vLLM 中负责什么？**
2. **这一层里哪些点会被新芯片平台替换或 patch？**
3. **这一层应该尽量复用，还是应该尽早平台特化？**

一个简单判断标准：

- 纯通用入口层：优先复用
- 纯配置层：适度 patch
- 平台执行层：重点接管
- kernel / graph / kv cache 层：重点自研

这样读源码，就不会迷失在大量实现细节里。

---

## 40. 第七层：`vllm/v1/engine/core.py`

### 40.1 这一层是做什么的

`EngineCore` 是 vLLM V1 执行内环的总控层。  
如果说：

- `LLM` 是离线入口
- `EngineArgs` 是配置中枢
- `Platform` 是平台声明层

那么 `EngineCore` 就是：

**真正把“配置”变成“执行系统”的地方。**

### 40.2 为什么新芯片团队必须认真读这一层

因为它负责把几个最关键的模块串起来：

- `Executor`
- KV cache 初始化
- Scheduler
- batch queue
- request 生命周期

新芯片平台如果只改平台类、不理解 `EngineCore`，通常会遇到两个问题：

1. 不知道自己的 `Worker`、`Executor`、`ModelRunner` 是什么时候被真正实例化的
2. 不知道 KV cache 和 warmup 为什么在引擎启动阶段就发生了

### 40.3 关键源码位置

最核心的初始化顺序在：

- `self.model_executor = executor_class(vllm_config)`  
  见 [v1/engine/core.py](/data/users/tudj/qemu/vllm/vllm/v1/engine/core.py:115)

- `kv_cache_config = self._initialize_kv_caches(vllm_config)`  
  见 [v1/engine/core.py](/data/users/tudj/qemu/vllm/vllm/v1/engine/core.py:126)

- `Scheduler = vllm_config.scheduler_config.get_scheduler_cls()`  
  见 [v1/engine/core.py](/data/users/tudj/qemu/vllm/vllm/v1/engine/core.py:130)

- `self.scheduler = Scheduler(...)`  
  见 [v1/engine/core.py](/data/users/tudj/qemu/vllm/vllm/v1/engine/core.py:145)

### 40.4 `EngineCore` 告诉新芯片团队什么

它说明执行链路的主分工是：

- `EngineCore`：总控
- `Executor`：执行器/worker 管理层
- `Scheduler`：排活
- `ModelRunner`：真正把张量送进模型

这意味着：

**新芯片团队真正需要重点接管的，不是入口层，而是 `Executor -> Worker -> ModelRunner` 这一段。**

### 40.5 `_initialize_kv_caches()` 为什么特别重要

`EngineCore._initialize_kv_caches()` 中，主仓库做了这些事：

1. 让 `model_executor` 报告模型需要哪些 KV cache spec
2. 让执行器先做显存 profiling
3. 根据可用显存计算 KV cache 配置
4. 把最终配置同步回系统
5. 初始化 KV cache 并 warmup 模型

关键源码位置：

- `self.model_executor.get_kv_cache_specs()`  
  见 [v1/engine/core.py](/data/users/tudj/qemu/vllm/vllm/v1/engine/core.py:235)

- `self.model_executor.determine_available_memory()`  
  见 [v1/engine/core.py](/data/users/tudj/qemu/vllm/vllm/v1/engine/core.py:249)

- `self.model_executor.initialize_from_config(kv_cache_configs)`  
  见 [v1/engine/core.py](/data/users/tudj/qemu/vllm/vllm/v1/engine/core.py:282)

### 40.6 对新芯片平台的启发

如果你的芯片要接入 vLLM，那么一定要提早设计：

- 如何做显存 profiling
- 如何决定 KV cache 大小
- 如何把模型 warmup 与图捕获接起来

不要把 KV cache 只当成一个 buffer，它其实是执行框架初始化的核心组成部分。

---

## 41. 第八层：`vllm/v1/worker/gpu_worker.py`

### 41.1 这一层是做什么的

`Worker` 是设备进程里的执行管理者。  
它负责把“平台 + 设备 + 分布式 + 模型 runner”真正接起来。

### 41.2 为什么说它是新芯片平台的第一执行落点

因为到了这一层，代码已经不再只是：

- 入口参数
- 配置对象
- 抽象接口

而是开始真正碰到：

- 设备选择
- 分布式初始化
- 显存快照
- workspace 初始化
- model runner 实例化

### 41.3 关键源码位置

最重要的是 `init_device()`：

- 设备选择  
  见 [v1/worker/gpu_worker.py](/data/users/tudj/qemu/vllm/vllm/v1/worker/gpu_worker.py:218)

- 设置当前设备  
  见 [v1/worker/gpu_worker.py](/data/users/tudj/qemu/vllm/vllm/v1/worker/gpu_worker.py:254)

- 检查 dtype  
  见 [v1/worker/gpu_worker.py](/data/users/tudj/qemu/vllm/vllm/v1/worker/gpu_worker.py:257)

- 初始化分布式环境  
  见 [v1/worker/gpu_worker.py](/data/users/tudj/qemu/vllm/vllm/v1/worker/gpu_worker.py:263)

- 显存快照与请求显存计算  
  见 [v1/worker/gpu_worker.py](/data/users/tudj/qemu/vllm/vllm/v1/worker/gpu_worker.py:281)

- 构造 `model_runner`  
  见 [v1/worker/gpu_worker.py](/data/users/tudj/qemu/vllm/vllm/v1/worker/gpu_worker.py:295)

### 41.4 为什么 `Worker` 不是简单的设备壳子

它不只是 “set_device + run_model”，而是承担了完整的设备生命周期管理：

- 初始化
- load model
- memory profiling
- KV cache 初始化
- warmup
- execute_model
- sleep/wake

所以如果新芯片团队想做平台接入，`Worker` 不是一个轻量 wrapper，而是一个真正需要定制的平台执行组件。

### 41.5 `load_model()` 为什么值得注意

主仓库里：

- `Worker.load_model()` 并不自己构造模型
- 它只是把调用下发到 `self.model_runner.load_model()`

见 [v1/worker/gpu_worker.py](/data/users/tudj/qemu/vllm/vllm/v1/worker/gpu_worker.py:318)

这说明：

- `Worker` 负责设备上下文
- `ModelRunner` 负责模型本体

这个边界对新芯片平台特别重要，建议保留。

### 41.6 新芯片团队在这一层应该怎么做

第一阶段建议：

- 直接仿照 `gpu_worker.py` 复制出 `mychip_worker.py`
- 先实现最小 `init_device()`
- 再实现最小 `load_model()`
- 再实现 `determine_available_memory()`
- 最后实现 `execute_model()`

不要在第一版就把所有 sleep mode / profiler / elastic EP 都搬过去。

---

## 42. 第九层：`vllm/v1/worker/gpu_model_runner.py`

### 42.1 这一层是做什么的

这是执行链里最重的一层。  
如果说 `Worker` 是设备进程总管，那么 `GPUModelRunner` 就是：

**真正把调度结果整理成模型输入，并调用模型 forward 的地方。**

### 42.2 为什么新芯片平台最终几乎一定会重度定制这一层

因为这层直接接触：

- attention metadata
- input batch
- KV cache layout
- sampler
- speculative decoding
- multimodal encoder
- graph capture
- async output copy

也就是说，芯片最敏感的性能路径都在这里。

### 42.3 关键源码位置

最先要看的位置：

- `class GPUModelRunner`
  见 [v1/worker/gpu_model_runner.py](/data/users/tudj/qemu/vllm/vllm/v1/worker/gpu_model_runner.py:394)

- `__init__()`
  见 [v1/worker/gpu_model_runner.py](/data/users/tudj/qemu/vllm/vllm/v1/worker/gpu_model_runner.py:397)

- `execute_model()`
  见 [v1/worker/gpu_model_runner.py](/data/users/tudj/qemu/vllm/vllm/v1/worker/gpu_model_runner.py:3788)

- `load_model()`
  见 [v1/worker/gpu_model_runner.py](/data/users/tudj/qemu/vllm/vllm/v1/worker/gpu_model_runner.py:4764)

- `initialize_kv_cache()`
  见 [v1/worker/gpu_model_runner.py](/data/users/tudj/qemu/vllm/vllm/v1/worker/gpu_model_runner.py:6762)

### 42.4 `__init__()` 告诉新芯片团队什么

在 `GPUModelRunner.__init__()` 中，主仓库做了大量“运行前准备”：

- 保存各种 config
- 确定 dtype 和 kv cache dtype
- 初始化 sampler
- 初始化 speculative decode 结构
- 初始化 multimodal 相关状态
- 准备 buffer 和输入批结构

这说明：

**ModelRunner 不是“只有一个 execute_model 函数”，而是平台运行时的中心态对象。**

### 42.5 `load_model()` 为什么是模型接入的关键

这一层会调用：

- `get_model_loader(self.load_config)`

见 [v1/worker/gpu_model_runner.py](/data/users/tudj/qemu/vllm/vllm/v1/worker/gpu_model_runner.py:4784)

这意味着：

- 模型类如何被解析
- 模型权重如何被加载
- 模型如何包裹 LoRA / graph / quant / draft model

都在 runner 这层真正汇总。

### 42.6 `execute_model()` 为什么是芯片价值主战场

因为这里最终决定了：

- 本轮请求如何整理成 batch
- attention metadata 如何构造
- KV cache 如何绑定
- forward 怎么被调用
- logits/sampling 如何输出

对新芯片平台来说，这一层往往是：

- 最需要 patch 的层
- 最容易产生平台分叉的层
- 最能体现芯片性能优势的层

### 42.7 新芯片团队在这一层的建议策略

不要试图首版完全兼容 `GPUModelRunner` 的所有能力。  
更可行的策略是：

#### 第一版

- 单模型
- 单卡
- 基础 prefill/decode
- 基础 logits 输出

#### 第二版

- KV cache
- 基础 paged attention
- 基础 sampler

#### 第三版

- graph capture
- async output
- multimodal
- spec decode

---

## 43. 执行核心三层的职责边界

为了避免代码设计混乱，强烈建议新芯片团队从一开始就固定下面这组边界。

### `EngineCore`

职责：

- 串起 Executor、KV cache、Scheduler
- 做总控
- 不直接写设备特化逻辑

### `Worker`

职责：

- 管设备
- 管分布式
- 管显存 profiling
- 管 model runner 生命周期

### `ModelRunner`

职责：

- 管模型
- 管输入张量准备
- 管 attention metadata
- 管 KV cache 绑定
- 管 forward 输出与采样

如果这三层职责混在一起，后面平台一复杂就会很难维护。

---

## 44. 为什么说“新芯片真正该接管的是执行核心”

到这里可以给出一个明确结论：

### 不必优先重写的层

- `LLM`
- `entrypoints`
- 通用 tokenizer / renderer
- 大部分上层 request/output 协议

这些层主要是通用产品壳。

### 必须重点接管的层

- `Platform`
- `Executor`
- `Worker`
- `ModelRunner`
- attention backend
- KV cache
- custom ops
- graph / runtime / communicator

原因很简单：

**芯片平台的差异，真正都发生在执行核心，而不是用户 API 层。**

这也是 `vllm-mlu` 的最重要启发之一。

---

## 45. 从主仓库继续往下读时，该怎么看

现在如果继续对照主仓库阅读，建议下一步按这个顺序走：

1. `vllm/v1/attention/backends/`
2. `vllm/model_executor/model_loader/`
3. `vllm/model_executor/models/`

对应的问题分别是：

1. 底层 attention 执行抽象是怎么做的？
2. 模型类是怎么被找到并加载的？
3. 具体模型结构是怎么定义的？

也就是说：

- 前面几层帮你理解“框架怎么跑起来”
- 后面几层帮你理解“模型怎么被塞进去、attention 怎么真正执行”

这正好和 `vllm-mlu` 的关键改动区域一一对应。

---

## 46. 第十层：`vllm/v1/attention/backend.py`

### 46.1 这一层是做什么的

这是 vLLM 对 attention backend 的统一抽象定义层。

简单说：

- `Platform` 决定“选哪个 backend”
- `AttentionBackend` 定义“一个 backend 必须提供哪些能力”

### 46.2 为什么新芯片团队必须理解这一层

因为很多人第一次看 `vllm-mlu` 会误以为：

- attention backend 只是一个 kernel 文件

实际上不是。  
在 vLLM 里，attention backend 至少要解决这些问题：

- 支持哪些 dtype
- 支持哪些 kv cache dtype
- 支持哪些 block size
- KV cache 的 shape 是什么
- metadata builder 是谁
- 真正 impl 类是谁
- 是否支持 MLA / sparse / sink / non-causal 等特性

### 46.3 关键源码位置

核心抽象类是：

- `class AttentionBackend`
  见 [v1/attention/backend.py](/data/users/tudj/qemu/vllm/vllm/v1/attention/backend.py:55)

这里最关键的方法有：

- `get_name()`
- `get_impl_cls()`
- `get_builder_cls()`
- `get_kv_cache_shape()`
- `validate_configuration()`

### 46.4 对新芯片平台意味着什么

这说明你不能把 attention 设计成只有一个：

```python
mychip_attention(q, k, v)
```

而必须回答：

- 这个 backend 在哪些配置组合下有效
- KV cache 在内存中长什么样
- metadata 如何从 scheduler / runner 输入中构造

### 46.5 这是 `vllm-mlu` 的哪一层

对应到 `vllm-mlu`，就是：

- `vllm_mlu/v1/attention/backends/flash_attn.py`
- `vllm_mlu/v1/attention/backends/mla/flashmla.py`

也就是平台专属 backend 实现层。

---

## 47. 第十一层：`vllm/v1/attention/backends/registry.py`

### 47.1 这一层是做什么的

这是 attention backend 的注册表与枚举层。

它回答的问题是：

- 系统里有哪些 backend
- backend 对应哪个类路径
- 是否允许被 override

### 47.2 关键源码位置

核心内容有：

- `AttentionBackendEnum`
  见 [v1/attention/backends/registry.py](/data/users/tudj/qemu/vllm/vllm/v1/attention/backends/registry.py:34)

- `register_backend(...)`
  见 [v1/attention/backends/registry.py](/data/users/tudj/qemu/vllm/vllm/v1/attention/backends/registry.py:211)

### 47.3 对新芯片平台的启发

这意味着新芯片平台有两种常见接入方式：

1. **平台类直接返回新的 backend 类路径**
2. **通过 registry 机制覆盖现有 backend 或注册 CUSTOM backend**

### 47.4 推荐做法

对新芯片平台来说，更常见也更清晰的做法是：

- 在 `Platform.get_attn_backend_cls()` 里明确返回自己的 backend

只有在你想复用 registry 的动态替换能力时，才需要进一步做 override。

### 47.5 为什么这一层重要

它把“平台选 backend”和“backend 真正类实现”解耦了。  
这正是 vLLM 能同时支持 CUDA / ROCm / XPU / OOT 平台的关键原因之一。

---

## 48. 第十二层：`vllm/model_executor/model_loader/__init__.py`

### 48.1 这一层是做什么的

这是模型加载方式选择层。

注意，这一层解决的不是“选哪个模型类”，而是：

**选哪一种加载器去加载模型。**

比如：

- 默认 HF 格式
- GGUF
- tensorizer
- sharded state
- dummy loader

### 48.2 关键源码位置

最关键的是：

- `_LOAD_FORMAT_TO_MODEL_LOADER`
  见 [model_loader/__init__.py](/data/users/tudj/qemu/vllm/vllm/model_executor/model_loader/__init__.py:48)

- `get_model_loader(load_config)`
  见 [model_loader/__init__.py](/data/users/tudj/qemu/vllm/vllm/model_executor/model_loader/__init__.py:120)

- `get_model(...)`
  见 [model_loader/__init__.py](/data/users/tudj/qemu/vllm/vllm/model_executor/model_loader/__init__.py:128)

### 48.3 对新芯片团队意味着什么

这说明“模型加载”至少分成两层：

1. **选加载器**
2. **加载器内部再去实例化模型类并加载权重**

如果新芯片只支持特殊权重格式，比如：

- 自己的 tensorized 格式
- 自己的压缩格式
- 自己的量化 checkpoint

那么最自然的切入点，就是新增自己的 `ModelLoader`。

### 48.4 对应到 `vllm-mlu`

`vllm-mlu` 已经在这条线上做过 patch，例如：

- `vllm_mlu/model_executor/model_loader/tensorizer_loader.py`
- `vllm_mlu/model_executor/model_loader/dummy_loader.py`

说明平台适配不仅仅是算子问题，也包括权重格式问题。

---

## 49. 第十三层：`vllm/model_executor/model_loader/utils.py`

### 49.1 这一层是做什么的

这是模型类解析与初始化的桥梁层。

它负责把：

- `ModelConfig`
- HuggingFace config 中的 `architectures`

最终变成：

- 一个具体的 Python 模型类

### 49.2 关键源码位置

最关键的方法有：

- `initialize_model(...)`
  见 [model_loader/utils.py](/data/users/tudj/qemu/vllm/vllm/model_executor/model_loader/utils.py:35)

- `_get_model_architecture(...)`
  见 [model_loader/utils.py](/data/users/tudj/qemu/vllm/vllm/model_executor/model_loader/utils.py:175)

- `get_model_architecture(...)`
  见 [model_loader/utils.py](/data/users/tudj/qemu/vllm/vllm/model_executor/model_loader/utils.py:210)

### 49.3 这里最重要的一句代码

核心桥接点是：

```python
model_cls, arch = model_config.registry.resolve_model_cls(
    architectures,
    model_config=model_config,
)
```

见 [model_loader/utils.py](/data/users/tudj/qemu/vllm/vllm/model_executor/model_loader/utils.py:180)

这句的意义非常大：

- HuggingFace config 提供架构名
- registry 根据架构名解析模型类
- 最终决定实例化哪个 `nn.Module`

### 49.4 `initialize_model()` 对新芯片有什么启发

`initialize_model()` 还说明一件很关键的事：

- 新风格模型类应该接受 `vllm_config` 和 `prefix`

见 [model_loader/utils.py](/data/users/tudj/qemu/vllm/vllm/model_executor/model_loader/utils.py:52)

这也是为什么你在 `vllm-mlu` 的模型类里总看到：

```python
def __init__(self, *, vllm_config: VllmConfig, prefix: str = "")
```

对新芯片平台来说，这个接口风格最好沿用，不要自创一套。

---

## 50. 第十四层：`vllm/model_executor/models/registry.py`

### 50.1 这一层是做什么的

这是模型注册表。  
它的职责是：

- 维护“架构名 -> 模型类”的映射
- 在运行时解析模型类
- 允许 out-of-tree 模型覆盖或注册

### 50.2 为什么新芯片团队必须看这一层

因为很多平台特化模型，并不是直接改 loader，而是：

- 注册一个新的模型类
- 用相同 architecture 名覆盖默认实现

### 50.3 关键源码位置

关键点有：

- `_ModelRegistry`
  见 [models/registry.py](/data/users/tudj/qemu/vllm/vllm/model_executor/models/registry.py:915)

- `register_model(...)`
  见 [models/registry.py](/data/users/tudj/qemu/vllm/vllm/model_executor/models/registry.py:922)

- `resolve_model_cls(...)`
  见 [models/registry.py](/data/users/tudj/qemu/vllm/vllm/model_executor/models/registry.py:1159)

### 50.4 对应到 `vllm-mlu`

`vllm-mlu` 正是通过这一层，注册了：

- `DeepseekV4ForCausalLM -> MLUDeepseekV4ForCausalLM`

也就是说：

- 对外架构名不变
- 对内实现类切成 MLU 版本

### 50.5 对新芯片团队意味着什么

如果某个模型：

- attention 结构特殊
- KV cache 特殊
- MoE 路由特殊
- 需要大量调用芯片自定义 op

那最自然的做法往往不是 patch 原模型，而是：

- 直接注册一个平台专属模型类

---

## 51. 把三层串起来：loader、registry、model class 的关系

这一段是新芯片团队最容易混淆的地方，必须单独说明。

### 51.1 三层分工

#### `model_loader/__init__.py`

负责：

- 选哪种加载器

#### `model_loader/utils.py`

负责：

- 根据 `ModelConfig` 和架构名找到模型类
- 初始化模型实例

#### `models/registry.py`

负责：

- 管理“架构名 -> 模型类”的映射关系

### 51.2 真实调用链

从 `GPUModelRunner.load_model()` 往下看，调用链可以理解成：

1. `get_model_loader(load_config)`
2. `loader.load_model(...)`
3. `get_model_architecture(model_config)`
4. `model_config.registry.resolve_model_cls(...)`
5. `initialize_model(...)`
6. 实例化具体模型类
7. 模型类自己的 `load_weights(...)`

### 51.3 新芯片团队最常见的两类改法

#### 改法 A：新增 loader

适合：

- 新权重格式
- 新 checkpoint 组织方式
- 新量化文件格式

#### 改法 B：新增模型类并注册

适合：

- 新 attention 结构
- 新 KV cache 结构
- 新芯片算子深度嵌入模型

很多情况下，两种都会做。

---

## 52. 为什么 `vllm-mlu` 会在 model executor 层大量改动

到这里你就能理解 `vllm-mlu` 为什么不只改：

- `platforms/mlu.py`

还要改：

- `model_executor/model_loader/`
- `model_executor/models/`
- `model_executor/layers/`

因为平台真正落地时，至少会遇到以下问题：

- 某些模型必须重写 attention 层
- 某些权重格式需要特殊处理
- 某些 fused 算子需要在模型层显式接入
- 某些模型类必须替换成平台专属版本

所以：

**平台适配最终一定会进入 model executor 层。**

---

## 53. 到这里为止，主仓库源码主线已经完整

现在已经可以把主仓库主线完整串起来：

1. `entrypoints/llm.py`
2. `engine/arg_utils.py`
3. `platforms/`
4. `v1/engine/llm_engine.py`
5. `v1/engine/core.py`
6. `v1/worker/`
7. `v1/attention/backend.py`
8. `v1/attention/backends/registry.py`
9. `model_executor/model_loader/`
10. `model_executor/models/registry.py`

这条主线分别回答了：

1. 用户怎么进来
2. 配置怎么形成
3. 平台怎么被识别
4. 引擎怎么创建
5. 执行核心怎么组装
6. 设备侧怎么执行
7. attention backend 长什么样
8. backend 怎么被注册和选择
9. 模型怎么被加载
10. 模型类怎么被解析和替换

这正好覆盖了新芯片平台从 0 接入 vLLM 的最重要骨架。

---

## 54. 下一步如何继续对照主仓库学习

如果还要继续深入，建议有两条路线：

### 路线 A：继续抽象层阅读

继续看：

- `vllm/v1/core/sched/`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/core/kv_cache_manager.py`

适合想搞懂：

- 调度
- KV cache 管理
- 请求生命周期

### 路线 B：继续模型与算子层阅读

继续看：

- `vllm/model_executor/layers/attention.py`
- 某个具体模型实现
- 某个具体 backend 实现

适合想搞懂：

- 具体一层 attention 是怎么接到 backend 上的
- 模型里哪些逻辑属于结构层，哪些属于 backend 层

对新芯片团队来说，通常建议先走路线 A，再走路线 B。  
因为没有调度和 KV cache 的整体观，后面看 attention 容易只见树木不见森林。

---

## 55. 第十五层：`vllm/v1/core/sched/scheduler.py`

### 55.1 这一层是做什么的

`Scheduler` 是请求调度总管。  
它负责决定：

- 哪些请求进入 running
- 哪些请求继续 waiting
- 这一轮每个请求分到多少 token
- 哪些请求需要 preempt
- 哪些请求需要释放资源

### 55.2 为什么新芯片团队必须认真看这一层

因为真正的推理框架不是“模型会 forward 就够了”，而是：

- 多请求并发
- prefill/decode 混合
- KV cache 受限
- 不同请求长度不同
- 不同特性组合不同

这时瓶颈往往不只是算子，而是**调度策略和资源编排**。

### 55.3 关键源码位置

初始化里最重要的内容有：

- waiting / running 队列初始化  
  见 [v1/core/sched/scheduler.py](/data/users/tudj/qemu/vllm/vllm/v1/core/sched/scheduler.py:166)

- `KVConnector` 初始化  
  见 [v1/core/sched/scheduler.py](/data/users/tudj/qemu/vllm/vllm/v1/core/sched/scheduler.py:123)

- `KVCacheManager` 初始化  
  见 [v1/core/sched/scheduler.py](/data/users/tudj/qemu/vllm/vllm/v1/core/sched/scheduler.py:224)

### 55.4 `Scheduler` 为什么不只是“排队”

它同时管理：

- `requests`
- `waiting`
- `running`
- `finished_req_ids`
- `encoder_cache_manager`
- `kv_cache_manager`
- `connector`

这说明调度器在 vLLM 中其实是一个**资源编排中心**，而不只是一个 FIFO 队列。

### 55.5 对新芯片平台意味着什么

新芯片团队通常最开始不需要重写整个 scheduler，但必须理解：

- scheduler 的每个决定，都会影响：
  - batch shape
  - graph capture 命中率
  - KV cache 碎片率
  - tokens/s

### 55.6 对应到 `vllm-mlu`

`vllm-mlu` 在这层做了典型平台改动：

- 自定义 chunk / unchunk scheduler
- async scheduler
- scheduler profiler

这说明一旦芯片平台有特殊执行节奏，scheduler 很快就会成为平台分叉点。

---

## 56. 新芯片团队如何理解 scheduler 的接入边界

推荐按阶段理解：

### 第一阶段

完全复用主仓库 scheduler。

适用条件：

- 你的芯片只是换了 device / kernel
- 请求编排逻辑还不需要特殊处理

### 第二阶段

只 patch scheduler 的少量行为。

适用条件：

- 需要改 chunked prefill 默认值
- 需要加 profiler
- 需要处理某些平台兼容约束

### 第三阶段

实现自己的 scheduler 子类。

适用条件：

- 芯片有特殊 graph 约束
- 芯片有特殊 cache 策略
- 芯片有特殊 prefill/decode pipeline
- 芯片有特殊 sequence parallel 或 disagg 流程

换句话说：

**调度器不一定是第一批改造对象，但通常会成为第二批或第三批关键改造对象。**

---

## 57. 第十六层：`vllm/v1/core/kv_cache_utils.py`

### 57.1 这一层是做什么的

这是 KV cache 的基础数据结构和工具层。

它负责定义：

- block hash
- block 元数据
- free block queue
- prefix cache 相关工具

### 57.2 为什么它对新芯片平台特别重要

因为很多新芯片团队一开始会把 KV cache 想成：

- 一个简单的大 tensor

但在 vLLM 里不是。  
KV cache 是被分块、被引用计数、被缓存、被复用、被回收的。

### 57.3 关键源码位置

例如：

- `KVCacheBlock`
  见 [v1/core/kv_cache_utils.py](/data/users/tudj/qemu/vllm/vllm/v1/core/kv_cache_utils.py:109)

- `FreeKVCacheBlockQueue`
  见 [v1/core/kv_cache_utils.py](/data/users/tudj/qemu/vllm/vllm/v1/core/kv_cache_utils.py:158)

### 57.4 为什么这些结构有意义

它们说明：

- block 有生命周期
- block 需要 O(1) 插入、删除、复用
- block 不只是存数据，还带元信息
- prefix cache 命中依赖 block hash

### 57.5 对新芯片平台的启发

如果你的芯片有特殊内存布局，比如：

- 大页内存
- 分级 cache
- 硬件压缩 cache
- 不同 head/group 需要不同布局

那你最终很可能要深入这层甚至重写这层的一部分逻辑。

---

## 58. 第十七层：`vllm/v1/core/kv_cache_manager.py`

### 58.1 这一层是做什么的

这是 KV cache 的管理层。  
它向 scheduler 提供的是一个更高层的接口：

- 某请求已有多少缓存块
- 某请求还能不能完整放下
- 某请求这一步应该分配哪些 block

### 58.2 关键源码位置

最关键的类是：

- `KVCacheManager`
  见 [v1/core/kv_cache_manager.py](/data/users/tudj/qemu/vllm/vllm/v1/core/kv_cache_manager.py:106)

里面几个特别重要的方法：

- `get_computed_blocks(...)`
  见 [v1/core/kv_cache_manager.py](/data/users/tudj/qemu/vllm/vllm/v1/core/kv_cache_manager.py:176)

- `can_fit_full_sequence(...)`
  见 [v1/core/kv_cache_manager.py](/data/users/tudj/qemu/vllm/vllm/v1/core/kv_cache_manager.py:218)

- `allocate_slots(...)`
  见 [v1/core/kv_cache_manager.py](/data/users/tudj/qemu/vllm/vllm/v1/core/kv_cache_manager.py:257)

### 58.3 为什么说它是 Scheduler 和底层 block 系统之间的桥

因为：

- Scheduler 关心的是 request 和 token
- Block pool 关心的是 block 和内存

`KVCacheManager` 正好做这层转换。

### 58.4 `allocate_slots()` 为什么是关键

这个方法本质上在回答：

- 当前请求这一轮需要多少新 token
- 有多少 token 已经在 cache 里
- 有多少 token 来自外部 connector
- 需要额外分配多少 block

也就是说，它是 **“调度 token” 到 “分配内存块”** 的核心桥梁。

### 58.5 对新芯片平台意味着什么

如果你的芯片：

- block size 有特殊要求
- prefix cache 代价很高
- cache 写入和读取路径不对称
- 图模式要求 block 对齐
- Mamba / MLA / sliding window 需要特殊块布局

那么这层迟早会成为平台特化重点。

这也是为什么 `vllm-mlu` 会在 KV cache 相关层大量 patch。

---

## 59. Scheduler 与 KV cache 的关系

这一点对新芯片团队特别关键，单独讲清楚。

### 59.1 不是两个独立系统

很多初学者会误以为：

- scheduler 只管请求
- KV cache 只管内存

实际上在 vLLM 里，这两者高度耦合：

- scheduler 决定这轮每个请求处理多少 token
- 这个决定会直接影响需要分配多少 KV block
- KV cache 是否足够，又反过来限制 scheduler 能调多少 token

### 59.2 这意味着什么

意味着你不能单独“优化 scheduler”或单独“优化 KV cache”，  
而必须从两者联动的角度设计平台策略。

例如：

- 一个芯片如果 graph mode 强依赖固定 batch
- 那 scheduler 就要更倾向稳定 batch 形状
- KV cache block size 也要配合图捕获策略

这正是平台优化真正困难的地方。

---

## 60. 对照 `vllm-mlu` 时，为什么 Scheduler/KV 线必须重点看

因为 `vllm-mlu` 的很多平台特化都不是单个算子层面，而是：

- scheduler 默认策略变化
- chunk / unchunk 分支
- benchmark 例外逻辑
- KV cache block 配置变化
- MLA / Mamba / prefix cache 约束
- graph capture 条件变化

这类改动如果只看 `worker` 或 `model_runner`，你会觉得很散。  
但如果把它们放回 `Scheduler + KVCacheManager` 这条主线，就会发现它们都在回答同一个问题：

**“在这个芯片平台上，一轮请求应该怎样被切成可执行批次，并如何落到可承受的 KV cache 布局上？”**

---

## 61. 新芯片团队在调度与 KV cache 层的推荐策略

### 阶段 1：先复用

建议：

- 先尽量复用主仓库 scheduler 与 kv cache manager

前提：

- block size 没有极端特殊要求
- prefix cache 不是强依赖
- graph 约束还不复杂

### 阶段 2：再加平台约束

建议先 patch：

- 默认 `block_size`
- `max_num_batched_tokens`
- chunked prefill 默认值
- graph capture batch 列表

### 阶段 3：最后做调度器分叉

只有在以下情况时才建议实现平台专属 scheduler：

- 芯片必须固定某些 batch 形状
- cache 策略与主仓库差异极大
- prefill/decode 物理执行路径完全不同
- disaggregated 方案与主仓库通用逻辑不兼容

这个顺序和 `vllm-mlu` 的演进方式也是一致的。

---

## 62. 到这里，主仓库执行主线已经真正闭环

现在已经可以把“从用户请求到模型执行”的整条链闭环写出来：

1. `entrypoints/llm.py`  
   用户调用 `LLM(...)`

2. `engine/arg_utils.py`  
   参数被整理成 `VllmConfig`

3. `platforms/`  
   当前平台被检测并激活

4. `v1/engine/llm_engine.py`  
   `LLMEngine` 被创建

5. `v1/engine/core.py`  
   `Executor`、KV cache、Scheduler 被真正装配

6. `v1/worker/`  
   设备、分布式、ModelRunner 被初始化

7. `v1/attention/backends/`  
   backend 定义 attention 执行抽象

8. `model_loader/`  
   loader 负责模型实例化和权重加载路径

9. `models/registry.py`  
   registry 负责模型类解析

10. `core/sched/ + kv_cache_*`  
    scheduler 与 KV cache manager 共同决定每轮请求如何落成可执行 batch

这条主线已经足够支撑一个新芯片团队系统地拆解 vLLM 与 `vllm-mlu`。

---

## 63. 下一步继续深入时的推荐方向

现在如果还要继续往下深挖，建议优先看下面两条。

### 方向 A：具体 attention 调用链

重点看：

- 某个具体 backend 实现
- 某个具体模型的 attention 层
- `model_executor/layers/attention.py`

适合回答：

- 模型里的 attention 层是怎么接到 backend 上的
- 哪些逻辑在模型层，哪些逻辑在 backend 层

### 方向 B：具体模型实例

重点看：

- 一个具体模型类
- 它的 `load_weights()`
- 它的 `forward()`

适合回答：

- 为什么平台特化模型要重写
- 权重映射和 fused 参数是怎么处理的

对新芯片团队来说，推荐 **先走 A，再走 B**。  
因为 attention 往往是芯片价值最集中的地方，而模型类通常是在 attention 和 KV cache 逻辑确定后再做稳定化。

---

## 64. 第十八层：模型层 attention 抽象从哪里开始

如果现在回到主仓库，想真正看清“模型里的 attention 层”和“backend 层”是怎么接起来的，第一站不要直接看某个 backend 文件，而要先看：

- `vllm/model_executor/layers/attention_layer_base.py`

这个文件的价值不在于逻辑多，而在于它把 v1 engine 眼里的“可被统一调度的 attention-like layer”抽象出来了。

核心接口只有两个：

- `get_attn_backend()`
- `get_kv_cache_spec()`

这两个接口已经把职责边界说得很清楚了：

- 模型层要告诉引擎“我这一层最终用哪个 attention backend”
- 模型层还要告诉引擎“我这一层需要怎样的 KV cache 规格”

也就是说，从 v1 engine 的视角看，模型里的 attention 层不是一个纯 PyTorch 子模块，而是一个要向执行框架暴露 backend 与 cache 约束的执行单元。

这一步很关键，因为它说明：

**模型层 attention 和 backend 不是平行关系，而是模型层通过统一接口把 backend 暴露给引擎。**

---

## 65. 第十九层：`Attention` 层如何真正选中 backend

接下来要看：

- `vllm/model_executor/layers/attention/attention.py`

主仓库的 `Attention` 类既是模型里的一个层，也是 `AttentionLayerBase` 的具体实现。

这个类在概念上做三件事：

1. 管理这一层自己的 KV cache
2. 决定本层用哪个 backend
3. 创建 backend 对应的 `impl`

最关键的桥接代码在初始化阶段：

```python
if attn_backend is None:
    self.attn_backend = get_attn_backend(...)
else:
    self.attn_backend = attn_backend
```

然后继续：

```python
impl_cls = self.attn_backend.get_impl_cls()
self.impl = impl_cls(...)
```

这两步合起来的意思是：

- 如果模型层没有手工指定 backend
- 就根据当前运行配置、dtype、head size、kv cache dtype、是否 MLA 等条件
- 调 `get_attn_backend(...)` 自动选一个 backend class
- 再从这个 backend class 取出真正执行 attention 的 `impl`

也就是说，模型层 attention 并不直接决定“用哪个 kernel 函数名”，它先决定“用哪类 backend”，再由 backend 决定“底层 impl 和 metadata builder”。

这里也能看出一个很重要的工程分层：

- `Attention` 类负责“我是一个 attention 层”
- backend class 负责“我支持哪些组合、我的 KV cache 怎么摆、我的 metadata 怎么构建”
- impl class 负责“真正执行一次 attention”

---

## 66. 第二十层：以 `FlashAttentionBackend` 为例看 backend 的真实职责

现在再看：

- `vllm/v1/attention/backends/flash_attn.py`

这个文件最适合当样板，因为它把 backend 该承担的职责写得很完整。

`FlashAttentionBackend` 至少承担了下面几类职责。

### 66.1 能力声明

比如：

- 支持哪些 `dtype`
- 支持哪些 `kv_cache_dtype`
- 支持哪些 `attn_type`
- 支持哪些 `head_size`
- 需要什么 `compute capability`

这说明 backend 不是“一个 kernel 包装器”那么简单，它首先是一个能力过滤器。

也就是在真正创建 impl 之前，先判断：

**当前模型形状和当前设备能力，能不能用这套 backend。**

### 66.2 KV cache 形状与布局声明

比如：

- `get_kv_cache_shape(...)`
- `get_kv_cache_stride_order(...)`

这说明 backend 还在定义：

- KV cache 应该长什么 shape
- 内存布局应该怎样排

这一步非常关键。  
因为对于推理框架来说，attention backend 并不只是一个计算算子，它还决定 cache layout，而 cache layout 又会反过来决定：

- block manager 怎么分配
- model runner 怎么写 cache
- kernel 怎么读 cache

### 66.3 impl 与 metadata builder 选择

比如：

- `get_impl_cls()`
- `get_builder_cls()`

这两个接口非常重要。

`get_impl_cls()` 说明：

- 真正执行 attention 的逻辑在哪个 impl 类里

`get_builder_cls()` 说明：

- 本 backend 需要的 metadata 由哪个 builder 来构造

这也就是 backend 层最典型的职责：

**不是直接算 attention，而是规定“用哪个执行实现 + 用哪套 metadata 构造规则”。**

### 66.4 backend 自己的 metadata 协议

同一个文件里还有：

- `FlashAttentionMetadata`
- `FlashAttentionMetadataBuilder`

这说明 backend 不只是暴露一个 impl，还定义了自己完整的 metadata 协议。

builder 负责根据本轮 batch 的公共信息构造：

- `query_start_loc`
- `seq_lens`
- `block_table`
- `slot_mapping`
- cascade / prefix 相关字段
- cudagraph / scheduler metadata

也就是说，scheduler 和 model runner 在上游组织出的“批次信息”，到了 backend 这里还要被翻译成该 backend 真正能吃的 metadata 格式。

所以 backend 是一层非常实的“执行协议层”，不是一个薄封装。

---

## 67. 模型层 attention 与 backend 层的职责边界

到这里就可以把两层边界明确写出来。

### 模型层 attention 更负责什么

模型层更偏向“网络结构语义”：

- q/k/v/o 投影怎么定义
- RoPE、ALiBi、局部窗口这些模型语义怎么接
- 某些模型特有的 sparse / compress / MLA / sink 逻辑
- 本层的 forward 输入输出怎么组织
- 这一层暴露给引擎的 KV cache 规格是什么

换句话说，模型层更接近“这一层在模型结构里怎么成立”。

### backend 层更负责什么

backend 层更偏向“执行协议与硬件实现”：

- 这个组合是否支持
- KV cache 具体 shape 与 stride order
- metadata 怎么构造
- 该选哪个 impl
- graph mode 能否支持
- 特定 kernel 对 block size、dtype、device capability 的限制

换句话说，backend 更接近“这一层最后怎么在当前平台上高效执行”。

### 两者怎么衔接

最准确的一句话是：

**模型层负责定义 attention 语义，backend 层负责把这个语义落成平台可执行协议。**

这也是为什么主仓库里要同时存在：

- `model_executor/layers/attention/...`
- `v1/attention/backends/...`

它们不是重复，而是分层。

---

## 68. 为什么 `vllm-mlu` 会同时改模型 attention 和 backend

理解了主仓库这套分层，再回看 `vllm-mlu`，很多看起来“重复”的改动就不奇怪了。

`vllm-mlu` 往往会同时动两类地方：

- `model_executor/models/...` 里的具体模型 attention
- `v1/attention/backends/...` 里的 backend

原因是这两层解决的问题不同。

### 改模型 attention，是因为模型语义变了

例如某些模型会有：

- MLA
- compress / window / sparse memory
- 特殊 q/kv 组织方式
- 特殊 residual / rope / output transform

这类逻辑如果主仓库通用 attention 层表达不了，就必须在模型类里重写。

### 改 backend，是因为硬件执行协议变了

例如某个新芯片可能有：

- 自己的 paged KV layout
- 自己的 block 粒度要求
- 自己的 graph capture 限制
- 自己的 metadata 输入格式
- 自己的 fused attention kernel

这类逻辑就应该放在 backend 层。

所以 `vllm-mlu` 同时改两层，不是架构乱，而是说明：

- 一部分问题属于模型语义层
- 一部分问题属于后端执行层

新芯片团队如果把这两类问题混在一个文件里，后面一定会变得难维护。

---

## 69. 新芯片团队在 attention 这条线的推荐策略

如果你是从 0 做一款新芯片平台，建议按下面顺序推进。

### 阶段 1：优先复用模型层 attention

先尽量不碰模型层大改，只做：

- 平台选择逻辑
- backend class
- impl
- metadata builder

前提是：

- 你的芯片可以接受主仓库现有 attention 层组织出来的 q/k/v 与 cache 语义
- 你只是换底层 kernel 和 cache layout

这一步的目标是先让主仓库现有模型尽快在新芯片上跑起来。

### 阶段 2：实现自己的 backend 执行协议

这是最常见、也是最值得最先投入的平台层工作：

- 定义 backend class
- 定义 metadata dataclass
- 定义 metadata builder
- 定义 impl
- 定义 KV cache shape 与 stride order
- 在平台层把 backend 接入选择逻辑

如果只允许做一层深度定制，优先做这一层。

### 阶段 3：只在必要时重写具体模型 attention

只有当下列情况出现时，再去改模型层：

- 某模型不是标准 q/k/v/o attention
- 某模型有 MLA / sparse / compress / mixed cache 语义
- 某模型必须直接调用专用自定义算子
- 某模型对 cache 写入与读取时机有特殊要求

这时候就要像 `vllm-mlu` 那样，在具体模型类里单独定制 attention 逻辑。

这个顺序的好处是：

- 先把通用平台能力做出来
- 再把模型特化逐步加进去
- 不会一开始就陷入“每个模型都要魔改一遍”的失控状态

---

## 70. 现在这条源码主线已经可以再闭一次环

如果把这次补充的 attention 分层也加进去，那么“新芯片如何参考 `vllm-mlu` 接入 vLLM”的源码闭环就更完整了：

1. `entrypoints/llm.py`  
   用户入口

2. `engine/arg_utils.py`  
   参数整理与配置生成

3. `platforms/`  
   平台识别、能力声明、配置修正

4. `v1/engine/llm_engine.py`  
   引擎外层桥梁

5. `v1/engine/core.py`  
   executor、scheduler、kv cache 装配

6. `v1/worker/`  
   设备初始化、model runner、执行落地

7. `core/sched/ + kv_cache_*`  
   每轮 batch 与 cache 资源编排

8. `model_executor/layers/attention_layer_base.py`  
   模型层 attention 抽象接口

9. `model_executor/layers/attention/attention.py`  
   attention 层与 backend 的桥接点

10. `v1/attention/backends/...`  
    平台 attention 执行协议层

11. `model_loader/ + models/registry.py + models/...`  
    具体模型类解析、实例化与权重加载

这时你再看 `vllm-mlu`，就能很系统地把它的改动归位到这 11 层中的某一层，而不是觉得“到处都改了很多文件”。

---

## 71. 回到 `vllm-mlu`：先看它怎么改主仓库里的 attention 层

如果现在从主仓库回到 `vllm-mlu`，最先应该看：

- `vllm_mlu/attention/layer.py`

这个文件非常关键，因为它正好处在：

- 主仓库模型层 attention 抽象
- MLU 平台具体 KV cache 规格
- MLU 平台 backend 选择

三者的交叉点上。

从这个文件可以清楚看出，`vllm-mlu` 不是简单重写全部 attention，而是先在主仓库 attention 层的接口上做最小必要替换。

### 71.1 它先改的是 KV cache spec

比如在 `Attention_MluHijack.get_kv_cache_spec()` 里，`vllm-mlu` 把主仓库默认的：

- `SlidingWindowSpec`
- `FullAttentionSpec`

替换成了：

- `MLUSlidingWindowSpec`
- `MLUFullAttentionSpec`

这一步的意义非常大。

因为这说明 MLU 平台首先不是在改“attention 数学逻辑”，而是在改：

**“这一层 attention 需要怎样的 KV cache 资源描述。”**

也就是说，`vllm-mlu` 在模型层接口处，优先把 cache 协议换成了 MLU 自己的版本。

### 71.2 它还改了 `MLAAttention` 的初始化路径

同一个文件里，`MLAAttention_MluHijack` 做了几件重要事情：

- 补充 `num_kv_heads`
- 根据 `mlu_config.decoder_attn_dtype` 注入解码 attention dtype
- 调 `get_attn_backend(..., use_mla=True, use_sparse=...)` 重新选择 backend
- 把 `MLAAttentionSpec` 替换成 `MLUMLAAttentionSpec`

这一步说明：

**就算还没有进入某个具体模型类，MLU 平台已经先把“通用 MLA attention 层”的 cache spec 和 backend 路径改掉了。**

这属于平台通用层改造，而不是某个模型特化。

### 71.3 它把 attention forward 的统一入口也改了

文件里还有一个重要函数：

- `unified_attention_with_output(...)`

这里最终会调用：

```python
self.impl.forward(...)
```

这说明 MLU 平台仍然保留了主仓库“attention 层 -> impl”的基本架构，只是在这个统一入口上加了自己的参数和返回值处理。

所以 `vllm_mlu/attention/layer.py` 的真正定位是：

**把主仓库 attention 层接口接到 MLU 自己的 KV cache spec、backend 选择和 forward 调用协议上。**

---

## 72. `vllm-mlu` 的平台通用 backend 层到底改了什么

接着看：

- `vllm_mlu/v1/attention/backends/flash_attn.py`
- `vllm_mlu/v1/attention/backends/mla/flashmla.py`

这两个文件代表的是 MLU 平台的“通用 attention 执行协议层”。

它们不是某个具体模型专属，而是平台级别的 backend 实现。

### 72.1 `MLUFlashAttentionBackend` 改的是平台通用协议

在 `MLUFlashAttentionBackend` 里，可以看到它做的事情非常典型：

- 定义支持的 kernel block sizes
- 定义支持的 head sizes
- 替换 impl class
- 替换 metadata class
- 替换 metadata builder
- 改掉 KV cache shape
- 增加 KV cache scale shape

这里最关键的是：

```python
def get_kv_cache_shape(...):
    return (2, num_blocks, num_kv_heads, block_size, head_size)
```

它和主仓库 `FlashAttentionBackend` 的默认布局已经不一样了。

这说明 MLU 平台在 backend 层明确地定义了自己偏好的 cache memory layout。

同时它还新增了：

- `MLUFlashAttentionMetadata`
- `MLUChunkFlashAttentionMetadata`
- `MLUFlashAttentionMetadataBuilder`

这些都说明 MLU 平台不仅换了 kernel，还换了 metadata 协议，尤其对 chunked prefill / decode 混合场景做了单独拆分。

也就是说：

**MLU backend 层主要解决的是“平台通用 attention 怎么执行”的问题。**

### 72.2 `FlashMLABackend` 改的是平台通用 MLA 协议

再看 `vllm_mlu/v1/attention/backends/mla/flashmla.py`，会更清楚。

这里除了定义：

- `FlashMLABackend`
- `FlashMLAImpl`
- `FlashMLAMetadataBuilder`

之外，还通过 hijack 修改了主仓库：

- `MLACommonBackend`
- `MLACommonMetadataBuilder`

而且它还专门根据 `deepseek_v4` 改 MLA dims 解析逻辑。

这说明 MLU 平台不只是做了一个“普通 attention backend”，它还把主仓库 MLA 通用层也接成了 MLU 版本。

这里要注意一个工程上的信号：

**当某个平台在 backend 层已经开始修改 MLACommon 这种抽象时，说明它的硬件执行协议和主仓库默认假设已经有明显差异。**

---

## 73. 再看具体模型：`MLUDeepseekV4Attention` 为什么必须单独写

现在再看：

- `vllm_mlu/model_executor/models/deepseek_v4.py`

前面我们已经讲过这个模型类，但这里要专门从“模型层 attention 和 backend 层分工”这个角度再看一次。

### 73.1 它内部仍然引用了通用 `MLAAttention`

在 `MLUDeepseekV4Attention.__init__()` 里，它内部会创建：

- `self.attn = MLAAttention(...)`

这说明它并没有完全脱离主仓库 attention 抽象体系。  
也就是说，它仍然在复用：

- attention layer
- backend 选择
- kv cache 容器

这些通用执行骨架。

### 73.2 但它在模型层自己组织了很多主仓库通用层表达不了的逻辑

例如它自己处理：

- `wq_a / wq_b / wkv / wo_a / wo_b`
- `rotary_emb / output_rotary_emb`
- compress ratio
- window size
- indexer / compressor
- output 投影

这些逻辑都强烈带有具体模型结构语义，不适合放进平台通用 backend。

也就是说，这部分工作必须留在模型层。

### 73.3 更关键的是，它在 forward 里直接调用了 MLU 专用算子

在 `forward_sparse_attn()` 里，它会直接做：

- `mlu_ops.reshape_paged_cache(...)`
- `mlu_ops.concat_block_table(...)`
- `mlu_ops.single_query_cached_kv_attn(...)`

这一步很重要，因为它说明：

**某些 DeepSeekV4 的注意力执行路径，已经不是“完全交给通用 backend 层”的模式，而是模型层自己直接下沉到了 MLU custom op。**

为什么会这样？

因为这里不仅仅是“算一个标准 attention”那么简单，它还包含：

- 压缩记忆
- window + compress block table 拼接
- indexer 选块
- 稀疏路径特化

这些都已经是模型特有执行逻辑了。

如果强行塞回平台通用 backend，反而会把 backend 抽象污染掉。

---

## 74. 这样就能看出 `vllm-mlu` 的 attention 双层结构

把上面几部分合起来，`vllm-mlu` 的 attention 体系其实可以清楚分成两层。

### 第一层：平台通用 attention 执行层

对应文件：

- `vllm_mlu/attention/layer.py`
- `vllm_mlu/v1/attention/backends/flash_attn.py`
- `vllm_mlu/v1/attention/backends/mla/flashmla.py`

这一层负责：

- 改 attention 层到 backend 的桥接点
- 改 KV cache spec
- 改 metadata builder
- 改 impl
- 改平台通用 cache layout
- 改平台通用 MLA / flash attention 协议

这一层更接近“平台执行协议层”。

### 第二层：具体模型特化 attention 层

对应文件：

- `vllm_mlu/model_executor/models/deepseek_v4.py`

这一层负责：

- 具体模型的 q/k/v/o 组织
- rotary / inverse rotary
- window / compress / indexer
- 特定 block table 拼接
- 特定算子直接调用

这一层更接近“模型语义 + 模型特化执行层”。

这个分层非常值得新芯片团队学习，因为它避免了两个常见错误：

- 把所有平台逻辑都写进具体模型类，导致无法复用
- 把所有模型特化逻辑都硬塞进 backend，导致 backend 抽象失控

---

## 75. 新芯片团队如何照着这个模式搭 attention 体系

如果你要照着 `vllm-mlu` 给一款新芯片做接入，attention 这条线建议按下面方式搭。

### 第一步：先做平台通用层

建议先完成：

- `<chip>/attention/layer.py`
- `<chip>/v1/attention/backends/<backend>.py`
- `<chip>/v1/attention/backends/mla/<backend>.py`

目标是先把：

- KV cache spec
- backend 选择
- metadata builder
- impl
- 通用 cache layout

跑通。

这一阶段尽量不要一上来就碰具体模型类。

### 第二步：只在必要时补具体模型 attention

只有当某些模型出现下面情况时，再去加：

- `<chip>/model_executor/models/<model>.py`

判定标准是：

- 主仓库通用 attention 层表达不了
- 模型需要压缩 / sparse / mixed cache / direct custom op
- 模型必须自己组织 block tables 或 cache 写入

也就是说，具体模型 attention 应该是平台通用 attention 层之上的特化补丁，而不是第一步。

### 第三步：让两层之间保持清晰边界

推荐规则：

- 通用 KV cache 布局、通用 metadata 协议放 backend 层
- 具体模型 q/k/v/o 与稀疏策略放模型层
- 只有模型特定的 custom op 直接调用，才放模型层
- 可被复用的 custom op 尽量由 backend / impl 吃掉

这个边界一旦守住，后面你要支持第二个、第三个模型时，成本会明显低很多。

---

## 76. 到这里，attention 这条主线算是和 `vllm-mlu` 真正对上了

现在可以把这条链压缩成一句话：

**主仓库提供了“模型层 attention -> backend -> impl”的三层结构；`vllm-mlu` 先在平台通用层改 KV cache spec、metadata 和 backend，再在少数具体模型里补充压缩、稀疏、direct custom op 这种模型特化逻辑。**

这也是一款新芯片平台最推荐的演进方式：

1. 先接平台通用 attention 执行协议
2. 再补模型特化 attention
3. 最后再针对热点模型深入做算子级定制

这比“一开始就重写整套模型 attention”更稳，也比“什么都往 backend 里塞”更可维护。

---

## 77. 一次真实 attention 执行，到底是怎么从 metadata builder 走到 custom op 的

现在继续往下，不再停留在职责分层，而是直接看一次真实执行链。

如果把 `vllm-mlu` 里一次 attention 执行压缩成一句话，就是：

**scheduler / runner 先产出 common metadata，backend 的 metadata builder 把它翻译成平台专用 metadata，impl.forward 再根据 infer mode 和 cache 状态选择具体 MLU 算子执行。**

这条链正好对应三个位置：

- metadata builder
- impl.forward
- `mlu_ops.*`

---

## 78. 第一步：common metadata 先被翻译成 MLU backend 能吃的 metadata

先看：

- `vllm_mlu/v1/attention/backends/flash_attn.py`
- `MLUFlashAttentionMetadataBuilder.build()`

这个函数的角色，不是“算 attention”，而是把上游统一格式翻译成 MLU backend 私有协议。

它吃进去的是：

- `MLUCommonAttentionMetadata`

这里面已经有了：

- `num_reqs`
- `num_actual_tokens`
- `query_start_loc`
- `seq_start_loc`
- `seq_lens`
- `block_table_tensor`
- `slot_mapping`
- `infer_mode`

这些字段本质上是 scheduler 和 model runner 在更上游整理好的“本轮批次公共信息”。

builder 的职责是把这些字段重组为：

- `MLUFlashAttentionMetadata`
- 必要时再生成 `MLUChunkFlashAttentionMetadata`

### 78.1 为什么 MLU 要单独扩展 metadata

因为它比主仓库通用 FlashAttention metadata 多关心几件事：

- `seq_start_loc`
- `infer_mode`
- `num_input_tokens`
- `compute_dtype`
- chunked prefill 下 prefill/decode 拆分后的上下文

也就是说，MLU backend 想知道的不只是“这一批 token 怎么排”，还想知道：

**“这批 token 当前处在什么推理阶段，以及是否值得拆成两条更高效的执行路径。”**

### 78.2 chunked prefill 是在 metadata builder 里真正分流的

在 `build()` 里有一个关键动作：

- 如果 `common_attn_metadata.infer_mode.is_chunked`
- 就构造 `MLUChunkFlashAttentionMetadata`

这个对象会把同一批请求再细分成：

- `prefill_ctx`
- `decode_ctx`

每一部分都单独带：

- batch size
- `cu_seqlens_q`
- `cu_seqlens_kv`
- `max_query_len`
- `max_seq_len`

这一步非常关键，因为它说明：

**chunked prefill 的“拆 prefill / decode”不是在 kernel 层临时硬判断，而是在 metadata builder 层就已经明确编码了。**

这对新芯片团队是个很重要的参考：

- 执行路径分流，尽量放在 metadata/build 阶段
- 不要把所有分流逻辑堆到 impl.forward 里

---

## 79. 第二步：`impl.forward` 决定本轮到底走哪条执行路径

继续看：

- `MLUFlashAttentionImpl.forward()`

这个函数才是真正的 attention 执行总入口。

但它本身也不直接只做一件事，而是在 metadata 已经翻译好的基础上，再做一次“执行决策”。

它主要做 4 件事。

### 79.1 先判断是否要写 KV cache

它会先拿到：

- `key_cache`
- `value_cache`
- `key_cache_scale`
- `value_cache_scale`

然后根据：

- 是否量化 KV cache
- 是否 MLA
- 是否 prefill-only
- 是否 fused MLA QKV
- 是否 sharing KV

决定是不是要先把当前 `key/value` 写入 paged cache。

如果要写，就调：

- `mlu_ops.reshape_paged_cache(...)`
- 或 `mlu_ops.quant_to_paged_cache(...)`

这一步说明：

**impl.forward 不是只负责“读 cache 算 attention”，它还负责“必要时把本轮 token 先写进 cache”。**

### 79.2 再根据 infer mode 选择执行分支

MLU 这里把执行模式分得很清楚：

- `prefill_only`
- `chunked`
- `decode_only`

然后分别走：

- `mlu_ops.flash_attention(...)`
- `_forward_prefill_chunk(...)`
- `_forward_decode_only(...)`

这说明真正的执行分支，不是靠调用方分别调三个函数，而是：

**统一进 `impl.forward`，再根据 metadata 里的 infer mode 进行分流。**

### 79.3 prefill 和 decode 在 MLU 上是两种完全不同的执行形态

在 `prefill_only` 场景下，MLU 直接调：

- `mlu_ops.flash_attention(...)`

这时候更像标准的 varlen flash attention。

而在 `decode_only` 场景下，会走：

- `_forward_decode_only()`
- 里面再调 `mlu_ops.single_query_cached_kv_attn(...)`

这时候就变成了：

- query 很短
- key/value 来自 paged KV cache
- block table 和 `context_lens` 参与访问

也就是说，MLU 把 prefill 和 decode 看成两类完全不同的 kernel 问题，这和很多硬件平台的优化路径是一致的。

### 79.4 chunked 场景不是简单复用其中一个分支，而是两段都跑

在 `infer_mode.is_chunked` 时，`impl.forward` 会：

- 先从 `chunk_fa_metadata` 拿 `prefill_ctx`
- 再拿 `decode_ctx`
- prefill 部分走 `_forward_prefill_chunk()`
- decode 部分走 `_forward_decode_only()`

也就是说，同一轮 step 里，MLU backend 允许：

- 一部分 token 走 prefill kernel
- 另一部分 token 走 decode kernel

这就是 chunked prefill 对性能真正有价值的地方。

---

## 80. 第三步：真正下沉到 custom op 时，已经不再讲抽象，只讲数据格式

继续往下看 `_forward_prefill_chunk()` 和 `_forward_decode_only()`，就会发现这里的逻辑已经非常“硬件后端化”了。

### 80.1 prefill chunk 走的是连续 attention 路径

`_forward_prefill_chunk()` 里，如果 KV cache 是量化的，它会先：

- `mlu_ops.dequant_from_paged_cache(...)`

把 paged cache 解成连续 K/V。

然后再调：

- `mlu_ops.flash_attention(...)`

而且传进去的是非常具体的执行参数：

- `cu_seq_lens_q`
- `cu_seq_lens_kv`
- `max_seq_len_q`
- `max_seq_len_kv`
- `block_tables`
- `compute_dtype`

这说明到了 custom op 层，已经不关心“这是哪个模型”，只关心：

- tensor shape
- cache layout
- 序列边界
- block table

### 80.2 decode only 走的是 cached single-query 路径

`_forward_decode_only()` 里会把 query/output 先 reshape 成：

- `[batch, query_len, num_heads, head_size]`

然后调：

- `mlu_ops.single_query_cached_kv_attn(...)`

输入信息包括：

- `block_tables`
- `context_lens`
- KV cache quant scale
- `max_contxt_len`
- window size
- `decoder_attn_dtype`

这时候已经完全是“paged decode kernel 接口”了。

也就是说，到了这一步：

**backend 抽象已经结束，真正生效的是 custom op 对输入格式的要求。**

---

## 81. MLA 路径也遵循同一套路，只是 metadata 更复杂

再看：

- `vllm_mlu/v1/attention/backends/mla/flashmla.py`

它的模式其实是一样的，只是 MLA 本身更复杂。

### 81.1 `FlashMLAMetadataBuilder.build()` 负责把 batch 拆成 MLA 可执行形态

这里会构造：

- `prefill_metadata`
- `decode_metadata`

还会根据：

- 是否 chunked prefill
- 是否有 context
- 是否使用 workspace

进一步准备 chunked context metadata。

也就是说，MLA builder 的职责比普通 flash attention builder 更重，因为它要为 MLA 的不同执行阶段准备更多上下文。

### 81.2 `FlashMLAImpl.forward()` 仍然保持统一入口

它依然是：

- 先看 metadata
- 再判断 only_prefill / only_decode
- 再决定是否写 cache
- 最后调 `mlu_ops.single_query_cached_kv_attn(...)` 或其他 MLA 相关路径

这说明一个很重要的设计原则：

**即便 backend 很复杂，也尽量维持“builder 负责翻译 metadata，impl.forward 负责统一调度执行”的结构不变。**

这个结构稳定了，后面你换 kernel、加量化、补 spec decode，改动范围都会更可控。

---

## 82. 这样就能看清一次真实执行里的三层边界

现在可以把真实执行中的三层边界写得非常清楚。

### metadata builder 负责“翻译批次”

它不算 attention，但它决定：

- 本轮有哪些 token
- prefill 和 decode 如何拆分
- block table 怎么解释
- cache 边界和序列边界怎么表达
- impl 需要哪些额外上下文

### impl.forward 负责“选择执行路径”

它不负责定义模型语义，但它决定：

- 是否写 cache
- 走 prefill / decode / chunked 哪条路径
- 量化 cache 如何处理
- 最后调哪个 custom op

### custom op 负责“真正算”

到 `mlu_ops.flash_attention`、`mlu_ops.single_query_cached_kv_attn` 这一步时，逻辑已经完全收敛成：

- 输入张量
- cache 张量
- metadata 张量
- 标量参数

这里不再关心模型结构，也不再关心 scheduler，只关心算子接口契约。

---

## 83. 新芯片团队应该照着这条链怎么落地

如果你要给自己的芯片复刻这条执行链，建议严格按下面顺序做。

### 第一步：先定义 common metadata 到 backend metadata 的翻译层

先做：

- `Metadata`
- `MetadataBuilder`

先把这些字段梳理清楚：

- `query_start_loc`
- `seq_start_loc`
- `seq_lens`
- `block_table`
- `slot_mapping`
- `infer_mode`

如果这一步没定义清楚，后面 impl 和 kernel 一定会越来越乱。

### 第二步：再实现统一的 `impl.forward`

这个函数至少要明确三件事：

- 什么时候写 cache
- 什么时候走 prefill
- 什么时候走 decode

不要一开始就把所有逻辑散落到多个入口函数里。  
先保持一个统一入口，后面再把热点逻辑下沉成私有 helper。

### 第三步：最后才把具体 kernel/op 接口稳定下来

也就是最后才固定：

- paged cache layout
- quant scale layout
- decode kernel 输入 shape
- prefill kernel 输入 shape

这个顺序非常重要。  
如果先固化 kernel 接口，再反推 metadata，经常会把上层抽象绑死。

---

## 84. 这条执行链给新芯片团队的最大启发

`vllm-mlu` 在这条链上最值得学的，不是某个具体 `mlu_ops` 名字，而是它把 attention 执行拆成了三段：

1. 上游统一调度信息
2. backend 私有 metadata 翻译
3. impl 决策并调用 custom op

这套拆法的价值是：

- scheduler 不需要知道 kernel 细节
- model 层不需要知道 paged cache 的底层布局
- custom op 不需要知道请求生命周期

每一层只关心自己的契约。

对一款新芯片来说，这比“直接把 kernel 塞进模型 forward 里”更容易扩展，也更容易支持第二种模型、第二种 cache 模式、第二种并行策略。

---

## 85. 最小可落地 attention backend 模板

下面给一个适合新芯片团队直接起步的最小骨架。  
这套模板的目标不是“一次写完”，而是先把 attention 执行链的 4 个边界固定住：

1. backend class
2. metadata dataclass
3. metadata builder
4. impl.forward

建议目录：

```text
vllm_<chip>/
├── platforms/
│   └── chip.py
├── attention/
│   └── layer.py
├── v1/
│   └── attention/
│       └── backends/
│           ├── flash_attn.py
│           └── mla/
│               └── flashmla.py
└── _chip_ops.py
```

---

## 86. 模板一：平台选择入口

先在平台层把 backend 挂进去。

文件：

- `vllm_<chip>/platforms/chip.py`

```python
from vllm.platforms.interface import Platform


class ChipPlatform(Platform):
    device_name = "chip"
    dispatch_key = "CHIP"
    dist_backend = "chipcl"
    device_control_env_var = "CHIP_VISIBLE_DEVICES"

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, head_size, dtype,
                             kv_cache_dtype, block_size, use_v1,
                             use_mla=False, has_sink=False,
                             use_sparse=False):
        if use_mla:
            return "vllm_<chip>.v1.attention.backends.mla.flashmla.FlashMLABackend"
        return "vllm_<chip>.v1.attention.backends.flash_attn.ChipFlashAttentionBackend"
```

这一层只做一件事：

- 根据 `use_mla`、平台能力和配置，返回 backend 类路径

不要在这里放 kernel 细节。

---

## 87. 模板二：先把 KV cache spec 接到你自己的平台

如果你的平台需要自定义 cache 布局，先改 attention layer 的 `get_kv_cache_spec()`。

文件：

- `vllm_<chip>/attention/layer.py`

```python
from vllm.attention.layer import Attention
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_<chip>.v1.kv_cache_interface import (
    ChipFullAttentionSpec,
    ChipSlidingWindowSpec,
)


class Attention_ChipHijack(Attention):
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        block_size = vllm_config.cache_config.block_size
        if self.sliding_window is not None:
            return ChipSlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                dtype=self.kv_cache_torch_dtype,
                sliding_window=self.sliding_window,
            )
        return ChipFullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=self.kv_cache_torch_dtype,
        )
```

这一层的目的很简单：

- 不碰模型结构
- 先把 cache 资源描述切成你平台自己的版本

如果你的平台 cache layout 和主仓库完全一致，这一层甚至可以先不改。

---

## 88. 模板三：backend class 骨架

文件：

- `vllm_<chip>/v1/attention/backends/flash_attn.py`

```python
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vllm.v1.attention.backend import AttentionImpl
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec


class ChipFlashAttentionBackend(FlashAttentionBackend):
    supported_kernel_block_sizes: ClassVar[list[int]] = [16, 32, 64]

    @staticmethod
    def get_name() -> str:
        return "CHIP_FLASH_ATTN"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 80, 96, 128, 160, 192, 256]

    @staticmethod
    def get_impl_cls() -> type["ChipFlashAttentionImpl"]:
        return ChipFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["FlashAttentionMetadata"]:
        return ChipFlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["ChipFlashAttentionMetadataBuilder"]:
        return ChipFlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, num_kv_heads, block_size, head_size)
```

这一层先固定 5 件事：

- backend 名字
- 支持的 head size
- impl class
- metadata class
- KV cache shape

一开始不要急着把所有 capability check 都写满。  
先把最小的 happy path 跑通。

---

## 89. 模板四：metadata dataclass

```python
@dataclass
class ChipFlashAttentionMetadata(FlashAttentionMetadata):
    seq_start_loc: torch.Tensor | None = None
    infer_mode: object | None = None
    num_input_tokens: int = 0
    compute_dtype: torch.dtype = torch.float32
    chunk_metadata: object | None = None
```

这一步的原则是：

- 先继承主仓库 metadata
- 只额外加你平台真正需要的字段

推荐最先只加这几个：

- `seq_start_loc`
- `infer_mode`
- `num_input_tokens`
- `compute_dtype`

如果后面要支持 chunked prefill，再补：

- `chunk_metadata`

不要一上来造一个完全脱离主仓库的新 metadata 体系，维护成本会很高。

---

## 90. 模板五：metadata builder 骨架

```python
class ChipFlashAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.uniform_decode_query_len = 1

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
    ) -> ChipFlashAttentionMetadata:
        attn_metadata = ChipFlashAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            use_cascade=False,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
            scheduler_metadata=None,
            prefix_scheduler_metadata=None,
            max_num_splits=0,
            causal=common_attn_metadata.causal,
            seq_start_loc=common_attn_metadata.seq_start_loc,
            infer_mode=common_attn_metadata.infer_mode,
            num_input_tokens=getattr(common_attn_metadata, "num_input_tokens", 0),
        )

        if getattr(common_attn_metadata.infer_mode, "is_chunked", False):
            attn_metadata.chunk_metadata = self._build_chunk_metadata(
                common_attn_metadata)

        return attn_metadata

    def _build_chunk_metadata(self, common_attn_metadata):
        return {
            "num_decodes": 0,
            "num_prefills": common_attn_metadata.num_reqs,
        }
```

builder 层最重要的不是代码长短，而是边界清楚：

- 输入是 `common_attn_metadata`
- 输出是 `ChipFlashAttentionMetadata`
- 如果有 chunked prefill，就在这里分流

这层不要直接调 kernel。

---

## 91. 模板六：impl.forward 最小骨架

```python
from vllm.v1.attention.backend import AttentionImpl

from vllm_<chip> import _chip_ops as chip_ops


class ChipFlashAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes,
        sliding_window,
        kv_cache_dtype: str,
        logits_soft_cap=None,
        attn_type=None,
        kv_sharing_target_layer_name=None,
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.sliding_window = (-1, -1) if sliding_window is None else (
            sliding_window - 1, 0)
        self.is_mla = extra_impl_args.get("is_mla", False)

    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: ChipFlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        kwargs: dict[str, Any] = {},
    ) -> torch.Tensor:
        assert output is not None
        if attn_metadata is None:
            return output.fill_(0)

        infer_mode = attn_metadata.infer_mode

        key_cache, value_cache = kv_cache[0].unbind(0)

        if self._should_write_cache(key, value, infer_mode):
            chip_ops.reshape_paged_cache(
                k=key[:attn_metadata.num_actual_tokens],
                v=value[:attn_metadata.num_actual_tokens],
                k_cache=key_cache,
                v_cache=value_cache,
                slot_mapping=attn_metadata.slot_mapping.flatten(),
            )

        if infer_mode.is_prefill_only:
            self._forward_prefill(
                query=query[:attn_metadata.num_actual_tokens],
                key=key[:attn_metadata.num_actual_tokens],
                value=value[:attn_metadata.num_actual_tokens],
                output=output[:attn_metadata.num_actual_tokens],
                attn_metadata=attn_metadata,
            )
        elif infer_mode.is_chunked:
            self._forward_chunked(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                output=output,
                attn_metadata=attn_metadata,
            )
        else:
            self._forward_decode(
                query=query[:attn_metadata.num_actual_tokens],
                key_cache=key_cache,
                value_cache=value_cache,
                output=output[:attn_metadata.num_actual_tokens],
                attn_metadata=attn_metadata,
            )

        return output

    def _should_write_cache(self, key, value, infer_mode) -> bool:
        return key is not None and value is not None and not infer_mode.is_prefill_only

    def _forward_prefill(self, query, key, value, output, attn_metadata):
        chip_ops.flash_attention(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seq_lens_q=attn_metadata.query_start_loc,
            cu_seq_lens_kv=attn_metadata.seq_start_loc,
            max_seq_len_q=attn_metadata.max_query_len,
            max_seq_len_kv=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
        )

    def _forward_chunked(self, query, key_cache, value_cache, output, attn_metadata):
        # 这里先留最小占位，后续再拆 prefill_ctx / decode_ctx
        self._forward_decode(
            query=query[:attn_metadata.num_actual_tokens],
            key_cache=key_cache,
            value_cache=value_cache,
            output=output[:attn_metadata.num_actual_tokens],
            attn_metadata=attn_metadata,
        )

    def _forward_decode(self, query, key_cache, value_cache, output, attn_metadata):
        batch_size = attn_metadata.block_table.shape[0]
        decode_query = query.view(batch_size, -1, self.num_heads, self.head_size)
        decode_output = output.view(batch_size, -1, self.num_heads, self.head_size)
        chip_ops.single_query_cached_kv_attn(
            q=decode_query,
            k_cache=key_cache,
            v_cache=value_cache,
            out=decode_output,
            block_tables=attn_metadata.block_table,
            context_lens=attn_metadata.seq_lens,
            max_contxt_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
        )
```

这个骨架体现了最重要的 3 个边界：

- cache 写入在 `forward()` 统一处理
- prefill / chunked / decode 分流在 `forward()` 统一处理
- 具体 kernel 调用下沉到 `_forward_*`

先把结构做对，再做性能优化。

---

## 92. 模板七：MLA backend 最小骨架

如果你的平台后面要支持 MLA，建议不要一开始做复杂版，先做一个“和普通 backend 结构一致”的最小模板。

文件：

- `vllm_<chip>/v1/attention/backends/mla/flashmla.py`

```python
from vllm.v1.attention.backends.mla.common import MLACommonBackend

from vllm_<chip>.v1.attention.backends.flash_attn import (
    ChipFlashAttentionImpl,
)


class FlashMLABackend(MLACommonBackend):
    @staticmethod
    def get_name() -> str:
        return "CHIP_FLASH_MLA"

    @staticmethod
    def get_impl_cls():
        return FlashMLAImpl

    @staticmethod
    def get_builder_cls():
        return FlashMLAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (1, num_blocks, num_kv_heads, block_size, head_size)


class FlashMLAMetadataBuilder:
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.kv_cache_spec = kv_cache_spec
        self.vllm_config = vllm_config
        self.device = device

    def build(self, common_prefix_len, common_attn_metadata,
              fast_build: bool = False, input_batch=None):
        return common_attn_metadata


class FlashMLAImpl(ChipFlashAttentionImpl):
    def forward(self, layer, query, key, value, kv_cache,
                attn_metadata, output=None, output_scale=None,
                kwargs=None):
        assert output is not None
        # 先复用普通 decode 路径，后续再补 MLA 专属 metadata 和 kernel
        return super().forward(
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            output_scale=output_scale,
            kwargs=kwargs or {},
        )
```

这一步的重点不是“马上支持完整 MLA”，而是先把：

- 平台选择入口
- backend class
- builder
- impl

这 4 个点都占住。

---

## 93. 模板八：自定义算子接口最小定义

文件：

- `vllm_<chip>/_chip_ops.py`

```python
def reshape_paged_cache(k, v, k_cache, v_cache, slot_mapping):
    raise NotImplementedError


def flash_attention(
    q,
    k,
    v,
    out,
    cu_seq_lens_q,
    cu_seq_lens_kv,
    max_seq_len_q,
    max_seq_len_kv,
    softmax_scale,
):
    raise NotImplementedError


def single_query_cached_kv_attn(
    q,
    k_cache,
    v_cache,
    out,
    block_tables,
    context_lens,
    max_contxt_len,
    softmax_scale,
):
    raise NotImplementedError
```

为什么这里故意只保留最少参数？

因为新芯片团队一开始最容易犯的错，就是直接把所有未来可能支持的参数一次塞进接口。  
更稳的做法是：

- 先只保留 happy path 所需参数
- 跑通后再逐步增加 quant scale、window、alibi、decoder dtype、chunk workspace 等扩展字段

---

## 94. 这套模板的推荐落地顺序

建议按下面顺序真正实现：

1. `ChipPlatform.get_attn_backend_cls()`
2. `ChipFlashAttentionBackend`
3. `ChipFlashAttentionMetadata`
4. `ChipFlashAttentionMetadataBuilder`
5. `ChipFlashAttentionImpl.forward()`
6. `_chip_ops.reshape_paged_cache`
7. `_chip_ops.flash_attention`
8. `_chip_ops.single_query_cached_kv_attn`
9. `Attention_ChipHijack.get_kv_cache_spec()`
10. `FlashMLABackend`

这个顺序的好处是：

- 先打通普通 attention
- 再补 cache spec
- 最后再做 MLA

不要反过来先做 MLA。  
MLA 会把 metadata、cache 布局和 kernel 复杂度同时放大。

---

## 95. 模板阶段的验收标准

这套最小模板什么时候算“搭起来了”，建议按下面标准验收。

### 第一阶段验收

- backend 可以被平台正确选中
- `Attention` 层能拿到 `impl`
- builder 能产出平台 metadata
- `impl.forward()` 能被调用

### 第二阶段验收

- prefill-only 能正确执行
- decode-only 能正确执行
- KV cache 能正确写入和读取

### 第三阶段验收

- chunked prefill 可以工作
- MLA 可以跑通最小 case
- 量化 KV cache 才开始补

这个验收顺序也建议写进团队排期里，否则很容易被“先支持所有 feature”带偏。

---

## 96. 最小可落地 worker / model runner 模板

attention backend 模板解决的是：

- backend 怎么选
- metadata 怎么构造
- kernel 怎么调

但一款新芯片平台真正能跑起来，还必须再有一条执行主线把它串起来：

- `Worker`
- `ModelRunner`
- `load_model()`
- `execute_model()`

这一节就专门给出最小骨架。

建议目录补成这样：

```text
vllm_<chip>/
├── v1/
│   ├── worker/
│   │   ├── gpu_worker.py
│   │   └── gpu_model_runner.py
│   └── engine/
│       └── core.py
└── model_executor/
    └── models/
        ├── __init__.py
        └── my_model.py
```

这里仍然沿用 vLLM 主仓库的命名习惯，即使你不是 GPU，也建议初期保留 `gpu_worker.py` / `gpu_model_runner.py` 这种路径名。  
先复用主仓库调用链，后面再考虑是否改名。

---

## 97. 模板一：最小 `Worker`

文件：

- `vllm_<chip>/v1/worker/gpu_worker.py`

```python
import torch

from vllm.worker.gpu_worker import Worker

from vllm_<chip>.v1.worker.gpu_model_runner import ChipModelRunner


class ChipWorker(Worker):
    def init_device(self) -> None:
        self.device = torch.device(f"chip:{self.local_rank}")
        torch.chip.set_device(self.device)
        torch.chip.empty_cache()

        self.model_runner = ChipModelRunner(self.vllm_config, self.device)

    def load_model(self) -> None:
        self.model_runner.load_model()

    def determine_available_memory(self) -> int:
        free_mem, _ = torch.chip.mem_get_info(self.device)
        return free_mem

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        return self.model_runner.execute_model(
            scheduler_output,
            intermediate_tensors=intermediate_tensors,
        )
```

最小 `Worker` 只需要先干 4 件事：

- 绑定设备
- 清理显存
- 创建 `ModelRunner`
- 把 `execute_model()` 转发给 `ModelRunner`

一开始不要急着加：

- 分布式初始化
- profiler
- graph capture
- kv transfer

先把单卡链路跑通。

---

## 98. 模板二：最小 `ModelRunner.__init__`

文件：

- `vllm_<chip>/v1/worker/gpu_model_runner.py`

```python
import torch


class ChipModelRunner:
    def __init__(self, vllm_config, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device

        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.quant_config = vllm_config.quant_config
        self.load_config = vllm_config.load_config

        self.model = None

        max_num_tokens = self.scheduler_config.max_num_batched_tokens
        hidden_size = self.model_config.get_hidden_size()

        self.input_ids = torch.empty(
            max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self.positions = torch.empty(
            max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self.inputs_embeds = torch.empty(
            (max_num_tokens, hidden_size),
            dtype=self.model_config.dtype,
            device=device,
        )
```

`__init__()` 阶段只做两件事：

- 保存配置
- 预分配运行时最基本的 buffer

这里先别把所有 buffer 都抄全。  
最小起步只保留：

- `input_ids`
- `positions`
- `inputs_embeds`

后面随着 attention、sampling、multimodal、spec decode 接入，再逐步加。

---

## 99. 模板三：最小 `load_model()`

```python
from vllm.model_executor.model_loader import get_model_loader


class ChipModelRunner:
    ...

    def load_model(self) -> None:
        model_loader = get_model_loader(self.load_config)
        self.model = model_loader.load_model(
            vllm_config=self.vllm_config,
            model_config=self.model_config,
        )

        self.model = self.model.to(self.device)
        self.model.eval()
```

这一步的关键思想是：

- `ModelRunner` 自己不要直接硬编码“实例化哪个模型类”
- 先复用主仓库 `model_loader`
- 通过 `ModelRegistry` 解析模型类

这样做的好处是：

- 你的平台一开始可以直接吃主仓库已有模型类
- 后面如果要替换成平台特化模型，只需要改 `registry`

如果你的平台确实需要在加载完模型后包一层 wrapper，也建议在这里做：

```python
self.model = ChipGraphWrapper(self.model)
```

但一开始先不要加 wrapper，先保证裸模型能跑。

---

## 100. 模板四：最小 `execute_model()`

这是整条执行链里最关键的函数。

```python
from vllm.forward_context import set_forward_context


class ChipModelRunner:
    ...

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        attn_metadata = self._build_attn_metadata(scheduler_output)
        model_input = self._prepare_inputs(scheduler_output)

        with set_forward_context(attn_metadata=attn_metadata):
            hidden_states = self.model(
                input_ids=model_input["input_ids"],
                positions=model_input["positions"],
                inputs_embeds=model_input.get("inputs_embeds"),
            )

        output = self._build_model_runner_output(
            hidden_states=hidden_states,
            scheduler_output=scheduler_output,
        )
        return output
```

这个最小骨架强烈建议你保持不变，先把 3 个步骤固定住：

1. `_build_attn_metadata()`
2. `_prepare_inputs()`
3. `_build_model_runner_output()`

因为这 3 个步骤几乎就是所有平台最终都会接管的核心边界。

---

## 101. 模板五：最小 `_prepare_inputs()`

```python
class ChipModelRunner:
    ...

    def _prepare_inputs(self, scheduler_output):
        num_tokens = scheduler_output.total_num_scheduled_tokens

        input_ids = self.input_ids[:num_tokens]
        positions = self.positions[:num_tokens]

        input_ids.copy_(scheduler_output.input_ids, non_blocking=True)
        positions.copy_(scheduler_output.positions, non_blocking=True)

        return {
            "input_ids": input_ids,
            "positions": positions,
        }
```

这一层只管一件事：

- 把 scheduler 下发的数据整理成模型 forward 能直接吃的 tensor

开始阶段只要能支持：

- `input_ids`
- `positions`

就够了。

后面再逐步补：

- `inputs_embeds`
- multimodal inputs
- LoRA inputs
- grammar / spec decode 辅助信息

---

## 102. 模板六：最小 `_build_attn_metadata()`

```python
from vllm.v1.attention.backends.registry import get_attn_backend


class ChipModelRunner:
    ...

    def _build_attn_metadata(self, scheduler_output):
        backend = get_attn_backend(
            head_size=self.model_config.get_head_size(),
            dtype=self.model_config.dtype,
            kv_cache_dtype=self.cache_config.cache_dtype,
            block_size=self.cache_config.block_size,
            use_mla=False,
        )

        kv_cache_spec = self.model.get_kv_cache_spec(self.vllm_config)
        builder_cls = backend.get_builder_cls()
        builder = builder_cls(
            kv_cache_spec=kv_cache_spec,
            layer_names=[],
            vllm_config=self.vllm_config,
            device=self.device,
        )

        common_attn_metadata = self._build_common_attn_metadata(scheduler_output)
        return builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            fast_build=False,
        )
```

这里的重点不是代码细节，而是顺序：

1. 先拿 backend
2. 再拿 builder
3. 再拿 common metadata
4. 最后 build 成平台 metadata

这一步正好把你前面写的 attention backend 模板，接回到真实执行链里。

### `_build_common_attn_metadata()` 最小形态

```python
class ChipModelRunner:
    ...

    def _build_common_attn_metadata(self, scheduler_output):
        return scheduler_output.attn_metadata
```

在最小起步阶段，可以先直接透传。  
等你开始自己接管 chunked prefill、prefix cache、graph batch padding 时，再把这里变成平台自己的构造逻辑。

---

## 103. 模板七：最小 `_build_model_runner_output()`

```python
class ChipModelRunner:
    ...

    def _build_model_runner_output(self, hidden_states, scheduler_output):
        return {
            "hidden_states": hidden_states,
            "num_scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
        }
```

一开始这个输出可以非常简陋。  
重点不是先把采样、logits、spec decode 都做全，而是先验证：

- 模型确实被执行了
- attention backend 确实被走到了
- forward 输出能从 `ModelRunner` 回到 `Worker`

后面再把它逐步升级成主仓库那种完整 `ModelRunnerOutput`。

---

## 104. 模板八：最小平台专属 `EngineCore` 接线

如果你的平台需要自定义 `EngineCore`，建议一开始只做最小接线，不改 `step()` 主逻辑。

文件：

- `vllm_<chip>/v1/engine/core.py`

```python
from vllm.v1.engine.core import EngineCore


class EngineCore_ChipHijack(EngineCore):
    def _get_executor_cls(self):
        return super()._get_executor_cls()
```

一开始尽量不要动：

- `schedule()`
- `execute_model()`
- `update_from_output()`

也就是先复用主仓库：

- scheduler
- executor
- engine loop

先把平台工作聚焦在：

- worker
- model runner
- attention backend

这 3 层。

---

## 105. 最小执行链是怎样串起来的

如果你把上面这些模板连起来，最小单卡执行链应该是这样：

1. `LLMEngine.step()` 调度出 `scheduler_output`
2. `Worker.execute_model()` 接住这一轮执行请求
3. `ModelRunner._build_attn_metadata()` 构造本轮 attention metadata
4. `ModelRunner._prepare_inputs()` 整理本轮输入
5. `self.model(...)` 被调用
6. 模型里的 `Attention` 层选到你平台的 backend
7. backend builder / impl / custom op 被触发
8. `ModelRunner` 产出输出
9. 输出回到 `Worker`
10. 输出回到 `EngineCore`

如果这 10 步能单卡闭环，你的平台执行骨架就算真正立住了。

---

## 106. 这一阶段推荐的最小实现顺序

建议按下面顺序落代码：

1. `ChipWorker.init_device()`
2. `ChipModelRunner.__init__()`
3. `ChipModelRunner.load_model()`
4. `ChipModelRunner._prepare_inputs()`
5. `ChipModelRunner._build_attn_metadata()`
6. `ChipModelRunner.execute_model()`
7. `ChipModelRunner._build_model_runner_output()`
8. `Attention backend` 模板
9. `_chip_ops` 最小实现

这个顺序的好处是：

- 先把模型加载起来
- 再把输入喂进去
- 再把 attention 链接起来

不要先写复杂 scheduler，也不要先做多卡。  
这一步的目标就是：

**让单卡模型在你平台上完成一次真实 forward。**

---

## 107. 这一阶段的验收标准

### 第一阶段验收

- `Worker` 能正常初始化设备
- `ModelRunner.load_model()` 能成功加载模型
- `execute_model()` 能成功调用到 `self.model(...)`

### 第二阶段验收

- `Attention` 层能正确选到你平台 backend
- builder 能构造 metadata
- `impl.forward()` 能进入你平台 custom op

### 第三阶段验收

- 单卡 prefill 跑通
- 单卡 decode 跑通
- KV cache 写入与读取正常

只有做到这里，才建议开始做：

- chunked prefill
- graph capture
- TP / EP / DP
- quantized KV cache
- MLA

---

## 108. 新芯片团队在执行层最容易犯的错

这一层最常见的坑有 4 个。

### 错误 1：`Worker` 写太重

很多团队会把：

- 输入准备
- metadata 构造
- 输出封装

都塞进 `Worker`。

更稳的边界是：

- `Worker` 管设备和转发
- `ModelRunner` 管模型执行细节

### 错误 2：`ModelRunner` 直接绑死具体模型

如果一开始就在 `load_model()` 里写死：

- `if model_type == xxx: self.model = XxxModel(...)`

后面支持第二个模型会越来越乱。  
更好的做法是优先复用：

- `model_loader`
- `ModelRegistry`

### 错误 3：`execute_model()` 没有固定边界

如果 `_prepare_inputs()`、`_build_attn_metadata()`、`_build_output()` 不拆开，后面接：

- spec decode
- multimodal
- chunked prefill

几乎一定会失控。

### 错误 4：过早把多卡逻辑塞进最小链路

最小模板阶段不要急着做：

- TP collectives
- CNCL/NCCL 封装
- pipeline parallel

单卡链路不稳定时，多卡问题只会放大排查成本。

---

## 109. 最小可落地 model registry / model class / load_weights 模板

到这里，平台、attention、worker、model runner 都已经有了最小骨架。  
还差最后一段：

- 模型怎么注册
- 模型类怎么写
- 权重怎么加载

这一段如果不先模板化，很多团队后面会把“模型结构定义”和“checkpoint 权重适配”混在一起，导致每接一个模型都要重新梳理。

建议目录再补成这样：

```text
vllm_<chip>/
└── model_executor/
    └── models/
        ├── __init__.py
        ├── registry.py
        └── my_model.py
```

---

## 110. 模板一：模型注册入口

文件：

- `vllm_<chip>/model_executor/models/__init__.py`

```python
from vllm.model_executor.models import ModelRegistry


def register_models() -> None:
    ModelRegistry.register_model(
        "MyModelForCausalLM",
        "vllm_<chip>.model_executor.models.my_model:ChipMyModelForCausalLM",
    )
```

这一步只做一件事：

- 把 HuggingFace config 里的 `architecture` 名字，映射到你平台自己的模型类

也就是说，先固定：

- 解析入口是 `ModelRegistry`
- 不是 `ModelRunner.load_model()` 里手写 `if architecture == ...`

这一步越早规范，后面支持第二个模型时越省事。

### 推荐的插件接线方式

在你的插件初始化里调用：

```python
from vllm_<chip>.model_executor.models import register_models


def register_chip_hijack():
    register_models()
```

这样做的好处是：

- 平台加载时模型注册自动生效
- `model_loader` 仍然走主仓库通用链路

---

## 111. 模板二：最小模型类骨架

文件：

- `vllm_<chip>/model_executor/models/my_model.py`

```python
from collections.abc import Iterable

import torch
from torch import nn

from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import SupportsLoRA


class ChipMyModel(nn.Module):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        hf_config = vllm_config.model_config.hf_config

        self.embed_tokens = VocabParallelEmbedding(
            hf_config.vocab_size,
            hf_config.hidden_size,
        )
        self.layers = nn.ModuleList([
            nn.Identity() for _ in range(hf_config.num_hidden_layers)
        ])
        self.norm = nn.Identity()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class ChipMyModelForCausalLM(nn.Module, SupportsLoRA):
    packed_modules_mapping = {}

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        hf_config = vllm_config.model_config.hf_config

        self.model = ChipMyModel(vllm_config=vllm_config, prefix=f"{prefix}.model")
        self.lm_head = ColumnParallelLinear(
            hf_config.hidden_size,
            hf_config.vocab_size,
            bias=False,
            prefix=f"{prefix}.lm_head",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return hidden_states
```

这个模板故意很简陋，重点是先分出两层：

- `ChipMyModel`
- `ChipMyModelForCausalLM`

这样后面你补：

- logits processor
- sampler
- spec decode
- LoRA wrapper

都更好扩展。

### 为什么先保留 `ForCausalLM` 外壳

因为对 vLLM 来说，真正接进 loader / registry 的一般是顶层语言模型类，而不是内部 backbone。

这层外壳通常负责：

- 承载 `lm_head`
- 暴露 `load_weights()`
- 暴露平台相关接口

---

## 112. 模板三：最小 `load_weights()` 模板

权重加载模板建议直接写在顶层模型类里。

```python
class ChipMyModelForCausalLM(nn.Module, SupportsLoRA):
    ...

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        name_mapping = {
            "model.embed_tokens.weight": "model.embed_tokens.weight",
            "lm_head.weight": "lm_head.weight",
        }

        for name, loaded_weight in weights:
            target_name = name_mapping.get(name, name)
            if target_name not in params_dict:
                continue

            param = params_dict[target_name]
            if param.shape != loaded_weight.shape:
                raise ValueError(
                    f"Shape mismatch for {target_name}: "
                    f"expected {tuple(param.shape)}, "
                    f"got {tuple(loaded_weight.shape)}"
                )
            param.data.copy_(loaded_weight)
            loaded_params.add(target_name)

        return loaded_params
```

这个模板只做最小的 4 件事：

1. 遍历 checkpoint 权重
2. 先做名字映射
3. 再做 shape 校验
4. 最后 copy 到参数里

这是新芯片团队最稳的起步方式。

### 第一阶段先不要做的事

不要一开始就同时引入：

- fused 参数拆分
- MoE expert 分片
- quantized weight loader
- tensor parallel shard 切分
- packed module 映射

这些都应该等基础模型先能完整 load 成功以后再加。

---

## 113. 模板四：稍微真实一点的权重名映射骨架

真实项目里，checkpoint 参数名和模型类内部参数名通常不会完全一致。  
建议一开始就把“名字映射”单独抽出来。

```python
def _map_weight_name(name: str) -> str | None:
    if name == "embed.weight":
        return "model.embed_tokens.weight"
    if name == "output.weight":
        return "lm_head.weight"
    return name
```

然后在 `load_weights()` 里调：

```python
target_name = _map_weight_name(name)
if target_name is None:
    continue
```

这样后面要补：

- 参数重命名
- packed 权重映射
- 忽略某些参数

都更稳。

### 推荐再加一个“跳过名单”

```python
SKIP_WEIGHTS = {
    "rotary_emb.inv_freq",
}
```

然后在 `load_weights()` 里先过滤：

```python
if name in SKIP_WEIGHTS:
    continue
```

这能避免把很多非持久参数、推导参数或者平台不需要的张量塞进加载逻辑里。

---

## 114. 模板五：最小模型注册表扩展

如果你想对主仓库 `ModelRegistry.register_model()` 做更平滑的覆盖，也可以单独放一个平台文件。

文件：

- `vllm_<chip>/model_executor/models/registry.py`

```python
from vllm.model_executor.models import ModelRegistry


def register_model(model_arch: str, model_cls: str) -> None:
    ModelRegistry.register_model(model_arch, model_cls)
```

第一阶段这个文件可以非常薄。  
它存在的价值主要是为了以后好扩展：

- 平台专属日志
- 重复注册覆盖提示
- 条件注册
- 只在某些模型类型上启用平台特化类

也就是说，先把“模型注册逻辑的扩展点”占住。

---

## 115. 模板六：`ModelRunner.load_model()` 如何与模型注册接线

前面你已经有了：

```python
model_loader = get_model_loader(self.load_config)
self.model = model_loader.load_model(
    vllm_config=self.vllm_config,
    model_config=self.model_config,
)
```

这条链真正生效的前提是：

- 插件初始化时已经执行 `register_models()`
- `architecture` 能解析到你的 `ChipMyModelForCausalLM`

所以这条接线可以理解成：

1. `ModelRunner.load_model()` 不知道具体模型类
2. `model_loader` 从 `model_config.architecture` 取名字
3. `ModelRegistry` 返回你注册的类路径
4. loader 实例化顶层模型类
5. loader 调顶层模型类的 `load_weights()`

只要这条链打通，你的平台就已经具备“主仓库通用加载链路 + 平台自定义模型类”的能力。

---

## 116. 推荐的最小模型接入顺序

建议按下面顺序落地：

1. `register_models()`
2. `ChipMyModel`
3. `ChipMyModelForCausalLM`
4. `_map_weight_name()`
5. `load_weights()`
6. `ModelRunner.load_model()` 验证接线

这个顺序的好处是：

- 先让 loader 能找到类
- 再让类能实例化
- 最后让权重能灌进去

很多团队会反过来先写复杂 `load_weights()`，结果模型类还没稳定，返工很多。

---

## 117. 这一阶段的验收标准

### 第一阶段验收

- `architecture` 能正确解析到平台模型类
- `ModelRunner.load_model()` 能实例化模型对象
- 模型对象能 `to(device)` 并 `eval()`

### 第二阶段验收

- checkpoint 权重能成功遍历
- 至少 embedding 与 `lm_head` 能成功加载
- shape mismatch 能被明确报错

### 第三阶段验收

- 模型 forward 能返回 hidden states
- 执行链能从 `load_model()` 接到 `execute_model()`
- attention 层能正确选到平台 backend

如果做到这里，说明：

- 模型注册
- 模型实例化
- 权重加载
- 执行链接线

这四段已经真正闭环了。

---

## 118. 新芯片团队在模型接入层最容易犯的错

这一层也有几个非常常见的坑。

### 错误 1：`load_weights()` 和模型结构一起改到失控

更稳的做法是：

- 先固定最小模型结构
- 再单独补名字映射
- 最后再加分片、融合和量化加载

### 错误 2：在 `ModelRunner` 里硬编码模型类型

这会绕过主仓库 loader / registry 体系，后面很难维护。

### 错误 3：一开始就做复杂的分布式切分加载

如果单卡加载都还没稳定，就先做 TP / EP shard 加载，问题会非常难定位。

### 错误 4：不提前设计名字映射函数

如果把重命名逻辑直接散在 `load_weights()` 里，随着模型复杂度上涨会很快失控。

建议至少从一开始就分出：

- `_map_weight_name()`
- `SKIP_WEIGHTS`

这两个最小扩展点。

---

## 119. 施工手册：把整套接入按阶段压成可执行清单

到这里，文档里已经有：

- 原理主线
- 源码阅读主线
- 各层代码模板

但项目真正落地时，团队最需要的通常不是“知道有哪些层”，而是：

**“第一周先干什么，第二周接什么，出了问题先查哪一层。”**

所以这一节把整套接入压成一个可施工阶段清单。

推荐总原则：

- 每个阶段只引入一类新复杂度
- 每个阶段都要求可验收
- 单卡闭环优先于多卡
- 普通 attention 优先于 MLA
- 非量化优先于量化

---

## 120. Phase 0：仓库与插件骨架

### 目标

先把平台仓库框架搭出来，让 vLLM 能识别你的平台插件。

### 需要新增的文件

- `vllm_<chip>/__init__.py`
- `vllm_<chip>/platforms/chip.py`
- `vllm_<chip>/mlu_hijack.py` 或对应平台 hijack 文件
- `setup.py` 或 `pyproject.toml` 里的 plugin entry points

### 这一阶段必须完成的内容

- 注册平台 plugin
- 注册 general plugin
- 确保 `current_platform` 能走到你的平台类
- 确保最小导入不会报错

### 依赖关系

- 无

### 验收标准

- 安装插件后，vLLM 启动时能识别到你的平台
- 能打印平台初始化日志
- `get_attn_backend_cls()` 至少能返回类路径字符串

### 调试顺序

1. 先看 plugin 是否被发现
2. 再看平台类是否被导入
3. 再看平台类方法是否被调用

### 典型风险

- entry point 写错
- 包路径写错
- 导入链太重，初始化时就触发底层依赖失败

---

## 121. Phase 1：普通 attention backend 最小闭环

### 目标

让模型里的 `Attention` 层能选中你平台的 backend，并成功走到你的 custom op。

### 需要新增的文件

- `vllm_<chip>/attention/layer.py`
- `vllm_<chip>/v1/attention/backends/flash_attn.py`
- `vllm_<chip>/_chip_ops.py`

### 这一阶段必须完成的内容

- `get_kv_cache_spec()` 最小接入
- backend class
- metadata dataclass
- metadata builder
- impl.forward
- 最小 `reshape_paged_cache`
- 最小 `flash_attention`
- 最小 `single_query_cached_kv_attn`

### 依赖关系

- 依赖 Phase 0 的平台插件成功加载

### 验收标准

- `Attention` 层能拿到你的 backend
- builder 能成功 build metadata
- `impl.forward()` 能进入你的 `_chip_ops`
- prefill-only 至少能跑通最小 case

### 调试顺序

1. 先验证 backend 是否被选中
2. 再验证 builder 是否被调用
3. 再验证 `impl.forward()` 是否进入
4. 最后验证 custom op 输入 shape

### 典型风险

- KV cache shape 和 kernel 预期不一致
- `slot_mapping` 或 `block_table` shape 错误
- prefill/decode 分支被混淆

---

## 122. Phase 2：单卡 `Worker + ModelRunner` 闭环

### 目标

让 `EngineCore.step()` 真的能调用到你的 `Worker` 和 `ModelRunner`，完成一次单卡 forward。

### 需要新增的文件

- `vllm_<chip>/v1/worker/gpu_worker.py`
- `vllm_<chip>/v1/worker/gpu_model_runner.py`
- 可选：`vllm_<chip>/v1/engine/core.py`

### 这一阶段必须完成的内容

- `Worker.init_device()`
- `Worker.execute_model()`
- `ModelRunner.__init__()`
- `ModelRunner.load_model()`
- `ModelRunner._prepare_inputs()`
- `ModelRunner._build_attn_metadata()`
- `ModelRunner.execute_model()`

### 依赖关系

- 依赖 Phase 1 的 backend 已经可用

### 验收标准

- 模型能被加载到设备
- `execute_model()` 能进入 `self.model(...)`
- attention backend 能在真实 forward 中被调用
- 单卡 prefill / decode 都能跑通

### 调试顺序

1. 先看设备初始化
2. 再看模型加载
3. 再看输入准备
4. 再看 attention metadata
5. 最后看 forward 内部的 attention 分支

### 典型风险

- `ModelRunner` 里直接写死模型类型
- 输入 tensor shape 和模型 forward 约定不一致
- `forward_context` 没正确设置，导致 attention 层拿不到 metadata

---

## 123. Phase 3：模型注册、模型类、权重加载闭环

### 目标

让你的平台模型类能通过主仓库 `model_loader` / `ModelRegistry` 被解析、实例化并加载权重。

### 需要新增的文件

- `vllm_<chip>/model_executor/models/__init__.py`
- `vllm_<chip>/model_executor/models/registry.py`
- `vllm_<chip>/model_executor/models/my_model.py`

### 这一阶段必须完成的内容

- `register_models()`
- 顶层 `ForCausalLM` 类
- backbone 类
- `load_weights()`
- `_map_weight_name()`
- `SKIP_WEIGHTS`

### 依赖关系

- 依赖 Phase 2 的 `ModelRunner.load_model()` 已经能走主仓库 loader

### 验收标准

- `architecture` 能映射到平台模型类
- embedding 和 `lm_head` 至少能正确加载
- 模型 forward 能返回 hidden states

### 调试顺序

1. 先看 registry 是否命中
2. 再看模型类能否实例化
3. 再看 checkpoint 参数名是否匹配
4. 最后看 shape mismatch

### 典型风险

- 过早引入复杂的 fused 参数
- `load_weights()` 混入太多模型结构逻辑
- 没有独立名字映射函数，后续维护失控

---

## 124. Phase 4：chunked prefill、MLA、模型特化 attention

### 目标

在普通 attention 跑通之后，再逐步支持：

- chunked prefill
- MLA
- 具体热点模型的特化 attention

### 需要新增或扩展的文件

- `vllm_<chip>/v1/attention/backends/flash_attn.py`
- `vllm_<chip>/v1/attention/backends/mla/flashmla.py`
- `vllm_<chip>/model_executor/models/<hot_model>.py`

### 这一阶段必须完成的内容

- chunk metadata
- prefill/decode 分流
- MLA backend class
- MLA metadata builder
- MLA impl
- 具体模型的特化 attention 逻辑

### 依赖关系

- 依赖普通 attention、普通执行链和普通模型加载都已经稳定

### 验收标准

- chunked prefill 可以正确分流
- MLA 最小 case 跑通
- 特化模型能稳定进入自定义 attention 路径

### 调试顺序

1. 先只开 chunked prefill
2. 再单独验证 MLA
3. 最后再加具体模型特化 attention

### 典型风险

- 还没稳定的普通路径和特化路径混在一起调
- 把模型特有逻辑塞进平台通用 backend
- chunked metadata 设计过晚，导致 impl 逻辑失控

---

## 125. Phase 5：多卡与分布式

### 目标

把单卡跑通的链路扩展到：

- TP
- DP
- EP
- sequence parallel

### 需要新增或扩展的文件

- `vllm_<chip>/distributed/...`
- `vllm_<chip>/v1/worker/gpu_worker.py`
- `vllm_<chip>/model_executor/layers/...`
- `vllm_<chip>/platforms/chip.py`

### 这一阶段必须完成的内容

- 分布式初始化
- 设备 communicator
- collectives 封装
- 并行配置约束
- 特定模型的 shard 加载

### 依赖关系

- 依赖单卡链路稳定
- 依赖模型加载已能在单卡工作

### 验收标准

- 2 卡 TP 最小模型跑通
- 同步通信正确
- attention / `lm_head` 的 shard 结果正确

### 调试顺序

1. 先只做 2 卡 TP
2. 再验证 all-reduce / all-gather
3. 再扩到 EP / DP

### 典型风险

- 单卡路径还不稳就上多卡
- 通信问题和模型问题混在一起排
- 过早引入多种并行模式

---

## 126. Phase 6：图模式、量化、性能优化

### 目标

把“能跑”升级到“跑得快”。

### 需要新增或扩展的文件

- `vllm_<chip>/platforms/chip.py`
- `vllm_<chip>/config/...`
- `vllm_<chip>/v1/attention/backends/...`
- `vllm_<chip>/v1/worker/gpu_model_runner.py`
- profiling / benchmark 文件

### 这一阶段必须完成的内容

- graph capture 批次策略
- chunk/un-chunk 默认策略
- KV cache 量化
- 性能统计
- benchmark 场景开关

### 依赖关系

- 依赖单卡与多卡执行路径都已经稳定

### 验收标准

- graph 模式在最小 batch 和常见 batch 下稳定
- KV cache 量化前后数值可控
- 至少能提供 step latency / forward latency

### 调试顺序

1. 先做 profiling
2. 再做 graph
3. 最后再做 KV cache quant

### 典型风险

- 在功能路径还不稳时过早引入 graph
- 一次同时改 graph、量化、chunked 策略
- 没有稳定 benchmark 基线，优化无法回归

---

## 127. 每个阶段真正的“完成定义”

建议团队不要用“代码已经写了”作为完成定义，而要用下面这种标准。

### 一个阶段完成，至少要满足

- 对应文件已经落地
- 有最小可运行路径
- 有清晰日志可以确认关键链路被调用
- 有最小验收 case
- 已知不支持项被明确写出来

### 一个阶段不算完成的典型信号

- 只能在某个特定模型上偶然跑通
- 没有日志，不知道实际走了哪条路径
- 报错时只能靠猜
- 同时引入了两个以上大复杂度

---

## 128. 团队建议的排工方式

如果是一个 3 到 6 人的小团队，比较稳的排法是：

### 角色 A：平台与执行侧

负责：

- plugin
- platform
- worker
- model runner

### 角色 B：attention 与算子

负责：

- backend
- metadata builder
- impl
- `_chip_ops`

### 角色 C：模型接入

负责：

- registry
- model class
- `load_weights()`

### 角色 D：分布式与性能

在前 3 个角色把单卡跑通后再介入：

- distributed
- graph
- profiling
- benchmark

这种排法的好处是：

- 单卡闭环可以尽早形成
- attention 与模型接入可以并行推进
- 多卡和性能优化不会过早阻塞主线

---

## 129. 最后的总建议：按这条顺序推进最稳

如果把整份文档压成一句最实用的落地顺序，就是：

1. 平台插件先能被识别
2. 普通 attention backend 先跑通
3. 单卡 `Worker + ModelRunner` 先闭环
4. 模型注册和权重加载再接进来
5. chunked prefill / MLA / 模型特化 attention 后补
6. 多卡最后做
7. graph、量化、性能优化放在功能稳定之后

这条顺序基本就是把 `vllm-mlu` 的经验压成了新芯片团队最不容易翻车的工程路径。
