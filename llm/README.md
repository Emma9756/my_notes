---
date: 2026-05-28
tags:
  - #LLM
  - #README
  - #from_me
  - #待整理
status: 更新中
---

## LLM 学习笔记

围绕 GPU 算子开发 → 大模型架构 → 推理部署的学习路径。

### 目录

```
llm/
├── cuda/
│   └── cuda_basics.md          # CUDA 编程基础概念
├── ops/
│   ├── mm/hold_abc.md          # 矩阵乘优化笔记
│   └── fla/fp8_fp4_paged_mqa_logits.md  # FlashAttention / FP8-FP4 / paged MQA
└── deepseek_v4/
    ├── source_code/            # DeepSeek V4 源码阅读
    │   ├── encoding/           # tokenizer & encoding
    │   ├── inference/          # generate / model / kernel / convert
    │   └── docs/               # 架构分析文档（14篇）
    └── (笔记)
```

### 学习路线

```
CUDA 基础 → 矩阵乘优化(GEMM) → FlashAttention → DeepSeek 架构 → vLLM 推理
```

### 关键标签

`#cuda` `#triton` `#flash_attention` `#fp8` `#fp4` `#deepseek` `#vllm` `#推理` `#从源码阅读`

### 笔记规范

- 源码阅读笔记放在对应 `source_code/` 下的子目录
- 分析/总结类笔记统一用 Obsidian frontmatter（title/date/tags/status）
- 涉及性能数据的标注 benchmark 来源和复现条件
