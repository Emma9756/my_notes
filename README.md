---
date: 2026-05-28
tags:
  - #索引
  - #README
  - #from_me
  - #待整理
status: 更新中
---

## 总览

小张的个人笔记与脚本库，通过 Obsidian + Claude Code Agent 协同使用。

```
my_notes/
├── secrets/          ← private 子仓：健身数据、生活记录
├── finance/          # 理财：A股/港股/黄金/美股/日韩
│   └── nasdaq/       # 纳指基金对比
├── llm/              # 学习：CUDA 算子 → DeepSeek 架构 → 推理框架
│   ├── cuda/
│   ├── ops/
│   └── deepseek_v4/
├── work_notes/       ← private 子仓
├── tools/            # 脚本 & 方法论
│   └── obisidian/    # Obsidian 使用笔记
└── templates/        # 笔记模板
```

## 标签体系

所有笔记统一在 frontmatter 中使用以下标签，便于 Obsidian 图谱和数据视图多角度检索：

| 维度 | 标签 | 说明 |
|------|------|------|
| **领域** | `#理财` `#健身` `#LLM` `#cuda` `#推理` | 笔记属于哪个知识领域 |
| **来源** | `#from_gpt` `#from_ds` `#from_me` | 内容来源：GPT / DeepSeek / 自己写的 |
| **类型** | `#笔记` `#脚本` `#模板` `#复盘` `#对比` | 内容的形态 |
| **状态** | `#更新中` `#已完成` `#待整理` | 完善程度 |

## 工作流

```
记录 (add.py / 手动)  →  CSV/笔记  →  复盘 (review.py)  →  提交 (commit_all.sh)
```

## 子仓

```bash
# 克隆时初始化子仓
git submodule update --init --recursive

# 一键提交（含子仓检测）
sh tools/commit_all.sh
```
