#!/usr/bin/env bash

set -e

# ===== 可配置：提交信息 =====
MAIN_COMMIT_MSG="[fix](tools): fix sh link"
SUB_COMMIT_MSG="[feat](mma): codex gen mma fp4"

# ===== 可配置：子仓目录 =====
SUBMODULE_PATH="work_notes"

# ===== 工具函数：判断当前仓库是否有变动 =====
has_changes() {
  # 有未暂存变动
  ! git diff --quiet || \
  # 有已暂存变动
  ! git diff --cached --quiet || \
  # 有未跟踪文件
  [ -n "$(git ls-files --others --exclude-standard)" ]
}

echo "==> Checking submodule: $SUBMODULE_PATH"

if [ ! -d "$SUBMODULE_PATH/.git" ] && [ ! -f "$SUBMODULE_PATH/.git" ]; then
  echo "Error: $SUBMODULE_PATH does not look like a Git submodule."
  exit 1
fi

# 记录主仓是否需要提交
MAIN_NEEDS_COMMIT=false

# ===== 1. 处理子仓 work_notes =====
cd "$SUBMODULE_PATH"

if has_changes; then
  echo "==> Submodule has changes. Committing..."

  git add .
  git commit -m "$SUB_COMMIT_MSG"
  git push

  MAIN_NEEDS_COMMIT=true
else
  echo "==> No changes in submodule."
fi

cd ..

# ===== 2. 判断主仓是否有变动 =====
# 注意：如果子仓刚刚提交了新 commit，主仓会显示 work_notes 指针变化
if has_changes; then
  MAIN_NEEDS_COMMIT=true
fi

# ===== 3. 处理主仓 my_notes =====
if [ "$MAIN_NEEDS_COMMIT" = true ]; then
  echo "==> Main repo has changes. Committing..."

  git add .
  git commit -m "$MAIN_COMMIT_MSG"
  git push
else
  echo "==> No changes in main repo."
fi

echo "==> Done."