#!/usr/bin/env bash

set -euo pipefail

# ===== 可配置：子仓列表 =====
SUBMODULES=(
  "work_notes"
  "secrets"
)

# ===== 可配置：默认分支 =====
DEFAULT_BRANCH="main"

# ===== 使用说明 =====
usage() {
  echo "Usage: $0"
  echo ""
  echo "  一键同步 my_notes 主仓及其子仓。"
  echo "  推荐工作流：先运行本脚本，再修改文件，最后运行 commit_all.sh。"
  echo "  若已有本地修改，会先临时 stash，同步后恢复；冲突时停下由人工处理。"
  echo ""
  echo "  -h              显示帮助"
  echo ""
  echo "  子仓: ${SUBMODULES[*]}"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h) usage ;;
    *)
      echo "[错误] 未知参数: $1" >&2
      usage ;;
  esac
done

# ===== 工具函数 =====
die() {
  echo "[错误] $*" >&2
  exit 1
}

has_changes() {
  ! git diff --quiet || \
  ! git diff --cached --quiet || \
  [ -n "$(git ls-files --others --exclude-standard)" ]
}

stash_if_needed() {
  local reason="$1"
  local before_stash
  local after_stash
  if has_changes; then
    echo "  -> 暂存当前修改，准备同步 $reason..."
    git status --short
    before_stash="$(git rev-parse -q --verify refs/stash 2>/dev/null || true)"
    git stash push -u -m "pull_all: temporary stash before $reason" >/dev/null
    after_stash="$(git rev-parse -q --verify refs/stash 2>/dev/null || true)"
    if [ "$before_stash" != "$after_stash" ]; then
      return 0
    fi
  fi
  return 1
}

restore_stash() {
  echo "  -> 恢复暂存修改..."
  if ! git stash pop --quiet; then
    die "恢复 stash 时产生冲突，请在 $(pwd) 解决冲突后继续"
  fi
}

sync_branch() {
  local branch="$1"
  local stashed=false

  if stash_if_needed "origin/$branch"; then
    stashed=true
  fi

  echo "  -> fetch origin..."
  git fetch origin

  if git show-ref --verify --quiet "refs/heads/$branch"; then
    git switch "$branch"
  elif git show-ref --verify --quiet "refs/remotes/origin/$branch"; then
    git switch -c "$branch" "origin/$branch"
  else
    die "找不到本地或远端分支 $branch"
  fi

  if git show-ref --verify --quiet "refs/remotes/origin/$branch"; then
    git branch --set-upstream-to="origin/$branch" "$branch" >/dev/null 2>&1 || true
    if ! git merge --ff-only "origin/$branch"; then
      if [ "$stashed" = true ]; then
        restore_stash
      fi
      die "$branch 无法快进到 origin/$branch，请先手动处理分叉后重试"
    fi
  else
    echo "  [提示] origin/$branch 不存在，跳过远端快进。"
  fi

  if [ "$stashed" = true ]; then
    restore_stash
  fi
}

sync_current_branch() {
  local branch
  local upstream
  local stashed=false

  branch="$(git symbolic-ref --quiet --short HEAD)" || die "主仓当前处于 detached HEAD，请先切到要同步的分支"
  upstream="$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null || true)"

  if [ -z "$upstream" ]; then
    echo "  [提示] 主仓 $branch 没有 upstream，跳过主仓 pull。"
    return
  fi

  if stash_if_needed "$upstream"; then
    stashed=true
  fi

  echo "  -> fetch ${upstream%%/*}..."
  git fetch "${upstream%%/*}"
  if ! git merge --ff-only "$upstream"; then
    if [ "$stashed" = true ]; then
      restore_stash
    fi
    die "主仓 $branch 无法快进到 $upstream，请先手动处理分叉后重试"
  fi

  if [ "$stashed" = true ]; then
    restore_stash
  fi
}

# ===== 主流程 =====
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "$ROOT_DIR 不是 git 仓库"

echo "=========================================="
echo "  my_notes 一键同步"
echo "=========================================="

echo ""
echo "==> 主仓 (my_notes)"
sync_current_branch

for path in "${SUBMODULES[@]}"; do
  echo ""
  echo "==> 子仓: $path"

  if [ ! -d "$path/.git" ] && [ ! -f "$path/.git" ]; then
    echo "  [跳过] 不是 git 子仓"
    continue
  fi

  pushd "$path" > /dev/null
  sync_branch "$DEFAULT_BRANCH"
  popd > /dev/null
done

echo ""
echo "==> 全部同步完成。"
