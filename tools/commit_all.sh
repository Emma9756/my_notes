#!/usr/bin/env bash

set -euo pipefail

# ===== 可配置：子仓列表 =====
# 格式: "path|default_commit_msg"
SUBMODULES=(
  "work_notes|[feat](common): codex gen core & me update gmm"
  "secrets|[feat](common): codex refactor 3 score & bali & eye & wechat draft"
)

# ===== 可配置：主仓默认提交信息 =====
MAIN_DEFAULT_MSG="[fix](tools): codex update commit sh for sub git main"
# MAIN_DEFAULT_MSG="[sync](sub): sync sub commit"

# ===== 可配置：默认分支 =====
DEFAULT_BRANCH="main"

# ===== 使用说明 =====
usage() {
  echo "Usage: $0 [-m <main commit msg>] [-s <submodule> <msg>]..."
  echo ""
  echo "  一键提交 my_notes 主仓及其子仓的变动。"
  echo "  自动检测各子仓是否有未提交变更，分别提交后更新主仓指针。"
  echo ""
  echo "  -m <msg>        主仓提交信息（覆盖默认）"
  echo "  -s <name> <msg> 指定某个子仓的提交信息"
  echo "  -h              显示帮助"
  echo ""
  echo "  子仓: $(printf '%s ' "${SUBMODULES[@]}" | sed 's/|[^ ]*//g')"
  echo ""
  echo "  示例:"
  echo "    $0                          # 自动检测，使用默认信息提交"
  echo "    $0 -m '[fix] urgent fix'    # 自定义主仓信息"
  echo "    $0 -s secrets '[fix](fitness): correct W3 scores'"
  exit 0
}

# ===== 解析参数 =====
MAIN_MSG="$MAIN_DEFAULT_MSG"
declare -A SUB_MSGS

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h) usage ;;
    -m)
      if [[ $# -lt 2 ]]; then
        echo "[错误] -m 需要提交信息" >&2
        exit 1
      fi
      MAIN_MSG="$2"
      shift 2 ;;
    -s)
      if [[ $# -lt 3 ]]; then
        echo "[错误] -s 需要子仓名和提交信息" >&2
        exit 1
      fi
      key="$2"
      SUB_MSGS["$key"]="$3"
      shift 3 ;;
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
  # 有未暂存变动
  ! git diff --quiet || \
  # 有已暂存变动
  ! git diff --cached --quiet || \
  # 有未跟踪文件
  [ -n "$(git ls-files --others --exclude-standard)" ]
}

stash_if_needed() {
  local reason="$1"
  local before_stash
  local after_stash
  if has_changes; then
    echo "  -> 暂存当前修改，准备同步 $reason..."
    before_stash="$(git rev-parse -q --verify refs/stash 2>/dev/null || true)"
    git stash push -u -m "commit_all: temporary stash before $reason" >/dev/null
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
    die "恢复 stash 失败，请在 $(pwd) 解决冲突后重试"
  fi
}

sync_branch_for_commit() {
  local branch="$1"
  local stashed=false

  if stash_if_needed "$branch"; then
    stashed=true
  fi

  echo "  -> 同步 origin/$branch..."
  git fetch origin

  if git show-ref --verify --quiet "refs/heads/$branch"; then
    git switch "$branch"
  elif git show-ref --verify --quiet "refs/remotes/origin/$branch"; then
    git switch -c "$branch" "origin/$branch"
  else
    git switch -c "$branch"
  fi

  if git show-ref --verify --quiet "refs/remotes/origin/$branch"; then
    git branch --set-upstream-to="origin/$branch" "$branch" >/dev/null 2>&1 || true
    if ! git merge --ff-only "origin/$branch"; then
      if [ "$stashed" = true ]; then
        echo "  [提示] 修改仍保存在当前仓库的最新 stash 中。" >&2
      fi
      die "$branch 无法快进到 origin/$branch，请先手动处理分叉后重试"
    fi
  fi

  if [ "$stashed" = true ]; then
    restore_stash
  fi
}

sync_current_branch_with_upstream() {
  local branch
  local upstream
  local stashed=false

  branch="$(git symbolic-ref --quiet --short HEAD)" || die "主仓当前处于 detached HEAD，请先切到要提交的分支"
  upstream="$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null || true)"

  if [ -z "$upstream" ]; then
    echo "  [提示] 主仓 $branch 没有 upstream，跳过快进同步。"
    return
  fi

  if stash_if_needed "$upstream"; then
    stashed=true
  fi

  echo "  -> 同步 $upstream..."
  git fetch "${upstream%%/*}"
  if ! git merge --ff-only "$upstream"; then
    if [ "$stashed" = true ]; then
      echo "  [提示] 修改仍保存在主仓的最新 stash 中。" >&2
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
echo "  my_notes 一键提交"
echo "=========================================="

MAIN_NEEDS_COMMIT=false

# 先检查主仓本身的非子仓文件是否有变更
if has_changes; then
  MAIN_NEEDS_COMMIT=true
fi

# 处理每个子仓
for entry in "${SUBMODULES[@]}"; do
  path="${entry%%|*}"
  default_msg="${entry#*|}"

  echo ""
  echo "==> 子仓: $path"

  if [ ! -d "$path/.git" ] && [ ! -f "$path/.git" ]; then
    echo "  [跳过] 不是 git 子仓"
    continue
  fi

  pushd "$path" > /dev/null

  if has_changes; then
    sync_branch_for_commit "$DEFAULT_BRANCH"
    msg="${SUB_MSGS[$path]:-$default_msg}"

    echo "  变更内容:"
    git status --short
    echo ""

    git add -A
    git commit -m "$msg"
    git push -u origin "HEAD:$DEFAULT_BRANCH"

    echo "  [已提交] $msg"
    MAIN_NEEDS_COMMIT=true
  else
    echo "  [无变更]"
  fi

  popd > /dev/null
done

# ===== 处理主仓 =====
echo ""

if [ "$MAIN_NEEDS_COMMIT" = true ]; then
  echo "==> 主仓 (my_notes)"

  sync_current_branch_with_upstream

  echo "  变更内容:"
  git status --short
  echo ""

  git add -A
  git commit -m "$MAIN_MSG"
  git push

  echo "  [已提交] $MAIN_MSG"
else
  echo "==> 主仓无变更，跳过。"
fi

echo ""
echo "==> 全部完成。"
