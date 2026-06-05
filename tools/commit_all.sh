#!/usr/bin/env bash

set -e

# ===== 可配置：子仓列表 =====
# 格式: "path|default_commit_msg"
SUBMODULES=(
  "work_notes|[fix](arch): codex update queue"
  "secrets|[fix](git): me update fit"
)

# ===== 可配置：主仓默认提交信息 =====
MAIN_DEFAULT_MSG="[sync](sub): sync sub commit"

# ===== 可配置：默认分支 =====
DEFAULT_BRANCH="main"

# ===== 使用说明 =====
usage() {
  echo "Usage: $0 [-m <main commit msg>] [-s <submodule> -m <msg>]..."
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
      MAIN_MSG="$2"
      shift 2 ;;
    -s)
      key="$2"
      SUB_MSGS["$key"]="$3"
      shift 3 ;;
    *) shift ;;
  esac
done

# ===== 工具函数 =====
has_changes() {
  # 有未暂存变动
  ! git diff --quiet || \
  # 有已暂存变动
  ! git diff --cached --quiet || \
  # 有未跟踪文件
  [ -n "$(git ls-files --others --exclude-standard)" ]
}

ensure_branch() {
  local branch="$1"
  if ! git symbolic-ref --quiet --short HEAD >/dev/null; then
    echo "  -> detached HEAD, switching to $branch..."
    git fetch origin 2>/dev/null || true
    git switch "$branch" 2>/dev/null || git switch -c "$branch" "origin/$branch" 2>/dev/null || true
  fi
}

# ===== 主流程 =====
echo "=========================================="
echo "  my_notes 一键提交"
echo "=========================================="

MAIN_NEEDS_COMMIT=false
ROOT_DIR="$(pwd)"

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

  ensure_branch "$DEFAULT_BRANCH"

  if has_changes; then
    msg="${SUB_MSGS[$path]:-$default_msg}"

    echo "  变更内容:"
    git status --short
    echo ""

    git add .
    git commit -m "$msg"
    git push origin "$DEFAULT_BRANCH" 2>/dev/null || git push origin HEAD

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

  echo "  变更内容:"
  git status --short
  echo ""

  git add .
  git commit -m "$MAIN_MSG"
  git push

  echo "  [已提交] $MAIN_MSG"
else
  echo "==> 主仓无变更，跳过。"
fi

echo ""
echo "==> 全部完成。"
