#!/usr/bin/env bash

# 一键将 MemReason 的代码和未被 .gitignore 忽略的结果同步到 GitHub。
# 用法：
#   ./sync_git.sh
#   ./sync_git.sh "本次更新说明"

set -Eeuo pipefail

readonly REPO_DIR="/home_data/home/zhouwl2023/project/MemReason"
readonly REMOTE_NAME="origin"
readonly EXPECTED_REPO="internbob/MemReason"
readonly EXPECTED_REMOTE_URL="ssh://git@ssh.github.com:443/${EXPECTED_REPO}.git"
readonly MAX_FILE_BYTES=$((95 * 1024 * 1024))

die() {
    printf '错误：%s\n' "$*" >&2
    exit 1
}

command -v git >/dev/null 2>&1 || die "没有找到 git 命令。"
[[ -d "$REPO_DIR/.git" ]] || die "$REPO_DIR 不是 Git 仓库。"

cd "$REPO_DIR"

# 防止两个同步任务同时修改仓库。
lock_dir=".git/memreason-sync.lock"
if ! mkdir "$lock_dir" 2>/dev/null; then
    die "另一个同步任务似乎正在运行。若确认没有，请删除 $REPO_DIR/$lock_dir 后重试。"
fi
trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT

branch="$(git symbolic-ref --quiet --short HEAD)" \
    || die "当前处于 detached HEAD 状态，请先切换到 main 分支。"

if remote_url="$(git config --get "remote.${REMOTE_NAME}.url")"; then
    case "$remote_url" in
        "https://github.com/${EXPECTED_REPO}"|"https://github.com/${EXPECTED_REPO}.git"|\
        "git@github.com:${EXPECTED_REPO}.git"|"ssh://git@ssh.github.com:443/${EXPECTED_REPO}.git")
            ;;
        *)
            die "origin 当前指向 $remote_url，而不是 GitHub 仓库 ${EXPECTED_REPO}。"
            ;;
    esac
else
    git remote add "$REMOTE_NAME" "$EXPECTED_REMOTE_URL"
fi

# GitHub 拒绝大于 100 MiB 的普通 Git 文件；提前报错，避免提交后才失败。
while IFS= read -r -d '' file; do
    [[ -f "$file" ]] || continue
    size="$(stat -c '%s' -- "$file")"
    if (( size > MAX_FILE_BYTES )); then
        die "文件 '$file' 大于 95 MiB。请将它加入 .gitignore，或改用 Git LFS。"
    fi
done < <(git ls-files --others --modified --exclude-standard -z)

printf '正在整理本地更新（分支：%s）...\n' "$branch"
git add -A

if ! git diff --cached --quiet; then
    if (($# > 0)); then
        commit_message="$*"
    else
        commit_message="sync: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    fi

    git commit -m "$commit_message"
else
    printf '本地没有需要提交的新改动。\n'
fi

printf '正在获取远端更新...\n'
git fetch "$REMOTE_NAME"

if git show-ref --verify --quiet "refs/remotes/${REMOTE_NAME}/${branch}"; then
    printf '正在把本地提交衔接到远端最新版本...\n'
    if ! git rebase "${REMOTE_NAME}/${branch}"; then
        git rebase --abort >/dev/null 2>&1 || true
        die "本地与远端存在冲突，已取消 rebase；你的本地提交仍然保留，请手动解决冲突后重试。"
    fi
fi

printf '正在推送到 GitHub...\n'
git push --set-upstream "$REMOTE_NAME" "HEAD:${branch}"

printf '\n同步完成：%s -> %s/%s\n' "$branch" "$EXPECTED_REPO" "$branch"
