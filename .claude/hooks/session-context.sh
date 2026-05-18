#!/usr/bin/env bash
# Hook contract: stdout is injected as context at session start.

set -u

branch="$(git branch --show-current 2>/dev/null || echo '<not a git repo>')"
last="$(git log --oneline -1 2>/dev/null || echo '')"

worktree=""
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  toplevel="$(git rev-parse --show-toplevel)"
  if git rev-parse --git-common-dir 2>/dev/null | grep -qE '\.git/worktrees/'; then
    worktree=" (worktree: $(basename "$toplevel"))"
  fi
fi

printf 'Session context — branch: %s%s\n  Last commit: %s\n' "$branch" "$worktree" "$last"
