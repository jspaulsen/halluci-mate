#!/usr/bin/env bash
set -euo pipefail

# Statusline for Claude Code. Reads JSON session data on stdin (schema:
# https://code.claude.com/docs/en/statusline) and prints one line:
#   <model> | <used>/<total> (<pct>%) | <branch>[ @<worktree>]
#
# Color tiers are percent-based so they stay meaningful across context window
# sizes. They map onto CLAUDE_CODE_AUTO_COMPACT_WINDOW=400000 (set in
# template/.claude/settings.json env block) for the 1M-token Opus 4.7 default:
#   green   < 10%  comfortable                              (< 100k @ 1M)
#   yellow  >=10%  non-trivial; think about scope           (>=100k @ 1M)
#   orange  >=20%  ramping up; auto-compact ~200k away      (>=200k @ 1M)
#   red     >=30%  act now — auto-compact ~100k away        (>=300k @ 1M)
# After auto-compact fires (at 400k by default) usage drops back to green.
# On a 200k-context model these scale to 20k / 40k / 60k.
# Surface context fill so you know when to /clear.

input=$(cat)

# Graceful fallback if jq is missing — don't crash the harness.
if ! command -v jq >/dev/null 2>&1; then
	printf '[no jq] | %s\n' "$(pwd)"
	exit 0
fi

# Single jq invocation; newline-separated fields read into a bash array.
mapfile -t fields < <(printf '%s' "$input" | jq -r '
	.model.display_name // "unknown",
	.context_window.total_input_tokens // 0,
	.context_window.context_window_size // 200000,
	(.context_window.used_percentage // 0),
	.workspace.git_worktree // "",
	.workspace.current_dir // "."
')

model="${fields[0]}"
used="${fields[1]}"
total="${fields[2]}"
pct_float="${fields[3]}"
worktree="${fields[4]}"
cwd="${fields[5]}"

# Integer percent for display and threshold comparison.
pct="${pct_float%.*}"
[[ -z "$pct" ]] && pct=0

# Real git branch when available; else worktree name; else cwd basename.
branch=""
if command -v git >/dev/null 2>&1 && git -C "$cwd" rev-parse --git-dir >/dev/null 2>&1; then
	branch=$(git -C "$cwd" rev-parse --abbrev-ref HEAD 2>/dev/null || true)
fi
if [[ -z "$branch" ]]; then
	branch="${worktree:-${cwd##*/}}"
fi

# Worktree marker: " @<name>" suffix only when in a linked worktree and the
# branch name doesn't already match (avoids "feature @feature" duplication).
suffix=""
if [[ -n "$worktree" && "$branch" != "$worktree" ]]; then
	suffix=" @${worktree}"
fi

# ANSI color tiers. 256-color 208 = orange; widely supported in modern terminals.
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
ORANGE=$'\033[38;5;208m'
RED=$'\033[31m'
RESET=$'\033[0m'

if   (( pct >= 30 )); then color="$RED"
elif (( pct >= 20 )); then color="$ORANGE"
elif (( pct >= 10 )); then color="$YELLOW"
else                       color="$GREEN"
fi

printf '%s | %s%s/%s (%s%%)%s | %s%s\n' \
	"$model" "$color" "$used" "$total" "$pct" "$RESET" "$branch" "$suffix"
