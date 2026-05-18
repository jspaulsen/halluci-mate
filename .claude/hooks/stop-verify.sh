#!/usr/bin/env bash
# Hook contract: stdin payload has .stop_hook_active.
# Exit 0 = allow stop, exit 2 = force continue (stderr -> Claude, next turn).
# Disable per-session: export CLAUDE_SKIP_STOP_VERIFY=1 in the launching shell.
#
# Verification commands come from CLAUDE_VERIFY_{TYPECHECK,LINT,TEST}_CMD in the
# checked-in .claude/settings.json env block (filled by /init-claude-tooling per
# stack). Each optional; all-unset = no-op. settings.local.json may override
# per-machine (Claude Code merges local over shared).

set -u

payload="$(cat)"
already_active="$(printf '%s' "$payload" | jq -r '.stop_hook_active // false' 2>/dev/null || echo false)"

# Short-circuits.
if [[ "$already_active" == "true" ]]; then
  exit 0
fi
if [[ "${CLAUDE_SKIP_STOP_VERIFY:-0}" == "1" ]]; then
  exit 0
fi

project_dir="${CLAUDE_PROJECT_DIR:-$PWD}"
cd "$project_dir" || exit 0

typecheck_cmd="${CLAUDE_VERIFY_TYPECHECK_CMD:-}"
lint_cmd="${CLAUDE_VERIFY_LINT_CMD:-}"
test_cmd="${CLAUDE_VERIFY_TEST_CMD:-}"

if [[ -z "$typecheck_cmd" && -z "$lint_cmd" && -z "$test_cmd" ]]; then
  printf 'stop-verify: no CLAUDE_VERIFY_*_CMD env vars set — skipping. See %s for setup.\n' \
    "${0}" >&2
  exit 0
fi

run_check() {
  local label="$1" cmd="$2"
  [[ -z "$cmd" ]] && return 0
  local output
  if ! output="$(bash -c "$cmd" 2>&1)"; then
    # MESSAGE AUDIENCE: this stderr is delivered to Claude (not the user) via the
    # Stop hook contract — exit 2 forces continuation and surfaces the message in the
    # next turn's context, where Claude can act on it (e.g. auto-invoke /test-and-fix).
    # The wording is therefore an instruction to Claude, not the user.
    printf 'Stop hook: %s is red (command: %s). Fix before declaring done.\n\n%s\n' \
      "$label" "$cmd" "$output" >&2
    exit 2
  fi
}

run_check "typecheck/static-analysis" "$typecheck_cmd"
run_check "lint"                      "$lint_cmd"
run_check "tests"                     "$test_cmd"

exit 0
