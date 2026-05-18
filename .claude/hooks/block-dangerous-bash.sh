#!/usr/bin/env bash
# Hook contract: receives tool input JSON on stdin (Bash command at .tool_input.command);
# block by printing reason to STDERR and exiting 2 — exit 1 is silent.
#
# Policy: rules live in dangerous-bash-policy.json (sibling). To add a rule, edit JSON;
# do not edit this script. Each rule has a stable {id, pattern, reason}.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_FILE="$SCRIPT_DIR/dangerous-bash-policy.json"

# Fail closed if the policy file is missing or unreadable.
if [[ ! -r "$POLICY_FILE" ]]; then
  printf 'Blocked by .claude/hooks/block-dangerous-bash.sh\n  Reason:  Policy file not readable: %s\n  Fix:     Restore template/.claude/hooks/dangerous-bash-policy.json from version control\n' "$POLICY_FILE" >&2
  exit 2
fi

# Read the proposed bash command from the JSON payload on stdin.
payload="$(cat)"
cmd="$(printf '%s' "$payload" | jq -er '.tool_input.command' 2>/dev/null || true)"

if [[ -z "$cmd" || "$cmd" == "null" ]]; then
  # Not a Bash invocation we can inspect — let it through.
  exit 0
fi

# Iterate rules from the policy file. Tab-separated to avoid collision with
# regex metachars in pattern; reasons must not contain tabs.
while IFS=$'\t' read -r id pattern reason; do
  if [[ -z "$id" ]]; then
    continue
  fi
  if [[ "$cmd" =~ $pattern ]]; then
    printf 'Blocked by .claude/hooks/block-dangerous-bash.sh [%s]\n  Command: %s\n  Reason:  %s\n' "$id" "$cmd" "$reason" >&2
    exit 2
  fi
done < <(jq -r '.rules[] | [.id, .pattern, .reason] | @tsv' "$POLICY_FILE")

exit 0
