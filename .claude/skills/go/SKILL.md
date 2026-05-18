---
name: go
description: This skill should be used when the user wants to finish a piece of work end-to-end — verify tests pass, simplify the diff, then commit/push/open a PR. Triggers on phrases like "/go", "ship it", "wrap this up", "finish it", or as a suffix on task prompts ("...do the thing /go"). Composes test-and-fix → code-simplifier → commit-push-pr.
allowed-tools: Bash, Read, Glob, Grep, Edit, Write
disable-model-invocation: true
---

Finish the current change end-to-end: verify, simplify, ship.

1. **Verify.** Invoke the `test-and-fix` skill to ensure typecheck/static-analysis + lint + tests are green for whatever toolchain this project uses (read its manifest to find the right commands). If anything is red, the skill fixes it (or fails out — do not proceed to step 2).
2. **Simplify.** Delegate to the `code-simplifier` subagent to review the diff for reuse, quality, and dead code. Apply the recommended simplifications.
3. **Ship.** Invoke the `commit-push-pr` skill to commit, push, and open a PR. The Stop hook re-runs typecheck + lint + tests at completion to confirm step 2's edits didn't break anything. Pass `$ARGUMENTS` through as additional PR description context if the user provided any.

Branch + status snapshot for the run: `!git status --short` — `!git diff --stat`.

If any step fails, stop and report the failure — do not skip to the next step.
