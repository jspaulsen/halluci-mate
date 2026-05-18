---
name: review-changes
description: This skill should be used when the user asks to review uncommitted working-tree changes before commit or push. Triggers on phrases like "review my changes", "check the diff", "review before commit", "what's in this diff". Reports findings grouped by severity.
allowed-tools: Bash(git diff*), Bash(git status*), Read, Glob, Grep
---

Review all uncommitted changes in the working directory.

1. Check what changed: `!git diff --stat`
2. For each changed file, review the diff for:
   - Logic errors or edge cases
   - Style violations against `.claude/rules/code-style.md`
   - Module boundary violations against `.claude/rules/architecture.md`
   - Missing or broken tests
   - Security concerns (credential leaks, injection, unvalidated input)
3. Summarize findings grouped by severity: blocking, should-fix, nit.
