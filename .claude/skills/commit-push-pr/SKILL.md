---
name: commit-push-pr
description: This skill should be used when the user wants to commit, push, and open a pull request for the current working-tree changes. Triggers on phrases like "commit and push", "open a PR", "ship this", "make a pull request". Uses conventional-commits format.
allowed-tools: Bash(git*), Bash(gh pr *), Bash(gh issue *), Read, Glob
disable-model-invocation: true
---

Commit, push, and open a PR for the current changes.

1. Stage changes
2. Write a conventional commit message based on the staged diff: `!git diff --cached --stat` — consult `/pr-conventions` for subject/body format and the one-logical-change-per-PR policy
3. Commit with that message
4. Push the current branch
5. Open a PR with `gh pr create` using the commit message subject as the title
6. If `$ARGUMENTS` is provided, use it as additional PR description context

Current branch: `!git branch --show-current`
Recent commits for context: `!git log --oneline -5`
