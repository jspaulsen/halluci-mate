---
name: pr-conventions
description: This skill should be used when the user is writing a commit message, opening a pull request, or asking about commit/PR conventions for this repo. Triggers on phrases like "commit message", "PR title", "conventional commit", "open a PR", "PR description". Enforces conventional-commits format and one-logical-change-per-PR policy.
allowed-tools: Read, Glob, Grep, Bash(git log*), Bash(git diff*)
---

## Commit messages (Conventional Commits)
Format: `<type>(<scope>): <short description>`

Types: `feat`, `fix`, `refactor`, `test`, `chore`, `docs`, `perf`.

- Subject line: imperative mood, lowercase after the colon, no trailing period, ≤72 chars.
- Body (optional): explain *why*, not *what*. Reference issue numbers with `Closes #123`.

## PR policy
- One logical change per PR. Split unrelated changes into separate PRs.
- PR title = commit message subject (will be squash-merged).
- Description must include: what changed, why, and how to test it.
- All PRs require passing CI (typecheck + lint + tests) before merge.
- Squash merge only. No merge commits.
