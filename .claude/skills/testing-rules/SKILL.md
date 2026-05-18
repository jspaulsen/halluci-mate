---
name: testing-rules
description: This skill should be used when the user is writing, reviewing, or debugging tests for this repo. Triggers on phrases like "write a test", "add a test", "unit test", "integration test", "fix the test", "what should I mock". Enforces file naming, mock policy, and the project's assertion library.
allowed-tools: Read, Glob, Grep, Bash
paths:
  - "tests/**"
  - "**/*_test.py"
  - "conftest.py"
---

The concrete conventions for this repo — test-file naming, what to mock, the assertion library — live in `CLAUDE.md` and `.claude/rules/`; follow those. The principles below apply on top of them.

## File naming

- Mirror the source layout; keep unit and integration tiers in distinct, discoverable locations per this repo's conventions (see `.claude/rules/`).
- Keep test helper utilities out of the test-discovery glob (e.g., a `helpers/` subdir excluded from the pattern).

## What to mock vs. what to test real

- **Mock:** external network calls (databases, third-party APIs, Slack, payment gateways).
- **Do NOT mock:** pure functions, internal modules, type/utility code.
- Integration tests use real local fixtures (in-memory DB, ephemeral container, fake HTTP server) — never a production or staging service.

## Assertion style

- Use the project's primary assertion library (look it up in the manifest if uncertain). Don't add a second one.
- A test that always passes is worse than no test. If you can't think of a way it could fail, the test is wrong.

## End-to-end test sizing

Default to **3 end-to-end tests per feature**: one happy path, two error paths. Add more only if a state machine or invariant genuinely requires it. The goal is a verification loop tight enough to catch regressions without the 30-test trap (an agent can write 30 redundant e2e tests as fast as 3 useful ones).

This is sizing for the e2e tier specifically. Unit tests follow normal coverage discipline; integration tests scale with the number of integration points.

Source: Erik Schluntz, vibe-coding masterclass (translated).

## Iterative scaling (1 → 10 → N)

For batch-fixing or batch-generating tests across many files, scale up in stages:
1. Run on **one** file. Read the diff carefully. Fix the prompt or the approach.
2. Run on **~10** files. Look for failure patterns. Fix again.
3. Only then run on **N**. The first two stages catch ~80% of issues at <1% of cost.

This applies to any agent loop touching ≥10 files (test backfills, codemod-style refactors, doc regenerations). Pair with `--max-turns` and `--allowedTools` from `code.claude.com/docs/en/cli-reference` to scope each invocation.

Source: Cat Wu, latent.space podcast (Apr 2026).
