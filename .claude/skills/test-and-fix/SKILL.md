---
name: test-and-fix
description: This skill should be used when the user wants to run the test suite and iteratively fix failures. Triggers on phrases like "run the tests", "fix the failing tests", "make tests pass", "tests are red", or after edits when verification is needed. Fixes implementations, not tests, unless tests are wrong.
allowed-tools: Bash, Read, Edit, Glob, Grep
---

Run the test suite and fix any failures.

1. Resolve the test command: use `$CLAUDE_VERIFY_TEST_CMD` if set (the same command the Stop hook runs), else discover it from the project manifest. If the repo has no automated test framework, its "tests" are whatever the verification surface defines (`CLAUDE_VERIFY_*`, lint, schema/parse checks, smoke scripts) — run those instead and apply the same fix-the-implementation discipline below.
2. Run the tests.
3. If tests pass, report success and stop.
4. If tests fail, for each failure:
   - Read the failing test to understand intent
   - Read the implementation under test
   - Attempt one fix to the **implementation**, not the test (unless the test is wrong — surface it for human review if so)
   - Re-run tests to confirm the fix
   - If the fix doesn't stick, OR the failure spans multiple components (CI → build → signing, API → service → database, etc.), invoke `/systematic-debugging` for root-cause analysis before further attempts
5. Repeat until all tests pass or you've identified a test that needs human review.
