# Review Changes

Review all pending changes for quality before committing.

## Steps

1. Gather all changes:
   ```bash
   git diff --cached
   git diff
   git ls-files --others --exclude-standard
   ```

2. Review each changed file for:
   - **Logic errors** — off-by-one, wrong variable, incorrect condition
   - **Edge cases** — empty inputs, None values, boundary conditions
   - **Style violations** — naming, imports, magic numbers (per CLAUDE.md)
   - **Missing tests** — new logic without corresponding test coverage
   - **Security** — hardcoded secrets, unsafe deserialization, path traversal

3. Report findings grouped by severity (blocking / warning / nit).
