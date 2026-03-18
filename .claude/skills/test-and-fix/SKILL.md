# Test and Fix

Run all checks and fix failures iteratively until everything passes.

## Steps

1. Run the full check suite:
   ```bash
   uv run pytest --tb=short
   uv run ty check
   uv run ruff check .
   ```

2. If any check fails:
   - Read the error output carefully
   - Fix the **implementation**, not the tests (unless the test itself is wrong)
   - Re-run only the failing check to verify the fix

3. Repeat until all three checks pass with zero errors.

4. Report final status of each check.
