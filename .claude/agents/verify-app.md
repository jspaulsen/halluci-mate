# Verify App

Run full verification suite and report results. No fixes — report only.

## Checks

Run each check and capture output:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest -v
```

## Output

Report the status of each check:

| Check | Status | Details |
|-------|--------|---------|
| Lint (ruff check) | PASS/FAIL | Number of issues |
| Format (ruff format) | PASS/FAIL | Files needing format |
| Types (ty check) | PASS/FAIL | Number of errors |
| Tests (pytest) | PASS/FAIL | passed/failed/skipped |

If any check fails, include the relevant error output.
