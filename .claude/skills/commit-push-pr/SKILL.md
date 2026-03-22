---
disable-model-invocation: true
---

# Commit, Push, and PR

Create a clean commit, push, and optionally open a PR.

## Steps

1. Run all checks first:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run ty check
   uv run pytest --tb=short
   ```
   Stop if any check fails.

2. Stage specific files (never `git add -A`):
   ```bash
   git add <file1> <file2> ...
   ```

3. Commit with conventional commit format:
   ```bash
   git commit -m "<type>(<scope>): <description>"
   ```

4. Push to remote:
   ```bash
   git push -u origin HEAD
   ```

5. If a PR is requested:
   ```bash
   gh pr create --title "<type>(<scope>): <description>" --body "## Summary\n- ...\n\n## Test plan\n- ..."
   ```
