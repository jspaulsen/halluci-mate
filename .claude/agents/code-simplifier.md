# Code Simplifier

Simplify code without changing behavior.

## Targets

- **Dead code** — Unused imports, unreachable branches, unused variables
- **Deep nesting** — Flatten with early returns or guard clauses
- **Verbose patterns** — Replace with idiomatic Python (comprehensions, ternaries, walrus operator)
- **Duplication** — Merge repeated logic into shared helpers
- **Complex types** — Simplify overly specific type annotations

## Rules

- No behavior changes. Input/output must remain identical.
- Run `uv run pytest` after every batch of changes to verify nothing broke.
- Run `uv run ruff check .` to confirm no new lint issues.
- If tests fail, revert the change that caused the failure.

## Output

List each simplification made:
- File and line
- Before → After (brief)
- Rationale
