# Code style

Python 3.12, uv-managed. `ruff` owns formatting + lint; `ty` is the type checker.
Run everything through `uv run` (see `## Common Mistakes` in CLAUDE.md).

## Naming

- Files / modules: `snake_case` (e.g. `chess_tokenizer.py`, `game_to_sequences.py`).
- Classes: `PascalCase` (e.g. `ChessTokenizer`, `ChessInferenceEngine`, `MovePrediction`).
- Functions / variables: `snake_case`.
- Module-level constants: `SCREAMING_SNAKE_CASE` (e.g. `PAD_TOKEN_ID`, `SHARD_SIZE`).
- Test files: `MODULE_test.py` suffix mirroring the source module — e.g.
  `chess_tokenizer.py` → `tests/chess_tokenizer_test.py`. NOT the `test_` prefix form.

## Imports

- Imports at module level. Local (in-function) imports ONLY for genuine circular- or
  optional-dependency reasons, with a one-line comment saying which.
- Absolute imports for internal code: `from halluci_mate.game import Game`. No
  relative imports within `src/`.
- No barrel / aggregating re-exports. `__init__.py` stays minimal (only `main()`).
  Import from the file that owns the symbol.
- `from __future__ import annotations` at the top of every source file. Put
  type-only imports under `if TYPE_CHECKING:`.

## Error handling

- Define custom domain exceptions; subclass the most specific stdlib base that fits
  (the codebase subclasses `ValueError` — see `IllegalMoveError`, `GameOverError`
  in `inference.py`).
- Raise with a descriptive message, including the offending value where useful
  (`ValueError(f"Unrecognized result: {result!r}")`).
- Never catch bare `Exception` except at a top-level reporting boundary. Catch the
  specific types you expect (`except (ValueError, KeyError):`).
- Let unexpected exceptions propagate — don't swallow errors, don't catch just to
  re-raise with a reworded message.

## Typing strictness

- Every function signature fully annotated (params + return). `ty` must pass:
  `uv run ty check`.
- Use `X | None` unions, not `Optional[X]`. Use `Protocol` for structural typing
  (e.g. the `Predictor` protocol in `inference.py`).
- Avoid `Any`. It is acceptable only in Pydantic test builders (`**overrides: Any`).
  Narrow external/untyped data explicitly instead.
- No silent null bypass — check before use; don't paper over `None` with casts.

## Docstrings

- Google-style triple-quoted docstrings with `Args:` / `Returns:` / `Raises:`
  sections. Match the existing style (see `pgn_to_uci.parse_movetext`). Module-level
  docstrings state the file's purpose.

## General

- No magic numbers — name them as module-level constants with a comment giving
  units/context.
- Keep functions under ~40 lines; extract helpers past that.
- No commented-out code in commits. Delete it.
- `ruff` owns formatting — never hand-adjust style. Line length is **180**
  (`[tool.ruff]` `line-length = 180`, `E501` ignored). Lint selects
  `E, F, I, UP, B, SIM, TCH, RUF`.
- Assign a dataclass default only when there's a meaningful one. No
  empty-string/empty-list defaults that hide missing data.
- Prefer module-level constants over fixtures for simple shared test values.
