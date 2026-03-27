# halluci-mate

## Project Overview

Chess LLM trained from scratch. Qwen3-0.6B architecture, custom ~1800-token UCI
chess tokenizer, Lichess streaming dataset. Python 3.12, uv, PyTorch, HuggingFace
Transformers. Entry point: scripts/train.py.

## Build & Test Commands

```
uv run python scripts/train.py    # run training
uv run pytest                     # run all tests
uv run ruff check .               # lint
uv run ruff check --fix .         # lint with auto-fix
uv run ruff format .              # format
uv run ruff format --check .      # check formatting
uv run ty check                   # type check
uv add <pkg>                      # add dependency
uv add --dev <pkg>                # add dev dependency
uv sync                           # install/sync all deps
```

Always run `uv run ruff check . && uv run ty check && uv run pytest` before committing.

## Development Workflow

After writing code and tests:
1. Invoke `/test-and-fix` — iterates pytest + ty + ruff until all pass

Before committing:
2. Invoke `@agent code-simplifier` — simplify code: dead code, duplication, verbose patterns
3. Invoke `@agent verify-app` — full suite PASS/FAIL report (read-only, no fixes)
4. Invoke `@agent build-validator` — build/packaging PASS/FAIL report (install, imports, deps)
5. Invoke `@agent code-architect` — module boundary and structural review → APPROVE/NEEDS_CHANGES

To ship:
6. Invoke `/review-changes` — review diff for logic errors, edge cases, style issues
7. Invoke `/commit-push-pr` — commit + push + open PR

The PostToolUse hook auto-formats/lints Python files on every Edit/Write.
Do not skip steps 2–5. Do not commit without passing verify-app and build-validator reports and code-architect APPROVE.

## Code Style Rules

### Naming

- Files/modules: snake_case (e.g., train.py, chess_tokenizer.py)
- Classes: PascalCase
- Constants: SCREAMING_SNAKE_CASE for module-level compile-time constants
- Functions/variables: snake_case

### Imports

- Keep imports at module level, not inside functions
- Only use local imports for genuine reasons (circular deps, optional deps)
- No barrel imports (__init__.py should be minimal or empty)
- Use type-only imports where appropriate: `from __future__ import annotations`

### Error Handling

- Never catch bare `Exception` unless at the highest level for error reporting
- Always catch specific exception types (e.g., `ValueError`, `FileNotFoundError`)
- Use custom exception classes for domain-specific errors
- Let unexpected exceptions propagate — don't swallow errors

### General

- No magic numbers — name them as constants with a comment explaining units/context
- Keep functions under 40 lines. Extract helpers for anything longer
- No commented-out code in commits. Delete it
- All function signatures must have type annotations
- ruff handles formatting — do not manually adjust style. Line length: 180
- Only assign dataclass defaults when there's a meaningful default
- Use module-level constants for simple test values, not fixtures

## Architecture Constraints

### Directory Structure

```
src/halluci_mate/   ← core library modules (tokenizer, data transforms)
scripts/            ← training scripts (train.py is the main entry point)
docs/               ← reference scripts and documentation
tests/              ← unit tests: <module_name>_test.py
  integration/      ← integration tests (data pipeline, model loading)
```

### Module Boundaries

- scripts/ may import from any internal module
- docs/ contains standalone reference scripts — no imports from scripts/
- tests/ imports from the module under test only

### Dependency Rules

- Do not add new dependencies without discussion. Prefer stdlib or existing deps
- All dev dependencies go in [dependency-groups] dev
- Never import from build artifacts or cached outputs
- Use `uv add`, never `pip install`

## Common Mistakes Claude Makes

### 1. Using from_pretrained instead of from_config

WRONG: `model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")`
RIGHT: `config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B"); model = AutoModelForCausalLM.from_config(config)`
This project trains from scratch — never load pretrained weights for the model.

### 2. Running Python without uv run

WRONG: `python scripts/train.py` or `pytest`
RIGHT: `uv run python scripts/train.py` or `uv run pytest`
The project uses uv for dependency management. All Python commands must use `uv run`.

### 3. Catching bare Exception

WRONG: `except Exception: ...`
RIGHT: `except (ValueError, KeyError): ...`
Catch specific types so bugs don't get silently swallowed.

### 4. Imports inside functions

WRONG: `def process(): import torch; ...`
RIGHT: `import torch` at module top
Only acceptable for circular deps or optional deps with a comment explaining why.

### 5. Empty string defaults on dataclass fields

WRONG: `name: str = ""`
RIGHT: `name: str` (require it to be provided)
Empty defaults hide missing data.

### 6. Using git add -A

Stage specific files by name. Never blindly stage everything.

## PR Conventions

### Commit Messages (Conventional Commits)

Format: `<type>(<scope>): <short description>`
Types: feat, fix, refactor, test, chore, docs, perf
- Subject line: imperative mood, lowercase after colon, no trailing period, ≤72 chars
- Body (optional): explain *why*, not *what*

### PR Policy

- One logical change per PR. Split unrelated changes into separate PRs.
- PR title = commit message subject (will be squash-merged).
- Description must include: what changed, why, and how to test it.
- All PRs require passing CI (lint + typecheck + tests) before merge.

## Testing Rules

### File Naming

- Unit tests: tests/<module_name>_test.py (mirrors src/halluci_mate/<module_name>.py)
- Integration tests: tests/integration/<feature>_test.py
- Test helpers: tests/conftest.py or tests/helpers/<name>.py

### What to Mock vs. What to Test Real

- Mock: external network calls (HuggingFace Hub, W&B API, Lichess)
- Mock: GPU/CUDA operations in unit tests (use CPU tensors)
- Do NOT mock: pure functions, data processing logic, tokenizer operations
- Integration tests use real (small) data samples, never production datasets

### Assertion Style

- Use pytest's plain assert. No additional assertion libraries.
- Use pytest.raises for expected exceptions, pytest.approx for floats.
- A test that always passes is worse than no test.
