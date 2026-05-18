# Common mistakes Claude makes in this repo

**How to add an entry:** after a correction, ask Claude *why* it went wrong and have it
propose the rule that would prevent it. Add the proposed rule below. Each entry should
name the mistake in one short heading, show WRONG/RIGHT snippets, and state the
underlying invariant in one sentence.

## 1. Using `from_pretrained` instead of `from_config`

WRONG: `model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")`
RIGHT: `config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B"); model = AutoModelForCausalLM.from_config(config)`

This project trains from scratch — never load pretrained model weights.

## 2. Running Python without `uv run`

WRONG: `python scripts/train.py` · `pytest`
RIGHT: `uv run python scripts/train.py` · `uv run pytest`

The project uses uv for dependency management; every Python invocation goes through `uv run`.

## 3. Catching bare `Exception`

WRONG: `except Exception: ...`
RIGHT: `except (ValueError, KeyError): ...`

Catch specific types so bugs aren't silently swallowed.

## 4. Imports inside functions

WRONG: `def process(): import torch; ...`
RIGHT: `import torch` at module top

Module-level imports only; in-function imports require a comment justifying a circular- or optional-dependency reason.

## 5. Empty-string defaults on dataclass fields

WRONG: `name: str = ""`
RIGHT: `name: str`  (require it to be provided)

Empty defaults hide missing data — only assign a default when it's a meaningful one.

## 6. Using `git add -A`

WRONG: `git add -A` / `git add .`
RIGHT: stage specific files by name

Never blindly stage everything; review and stage intended files only.
