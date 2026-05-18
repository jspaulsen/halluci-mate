# Architecture constraints

`halluci-mate` is a chess LLM trained from scratch (Qwen3-0.6B architecture, custom
UCI tokenizer, Lichess streaming data). Python 3.12 + uv; PyTorch / HuggingFace
Transformers. No long-running service — the surfaces are CLI training/eval scripts
and an inference engine.

## Directory structure

```
src/halluci_mate/        ← core library (pure logic + model glue, no CLI concerns)
  __init__.py            ← minimal: exposes main() only. NOT a barrel.
  chess_tokenizer.py     ← HuggingFace PreTrainedTokenizer subclass for UCI moves
  pgn_to_uci.py          ← PGN movetext → UCI move list
  game_to_sequences.py   ← parsed games → training token sequences
  data_preparation.py    ← dataset preprocessing, stratified splits
  game.py                ← Game state: board, perspective, KV cache tracking
  game_metadata.py       ← classifiers: ELO bucket, opening family, termination
  inference.py           ← ChessInferenceEngine, constrained decoding, MovePrediction
  logging_setup.py       ← shared logging config for scripts
  eval/                  ← evaluation harness (depends on inference + records)
    records.py           ← Pydantic record models (PerMoveRecord, PerGameRecord, …)
    metrics.py           ← metric aggregation from records
    runs.py              ← eval run management
    evaluators/          ← perplexity.py, legal_rate.py, vs_stockfish.py
scripts/                 ← Typer CLI entrypoints (I/O boundary)
  train.py               ← MAIN entry point (training loop)
  setup.py               ← build base model with resized embeddings
  prepare_data.py        ← stream Lichess, parse, tokenize, split
  prepare_finetune_data.py
  finetune.py
  eval.py                ← eval harness CLI
  bench_kv_cache.py      ← KV-cache benchmarking
docs/                    ← design docs + standalone reference scripts
tests/                   ← unit tests, *_test.py mirroring src/halluci_mate/
  helpers/               ← test data builders (e.g. eval_records.py factories)
  eval/                  ← tests for src/halluci_mate/eval/
  integration/           ← integration tests (NOT run by default — see pyproject)
  scripts/               ← tests for scripts/
```

## Module boundaries

Dependency direction flows one way. Never reverse it.

- **Data layer** — `pgn_to_uci` → `game_to_sequences` → `data_preparation`, plus
  `chess_tokenizer`. No imports from `inference`/`eval`.
- **Game state** — `game`, `game_metadata`. Depends on the tokenizer only.
- **Inference** — `inference.py` depends on game state, tokenizer, `eval/records`.
- **Eval** — `eval/` depends on `inference` and `eval/records`. Nothing depends on `eval/`.
- `scripts/` (the I/O boundary) may import any `halluci_mate.*`; library code must
  NOT import from `scripts/`.
- `docs/` holds standalone reference scripts — no imports from `scripts/` or each other.
- `tests/` imports only the module under test, plus `tests/helpers/`.
- `src/halluci_mate/__init__.py` stays minimal (only `main()`). No barrel / aggregating
  re-exports — import from the file that owns the symbol.
- Every source file starts with `from __future__ import annotations`; type-only imports
  go under a `TYPE_CHECKING` guard so they don't create runtime import cycles.
- Circular imports are forbidden. A shared abstraction belongs in the lowest layer
  that needs it (typically the data layer or `eval/records`).

## Dependency rules

- No new dependencies without discussion. Prefer stdlib or an existing dep.
- Add deps with `uv add` (runtime) / `uv add --dev` (tooling). Never `pip install`.
- Dev-only tools (`pytest`, `ruff`, `ty`) live in `[dependency-groups] dev` in
  `pyproject.toml` and must justify their presence.
- `torch` resolves through the explicit `pytorch-cu130` uv index
  (`[[tool.uv.index]]` + `[tool.uv.sources]`). Do not reorder, duplicate, or drop
  that source when editing `pyproject.toml`.
- Never import from build artifacts or cached outputs (`.venv/`, `__pycache__/`,
  `build/`, `dist/`).
