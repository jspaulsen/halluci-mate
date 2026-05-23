# halluci-mate

Chess LLM trained from scratch using the Qwen3-0.6B architecture with a custom UCI chess tokenizer.

## Overview

- **Architecture**: Qwen3-0.6B (~600M parameters)
- **Tokenizer**: Custom ~1,796 token vocabulary for UCI chess moves
- **Dataset**: Lichess streaming dataset
- **Stack**: Python 3.12, uv, PyTorch, HuggingFace Transformers

## Setup

```bash
# Install dependencies
uv sync

# Create base model with resized embeddings for chess tokenizer
uv run python scripts/setup.py

# Run training
uv run python scripts/train.py
```

## Evaluation

The eval harness lives at `scripts/eval.py` with four subcommands. Each run
writes to `evals/<run-id>/{config.json, records.jsonl, metrics.json}`.

```bash
# Play N games vs Stockfish; tag every move with CPL + blunder via --sf-analyze.
# Required for the quality DPO flavor below.
uv run python scripts/eval.py vs-stockfish \
    --checkpoint runs-v1/<run>/checkpoint-<step> \
    --games 50 --stockfish-skill 5 --stockfish-depth 12 --sf-analyze

# Unconstrained top-1 legality on a position set.
uv run python scripts/eval.py legal-rate \
    --checkpoint <ckpt> --sample-from-games data/test.pgn --n 10000

# Token-level cross-entropy on held-out sequences.
uv run python scripts/eval.py perplexity \
    --checkpoint <ckpt> --data data/test.jsonl

# Recompute metrics.json from an existing run's records.jsonl.
uv run python scripts/eval.py report <run-id>
```

### Exporting a DPO dataset

`export-dpo` is a post-hoc transform over a `vs-stockfish` run directory; it
does not replay games.

```bash
# Legality pairs (any vs-stockfish run): rescued mask move (chosen) vs.
# illegal unconstrained top-1 (rejected).
uv run python scripts/eval.py export-dpo <run-id> \
    --output data/dpo_legality.jsonl --flavor legality

# Quality pairs (requires --sf-analyze): Stockfish best (chosen) vs. the
# model's move (rejected), filtered by centipawn_loss > threshold.
uv run python scripts/eval.py export-dpo <run-id> \
    --output data/dpo_quality.jsonl --flavor quality --threshold 200

# Both flavors into one file, deduped by FEN (first pair wins).
uv run python scripts/eval.py export-dpo <run-id> \
    --output data/dpo_both.jsonl --flavor both --threshold 200 --dedup-by-fen
```

See `docs/eval_harness.md` for the full record schema, metrics, and design
rationale.

## Development

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run ty check              # type check
uv run pytest                # run tests
```
