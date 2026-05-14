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
does not replay games. Each output line is a JSON object with fields:

- `prompt` — FEN before the move (human-readable, also used for dedup).
- `moves_uci` — list of UCI moves from the game's start position up to (but
  not including) the labeled move. This is the prompt the model was actually
  conditioned on; the FEN string is not in the tokenizer's vocabulary.
- `model_side` — `"white"` or `"black"`; determines the leading `<WHITE>` /
  `<BLACK>` perspective token at training time.
- `chosen` / `rejected` — UCI moves.

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

# Quality pairs scoped to moves played from positions that are not yet lost
# and that do not recur within the same game — the recommended filters for
# the Phase 3 (quality) DPO training set.
uv run python scripts/eval.py export-dpo <run-id> \
    --output data/dpo_consequential.jsonl --flavor quality \
    --require-consequential --exclude-repetition
```

### DPO training

`scripts/train_dpo.py` consumes the JSONL produced above via TRL's
`DPOTrainer`. It loads a base checkpoint (default
`jspaulsen/halluci-mate-v1b`) as both policy and reference model, formats
each pair into the prompt sequence the chess tokenizer expects (perspective
token + UCI history), and trains a single move completion against
chosen/rejected.

```bash
uv run accelerate launch scripts/train_dpo.py
```

Defaults are conservative (LR 5e-7, beta 0.1, single epoch); edit the
`main()` signature for hyperparam sweeps.

See `docs/eval_harness.md` for the full record schema, metrics, and design
rationale.

### Comparing two models

`scripts/compare.py` is a Streamlit dashboard that scans `evals/`, groups runs
by checkpoint, and renders a side-by-side metrics table + bar charts for
vs-stockfish, legal-rate, and perplexity. Each section has a run picker per
model (defaulting to the most recent) so you can flip between, e.g., a
general-blitz perplexity run and a high-elo perplexity run for the same
checkpoint.

```bash
uv run streamlit run scripts/compare.py
```

## Development

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run ty check              # type check
uv run pytest                # run tests
```
