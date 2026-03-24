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

## Development

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run ty check              # type check
uv run pytest                # run tests
```
