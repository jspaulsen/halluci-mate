"""Stage 2: tokenize raw Stockfish self-play shards into train/eval/test splits.

Reads the raw game shards produced by ``generate_distill_games.py``, tokenizes
them with the chess tokenizer, and writes stratified Parquet splits consumable
by ``scripts/train.py`` unchanged. Stratifies on result + opening family
(engine self-play has no single Elo, so ``elo_bucket`` is dropped).

Usage:
    uv run python scripts/prepare_distill_data.py --raw-dir data/distill/_raw --output-dir data/distill
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Annotated

import typer

from halluci_mate.data_preparation import create_tokenizer, save_splits
from halluci_mate.distill.tokenize import tokenize_engine_shards
from halluci_mate.logging_setup import configure_script_logging

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/distill/_raw")
DEFAULT_OUTPUT_DIR = Path("data/distill")
_ENGINE_STRATIFY_COLUMNS = ("result", "opening_family")


def prepare_dataset(raw_dir: Path, output_dir: Path) -> None:
    """Tokenize raw shards, then build + save stratified splits."""
    tokenizer = create_tokenizer()
    shard_dir = output_dir / "_shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True)

    total_examples = tokenize_engine_shards(raw_dir, tokenizer, shard_dir)
    if total_examples == 0:
        # Pre-empt save_splits' "increase --num-games" message, which is wrong for
        # Stage 2 (no such flag) -- the real fix is to run Stage 1 generation first.
        raise ValueError(f"No tokenized examples from {raw_dir}; run generate_distill_games.py first.")
    save_splits(shard_dir, total_examples, output_dir, stratify_columns=_ENGINE_STRATIFY_COLUMNS)


def main(
    raw_dir: Annotated[Path, typer.Option(help="Directory of raw game shards from Stage 1")] = DEFAULT_RAW_DIR,
    output_dir: Annotated[Path, typer.Option(help="Output directory for train/eval/test Parquet")] = DEFAULT_OUTPUT_DIR,
) -> None:
    """Prepare tokenized distillation splits from raw self-play shards."""
    configure_script_logging(__name__)
    prepare_dataset(raw_dir=raw_dir, output_dir=output_dir)


if __name__ == "__main__":
    typer.run(main)
