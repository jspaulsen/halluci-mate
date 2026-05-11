"""Prepare Lichess chess games for training.

Loads games from the HuggingFace Lichess dataset, parses PGN movetext
into UCI sequences, applies the training format from docs/data_format.md,
tokenizes, and saves train/eval/test splits as Parquet files.

Writes intermediate shards to disk to avoid OOM on large datasets, then
builds stratified eval and test sets balanced across ELO, result, and
opening dimensions.

Only blitz games are included — mixing time controls confounds the Elo
signal since move quality varies significantly across formats.
Ref: Allie (ICLR 2025) https://openreview.net/forum?id=bc2H72hGxB

Usage:
    uv run python scripts/prepare_data.py
    uv run python scripts/prepare_data.py --num-games 1000000 --output-dir data/v2
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Annotated

import typer
from datasets import load_dataset

from halluci_mate.data_preparation import create_tokenizer, save_splits, stream_and_shard
from halluci_mate.logging_setup import configure_script_logging

logger = logging.getLogger(__name__)

DEFAULT_NUM_GAMES = 5_000_000
DEFAULT_OUTPUT_DIR = Path("data")


def prepare_dataset(
    num_games: int = DEFAULT_NUM_GAMES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Load, process, shard, stratify, and save the dataset."""
    tokenizer = create_tokenizer()
    shard_dir = output_dir / "_shards"

    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True)

    logger.info("Loading Lichess dataset (streaming)...")
    stream = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)
    # Blitz-only: mixing time controls confounds the Elo signal since move quality
    # varies significantly across formats (Allie validated blitz-only → GM-calibrated play).
    # Ref: https://openreview.net/forum?id=bc2H72hGxB
    stream = stream.filter(lambda x: x["Termination"] == "Normal" and "blitz" in x.get("Event", "").lower())

    total_examples, _ = stream_and_shard(stream, tokenizer, num_games, shard_dir)
    save_splits(shard_dir, total_examples, output_dir)


def main(
    num_games: Annotated[int, typer.Option(help="Number of games to process")] = DEFAULT_NUM_GAMES,
    output_dir: Annotated[Path, typer.Option(help="Output directory for Parquet files")] = DEFAULT_OUTPUT_DIR,
) -> None:
    """Prepare Lichess data for training."""
    configure_script_logging(__name__)
    prepare_dataset(num_games=num_games, output_dir=output_dir)


if __name__ == "__main__":
    typer.run(main)
