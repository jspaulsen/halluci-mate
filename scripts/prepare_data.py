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

import argparse
import logging
import shutil
from pathlib import Path

from datasets import load_dataset

from halluci_mate.data_preparation import create_tokenizer, save_splits, stream_and_shard

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
    stream = stream.filter(lambda x: x["Termination"] == "Normal" and "Blitz" in x.get("Event", ""))

    total_examples, _ = stream_and_shard(stream, tokenizer, num_games, shard_dir)
    save_splits(shard_dir, total_examples, output_dir)


def _configure_logging() -> None:
    """Configure logging for this script and halluci_mate modules only."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    # Configure our loggers without affecting third-party libraries (e.g., HuggingFace)
    for name in (__name__, "halluci_mate"):
        log = logging.getLogger(name)
        log.setLevel(logging.INFO)
        log.addHandler(handler)


def main() -> None:
    _configure_logging()

    parser = argparse.ArgumentParser(description="Prepare Lichess data for training.")
    parser.add_argument("--num-games", type=int, default=DEFAULT_NUM_GAMES, help="Number of games to process")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for Parquet files")
    args = parser.parse_args()

    prepare_dataset(num_games=args.num_games, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
