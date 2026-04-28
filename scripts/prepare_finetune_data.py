"""Prepare a high-Elo Lichess subset for fine-tuning the v1a model.

Same pipeline as prepare_data.py but with an additional filter that keeps only
games where both players are strong (>= min Elo) and the rating signal is
stable: small RatingDiff magnitude (excludes provisional / volatile players)
and small Elo gap between players (excludes off-pool pairings, smurfs).

Default target: 500k post-filter games. The pass rate for both-players-2000+
on Lichess Blitz is roughly 3-5%, so the stream consumes substantially more
raw games to reach the target.

Usage:
    uv run python scripts/prepare_finetune_data.py
    uv run python scripts/prepare_finetune_data.py --num-games 1000000
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from datasets import load_dataset

from halluci_mate.data_preparation import create_tokenizer, passes_highelo_filter, save_splits, stream_and_shard
from halluci_mate.logging_setup import configure_script_logging

logger = logging.getLogger(__name__)

DEFAULT_NUM_GAMES = 1_000_000
DEFAULT_OUTPUT_DIR = Path("data/v1a-highelo")
DEFAULT_MIN_ELO = 2000
DEFAULT_MAX_RATING_DIFF = 30
DEFAULT_MAX_ELO_GAP = 200


def prepare_dataset(
    num_games: int,
    output_dir: Path,
    min_elo: int,
    max_rating_diff: int,
    max_elo_gap: int,
) -> None:
    """Stream, filter for high-Elo + stable ratings, shard, stratify, save."""
    tokenizer = create_tokenizer()
    shard_dir = output_dir / "_shards"

    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True)

    logger.info("Loading Lichess dataset (streaming, high-Elo filter)...")
    stream = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)
    stream = stream.filter(lambda x: x["Termination"] == "Normal" and "blitz" in x.get("Event", "").lower())
    stream = stream.filter(lambda x: passes_highelo_filter(x, min_elo, max_rating_diff, max_elo_gap))

    total_examples, _ = stream_and_shard(stream, tokenizer, num_games, shard_dir)
    save_splits(shard_dir, total_examples, output_dir)


def main() -> None:
    configure_script_logging(__name__)

    parser = argparse.ArgumentParser(description="Prepare high-Elo Lichess data for fine-tuning.")
    parser.add_argument("--num-games", type=int, default=DEFAULT_NUM_GAMES, help="Target number of post-filter games")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for Parquet files")
    parser.add_argument("--min-elo", type=int, default=DEFAULT_MIN_ELO, help="Minimum Elo for both players")
    parser.add_argument("--max-rating-diff", type=int, default=DEFAULT_MAX_RATING_DIFF, help="Max |RatingDiff| for either player")
    parser.add_argument("--max-elo-gap", type=int, default=DEFAULT_MAX_ELO_GAP, help="Max |WhiteElo - BlackElo|")
    args = parser.parse_args()

    prepare_dataset(
        num_games=args.num_games,
        output_dir=args.output_dir,
        min_elo=args.min_elo,
        max_rating_diff=args.max_rating_diff,
        max_elo_gap=args.max_elo_gap,
    )


if __name__ == "__main__":
    main()
