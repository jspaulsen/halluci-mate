"""Prepare Lichess chess games for training.

Loads games from the HuggingFace Lichess dataset, parses PGN movetext
into UCI sequences, applies the training format from docs/data_format.md,
tokenizes, and saves train/eval splits as Parquet files.

Usage:
    uv run python scripts/prepare_data.py
    uv run python scripts/prepare_data.py --num-games 1000000 --output-dir data/v2
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

from datasets import Dataset, load_dataset

from halluci_mate.chess_tokenizer import ChessTokenizer
from halluci_mate.game_to_sequences import game_to_sequences
from halluci_mate.pgn_to_uci import parse_movetext, parse_result

logger = logging.getLogger(__name__)

DEFAULT_NUM_GAMES = 5_000_000
DEFAULT_OUTPUT_DIR = Path("data")
EVAL_SIZE = 10_000
# Skip games with fewer moves — likely disconnects or early resignations
MIN_MOVES = 10
SHUFFLE_SEED = 42
# Print progress every N games
LOG_INTERVAL = 100_000


def process_game(sample: dict, tokenizer: ChessTokenizer) -> list[dict[str, list[int]]]:
    """Process a single Lichess game into tokenized training examples.

    Args:
        sample: Raw dataset row with "movetext" and "Result" fields.
        tokenizer: Chess tokenizer for encoding sequences.

    Returns:
        List of dicts with "input_ids" and "attention_mask" keys.
        Empty list if the game should be skipped.
    """
    try:
        moves = parse_movetext(sample["movetext"])
    except ValueError:
        return []

    if len(moves) < MIN_MOVES:
        return []

    try:
        outcome = parse_result(sample["Result"])
    except ValueError:
        return []

    sequences = game_to_sequences(moves, outcome)

    results: list[dict[str, list[int]]] = []
    for seq in sequences:
        encoded = tokenizer(seq, add_special_tokens=False)
        results.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        )
    return results


def prepare_dataset(
    num_games: int = DEFAULT_NUM_GAMES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Load, process, split, and save the dataset.

    Args:
        num_games: Number of raw games to process from the stream.
        output_dir: Directory to save train.parquet and eval.parquet.
    """
    tokenizer = ChessTokenizer()

    logger.info("Loading Lichess dataset (streaming)...")
    stream = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)
    stream = stream.filter(lambda x: x["Termination"] == "Normal")

    examples: list[dict[str, list[int]]] = []
    skipped = 0

    logger.info("Processing %d games...", num_games)
    for i, sample in enumerate(stream):
        if i >= num_games:
            break

        game_examples = process_game(sample, tokenizer)
        if game_examples:
            examples.extend(game_examples)
        else:
            skipped += 1

        if (i + 1) % LOG_INTERVAL == 0:
            logger.info("Processed %d/%d games (%d sequences so far, %d skipped)", i + 1, num_games, len(examples), skipped)

    logger.info("Done: %d sequences from %d games (%d skipped)", len(examples), num_games, skipped)

    if len(examples) <= EVAL_SIZE:
        raise ValueError(f"Only {len(examples)} sequences produced — need more than {EVAL_SIZE} for a train/eval split. Increase --num-games.")

    # Shuffle and split
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(examples)

    eval_examples = examples[-EVAL_SIZE:]
    train_examples = examples[:-EVAL_SIZE]

    logger.info("Saving %d train / %d eval sequences to %s", len(train_examples), len(eval_examples), output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Dataset.from_list(train_examples).to_parquet(str(output_dir / "train.parquet"))
    Dataset.from_list(eval_examples).to_parquet(str(output_dir / "eval.parquet"))

    logger.info("Saved to %s/train.parquet and %s/eval.parquet", output_dir, output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Prepare Lichess data for training.")
    parser.add_argument("--num-games", type=int, default=DEFAULT_NUM_GAMES, help="Number of games to process")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for Parquet files")
    args = parser.parse_args()

    prepare_dataset(num_games=args.num_games, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
