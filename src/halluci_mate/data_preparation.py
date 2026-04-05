"""Data preparation logic for tokenizing and splitting Lichess games.

Processes raw Lichess game samples into tokenized training examples with
metadata, writes intermediate shards to disk, and builds stratified
train/eval/test splits.
"""

from __future__ import annotations

import logging
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path  # noqa: TC003 — used at runtime in path operations

from datasets import Dataset, IterableDataset, load_dataset
from tqdm import tqdm

from halluci_mate.chess_tokenizer import ChessTokenizer
from halluci_mate.game_metadata import classify_elo_bucket, classify_opening_family, classify_termination_type
from halluci_mate.game_to_sequences import game_to_sequences
from halluci_mate.pgn_to_uci import parse_movetext, parse_result

logger = logging.getLogger(__name__)

SHUFFLE_SEED = 42
# Number of examples buffered in memory before flushing to a shard file
SHARD_SIZE = 100_000

# 95/4/1 train/eval/test split — test is held-out for final convergence
# evaluation only, never touched during development.
# Ref: Allie (ICLR 2025) https://openreview.net/forum?id=bc2H72hGxB
EVAL_FRACTION = 0.04
TEST_FRACTION = 0.01

_METADATA_COLUMNS = ["elo_bucket", "result", "opening_family", "termination_type"]


def create_tokenizer() -> ChessTokenizer:
    """Create a ChessTokenizer instance for use in the pipeline."""
    return ChessTokenizer()


def _parse_elo(sample: dict) -> tuple[int, int] | None:
    """Extract integer ELO ratings, returning None if unparseable."""
    try:
        return int(sample["WhiteElo"]), int(sample["BlackElo"])
    except (ValueError, KeyError, TypeError):
        return None


def process_game(sample: dict, tokenizer: ChessTokenizer) -> list[dict]:
    """Process a single Lichess game into tokenized training examples with metadata.

    Returns:
        List of dicts with input_ids, attention_mask, and metadata columns.
        Empty list if the game should be skipped.
    """
    try:
        moves = parse_movetext(sample["movetext"])
    except ValueError:
        return []

    try:
        outcome = parse_result(sample["Result"])
    except ValueError:
        return []

    elo = _parse_elo(sample)
    elo_bucket = classify_elo_bucket(*elo) if elo else "unknown"
    opening_family = classify_opening_family(moves[0])
    termination_type = classify_termination_type(sample["Result"])

    sequences = game_to_sequences(moves, outcome)
    results: list[dict] = []
    for seq in sequences:
        encoded = tokenizer(seq, add_special_tokens=False)
        results.append({
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "elo_bucket": elo_bucket,
            "result": outcome,
            "opening_family": opening_family,
            "termination_type": termination_type,
        })
    return results


def write_shard(examples: list[dict], shard_dir: Path, shard_index: int) -> Path:
    """Write a batch of examples to a numbered Parquet shard file."""
    path = shard_dir / f"shard_{shard_index:05d}.parquet"
    Dataset.from_list(examples).to_parquet(str(path))
    return path


def build_stratified_splits(shard_dir: Path, eval_size: int, test_size: int, seed: int) -> tuple[Dataset, Dataset, Dataset]:
    """Load all shards and split into stratified train/eval/test datasets."""
    shard_files = sorted(str(p) for p in shard_dir.glob("shard_*.parquet"))
    all_data = load_dataset("parquet", data_files=shard_files, split="train")

    stratum_indices: dict[str, list[int]] = defaultdict(list)
    for idx in range(len(all_data)):
        row = all_data[idx]
        key = f"{row['elo_bucket']}|{row['result']}|{row['opening_family']}"
        stratum_indices[key].append(idx)

    holdout_size = eval_size + test_size
    holdout_indices = _sample_stratified(stratum_indices, holdout_size, len(all_data), seed)
    test_indices = holdout_indices[:test_size]
    eval_indices = holdout_indices[test_size:]

    holdout_set = set(holdout_indices)
    train_indices = [i for i in range(len(all_data)) if i not in holdout_set]

    return all_data.select(train_indices), all_data.select(eval_indices), all_data.select(test_indices)


def _sample_stratified(stratum_indices: dict[str, list[int]], eval_size: int, total: int, seed: int) -> list[int]:
    """Proportionally sample from each stratum to reach eval_size."""
    rng = random.Random(seed)
    sampled: list[int] = []

    for indices in stratum_indices.values():
        n = max(1, round(eval_size * len(indices) / total))
        sampled.extend(rng.sample(indices, min(n, len(indices))))

    rng.shuffle(sampled)
    if len(sampled) > eval_size:
        return sampled[:eval_size]
    if len(sampled) < eval_size:
        remaining = list(set(range(total)) - set(sampled))
        extra = rng.sample(remaining, eval_size - len(sampled))
        sampled.extend(extra)
    return sampled


def strip_metadata(dataset: Dataset) -> Dataset:
    """Remove metadata columns, keeping only input_ids and attention_mask."""
    cols_to_remove = [c for c in _METADATA_COLUMNS if c in dataset.column_names]
    return dataset.remove_columns(cols_to_remove)


def log_eval_distribution(eval_data: Dataset) -> None:
    """Log the distribution of each metadata dimension in the eval set."""
    for col in ["elo_bucket", "result", "opening_family"]:
        if col in eval_data.column_names:
            counts = Counter(eval_data[col])
            logger.info("Eval %s distribution: %s", col, dict(sorted(counts.items())))


def stream_and_shard(stream: IterableDataset, tokenizer: ChessTokenizer, num_games: int, shard_dir: Path) -> tuple[int, int]:
    """Process streamed games into tokenized shards on disk.

    Returns:
        (total_examples, skipped) counts.
    """
    buffer: list[dict] = []
    shard_index = 0
    skipped = 0
    total_examples = 0
    games_processed = 0

    with tqdm(total=num_games, desc="Processing games") as pbar:
        for sample in stream:
            if games_processed >= num_games:
                break

            game_examples = process_game(sample, tokenizer)
            games_processed += 1
            pbar.update(1)

            if game_examples:
                buffer.extend(game_examples)
            else:
                skipped += 1

            if len(buffer) >= SHARD_SIZE:
                write_shard(buffer, shard_dir, shard_index)
                total_examples += len(buffer)
                buffer.clear()
                shard_index += 1

    if buffer:
        write_shard(buffer, shard_dir, shard_index)
        total_examples += len(buffer)
        shard_index += 1

    logger.info("Done: %d sequences from %d games (%d skipped, %d shards)", total_examples, num_games, skipped, shard_index)
    return total_examples, skipped


def save_splits(shard_dir: Path, total_examples: int, output_dir: Path) -> None:
    """Build stratified splits from shards and save as Parquet files."""
    eval_size = max(1, round(total_examples * EVAL_FRACTION))
    test_size = max(1, round(total_examples * TEST_FRACTION))
    min_required = eval_size + test_size + 1

    if total_examples <= min_required:
        raise ValueError(f"Only {total_examples} sequences produced — need more than {min_required} for train/eval/test splits. Increase --num-games.")

    logger.info("Building stratified splits (eval=%d, test=%d)...", eval_size, test_size)
    train_data, eval_data, test_data = build_stratified_splits(shard_dir, eval_size, test_size, SHUFFLE_SEED)
    log_eval_distribution(eval_data)

    train_data = strip_metadata(train_data)
    eval_data = strip_metadata(eval_data)
    test_data = strip_metadata(test_data)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_data.to_parquet(str(output_dir / "train.parquet"))
    eval_data.to_parquet(str(output_dir / "eval.parquet"))
    test_data.to_parquet(str(output_dir / "test.parquet"))

    logger.info("Saved %d train / %d eval / %d test to %s", len(train_data), len(eval_data), len(test_data), output_dir)

    shutil.rmtree(shard_dir)
    logger.info("Cleaned up shard directory")
