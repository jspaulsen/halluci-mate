"""Stage 2 tokenization: raw self-play shards -> tokenized training shards.

Engine analogue of ``data_preparation.process_game`` / ``stream_and_shard``.
Imports core helpers (one-directional: distill -> core); emits no
``elo_bucket`` because self-play continuations have no single rating.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from halluci_mate.data_preparation import SHARD_SIZE, write_shard
from halluci_mate.distill.raw_shards import read_raw_games
from halluci_mate.game_metadata import classify_opening_family
from halluci_mate.game_to_sequences import game_to_sequences

if TYPE_CHECKING:
    from pathlib import Path

    from halluci_mate.chess_tokenizer import ChessTokenizer
    from halluci_mate.distill.raw_shards import RawGameRow

logger = logging.getLogger(__name__)


def process_engine_game(game: RawGameRow, tokenizer: ChessTokenizer) -> list[dict]:
    """Tokenize one validated raw self-play game into training examples.

    Moves are already UCI (no PGN parsing). Decisive games yield one
    winner-perspective sequence; draws yield two (one per perspective).
    """
    opening_family = classify_opening_family(game.moves_uci[0])
    # ``game.outcome`` is already normalized to "white"/"black"/"draw"; this is
    # NOT ``classify_termination_type`` (which maps PGN result strings).
    termination_type = "draw" if game.outcome == "draw" else "decisive"

    results: list[dict] = []
    for seq in game_to_sequences(game.moves_uci, game.outcome):
        encoded = tokenizer(seq, add_special_tokens=False)
        results.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "result": game.outcome,
                "opening_family": opening_family,
                "termination_type": termination_type,
            }
        )
    return results


def tokenize_engine_shards(raw_dir: Path, tokenizer: ChessTokenizer, shard_dir: Path, *, dedup_by_game_id: bool = True) -> int:
    """Tokenize all raw shards into tokenized shards; return the example count.

    The count is *training examples*, not games: draws contribute two (one per
    perspective), decisive games one. ``dedup_by_game_id`` drops games whose id
    was already seen -- defends against the handful of duplicate games a
    parallel-generation crash+resume can leave.
    """
    buffer: list[dict] = []
    shard_index = 0
    total = 0
    seen: set[str] = set()  # one short id per game; ~tens of MB at corpus scale
    for game in read_raw_games(raw_dir):
        if dedup_by_game_id:
            if game.game_id in seen:
                continue
            seen.add(game.game_id)
        buffer.extend(process_engine_game(game, tokenizer))
        if len(buffer) >= SHARD_SIZE:
            write_shard(buffer, shard_dir, shard_index)
            total += len(buffer)
            buffer.clear()
            shard_index += 1
    if buffer:
        write_shard(buffer, shard_dir, shard_index)
        total += len(buffer)
    logger.info("Tokenized %d examples from %s", total, raw_dir)
    return total
