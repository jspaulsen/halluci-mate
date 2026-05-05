"""Shared eval-record builder factories.

Lifted out of `tests/eval/conftest.py` so suites under any subdirectory of
`tests/` (including `tests/scripts/`) can build records without importing
across pytest collection roots. Builders take an `event_id` plus arbitrary
keyword overrides (validated by pydantic), so individual tests can pin only
the fields they care about.
"""

from __future__ import annotations

from typing import Any

from halluci_mate.eval.records import (
    Evaluator,
    PerGameRecord,
    PerLegalRateRecord,
    PerMoveRecord,
    PerPerplexityRecord,
    PerPuzzleRecord,
    Phase,
    Side,
    TopKEntry,
)

DEFAULT_RUN_ID = "2026-04-19T20-15-00_marvelous-deer-608-ckpt9660_vs-stockfish"
DEFAULT_CHECKPOINT = "runs-v1/marvelous-deer-608/checkpoint-9660"
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def make_per_move_record(event_id: int = 0, **overrides: Any) -> PerMoveRecord:
    """Build a `PerMoveRecord` with sensible defaults; overrides win."""
    fields: dict[str, Any] = {
        "run_id": DEFAULT_RUN_ID,
        "event_id": event_id,
        "evaluator": Evaluator.VS_STOCKFISH,
        "checkpoint": DEFAULT_CHECKPOINT,
        "game_id": f"g{event_id}",
        "ply": event_id,
        "phase": Phase.OPENING,
        "side_to_move": Side.WHITE,
        "model_side": Side.WHITE,
        "fen_before": START_FEN,
        "legal_moves": ["e2e4", "d2d4"],
        "model_move": "e2e4",
        "model_top_k": [
            TopKEntry(move="e2e4", logprob=-0.1),
            TopKEntry(move="d2d4", logprob=-1.2),
        ],
        "mask_used": True,
        "raw_sample_move": "e2e4",
        "raw_sample_legal": True,
        "prior_opponent_move": None,
        "sf_best_move": None,
        "sf_eval_before_cp": None,
        "sf_eval_after_cp": None,
        "centipawn_loss": None,
        "is_blunder": None,
    }
    fields.update(overrides)
    return PerMoveRecord(**fields)


def make_per_game_record(event_id: int = 0, **overrides: Any) -> PerGameRecord:
    """Build a `PerGameRecord` with sensible defaults; overrides win."""
    fields: dict[str, Any] = {
        "run_id": DEFAULT_RUN_ID,
        "event_id": event_id,
        "evaluator": Evaluator.VS_STOCKFISH,
        "checkpoint": DEFAULT_CHECKPOINT,
        "game_id": f"game-{event_id:04d}",
        "model_side": Side.WHITE,
        "result": "1-0",
        "termination": "natural",
        "ply_count": 40,
    }
    fields.update(overrides)
    return PerGameRecord(**fields)


def make_per_puzzle_record(event_id: int = 0, **overrides: Any) -> PerPuzzleRecord:
    """Build a `PerPuzzleRecord` with sensible defaults; overrides win."""
    fields: dict[str, Any] = {
        "run_id": DEFAULT_RUN_ID,
        "event_id": event_id,
        "evaluator": Evaluator.PUZZLES,
        "checkpoint": DEFAULT_CHECKPOINT,
        "puzzle_id": f"p{event_id}",
        "rating": 1450,
        "themes": ["fork", "mateIn2"],
        "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "solution": ["f3e5", "c6e5"],
        "model_attempt": ["f3e5", "c6e5"],
        "solved": True,
    }
    fields.update(overrides)
    return PerPuzzleRecord(**fields)


def make_per_legal_rate_record(event_id: int = 0, **overrides: Any) -> PerLegalRateRecord:
    """Build a `PerLegalRateRecord` with sensible defaults; overrides win."""
    fields: dict[str, Any] = {
        "run_id": DEFAULT_RUN_ID,
        "event_id": event_id,
        "evaluator": Evaluator.LEGAL_RATE,
        "checkpoint": DEFAULT_CHECKPOINT,
        "position_id": f"pos{event_id}",
        "fen": START_FEN,
        "model_move": "e2e4",
        "legal": True,
    }
    fields.update(overrides)
    return PerLegalRateRecord(**fields)


def make_per_perplexity_record(event_id: int = 0, **overrides: Any) -> PerPerplexityRecord:
    """Build a `PerPerplexityRecord` with sensible defaults; overrides win."""
    fields: dict[str, Any] = {
        "run_id": DEFAULT_RUN_ID,
        "event_id": event_id,
        "evaluator": Evaluator.PERPLEXITY,
        "checkpoint": DEFAULT_CHECKPOINT,
        "position_id": f"seq{event_id}",
        "fen": START_FEN,
        "token_logprobs": [-0.1, -0.2, -0.3],
    }
    fields.update(overrides)
    return PerPerplexityRecord(**fields)
