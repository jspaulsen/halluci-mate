"""Round-trip serialization tests for eval record dataclasses."""

from __future__ import annotations

import json

import pytest

from halluci_mate.eval.records import (
    Evaluator,
    PerLegalRateRecord,
    PerMoveRecord,
    PerPerplexityRecord,
    PerPuzzleRecord,
    Phase,
    Record,
    Side,
    TopKEntry,
    record_from_dict,
    record_to_dict,
)

PER_MOVE = PerMoveRecord(
    run_id="2026-04-19T20-15-00_ckpt-9660_vs-stockfish",
    event_id=0,
    evaluator=Evaluator.VS_STOCKFISH,
    checkpoint="runs-v1/marvelous-deer-608/checkpoint-9660",
    game_id="g0",
    ply=1,
    phase=Phase.OPENING,
    side_to_move=Side.WHITE,
    model_side=Side.WHITE,
    fen_before="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    legal_moves=["e2e4", "d2d4"],
    model_move="e2e4",
    model_top_k=[
        TopKEntry(move="e2e4", logprob=-0.1),
        TopKEntry(move="d2d4", logprob=-1.2),
    ],
    mask_used=True,
    raw_sample_move="e2e4",
    raw_sample_legal=True,
    prior_opponent_move=None,
    sf_best_move=None,
    sf_eval_before_cp=None,
    sf_eval_after_cp=None,
    centipawn_loss=None,
    is_blunder=None,
)

PER_PUZZLE = PerPuzzleRecord(
    run_id="2026-04-19T20-15-00_ckpt-9660_puzzles",
    event_id=42,
    evaluator=Evaluator.PUZZLES,
    checkpoint="runs-v1/marvelous-deer-608/checkpoint-9660",
    puzzle_id="abcde",
    rating=1450,
    themes=["fork", "mateIn2"],
    fen="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    solution=["f3e5", "c6e5"],
    model_attempt=["f3e5", "c6e5"],
    solved=True,
)

PER_LEGAL_RATE = PerLegalRateRecord(
    run_id="2026-04-19T20-15-00_ckpt-9660_legal-rate",
    event_id=7,
    evaluator=Evaluator.LEGAL_RATE,
    checkpoint="runs-v1/marvelous-deer-608/checkpoint-9660",
    position_id="pos-7",
    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    model_move="e2e4",
    legal=True,
)

PER_PERPLEXITY = PerPerplexityRecord(
    run_id="2026-04-19T20-15-00_ckpt-9660_perplexity",
    event_id=3,
    evaluator=Evaluator.PERPLEXITY,
    checkpoint="runs-v1/marvelous-deer-608/checkpoint-9660",
    position_id="seq-3",
    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    token_logprobs=[-0.1, -0.2, -0.3],
)

ROUND_TRIP_CASES: list[Record] = [
    PER_MOVE,
    PER_PUZZLE,
    PER_LEGAL_RATE,
    PER_PERPLEXITY,
]


@pytest.mark.parametrize("record", ROUND_TRIP_CASES)
def test_record_round_trips_through_dict(record: Record) -> None:
    assert record_from_dict(record_to_dict(record)) == record


@pytest.mark.parametrize("record", ROUND_TRIP_CASES)
def test_record_round_trips_through_json(record: Record) -> None:
    encoded = json.dumps(record_to_dict(record))
    assert record_from_dict(json.loads(encoded)) == record


def test_record_to_dict_serializes_top_k_as_list_of_dicts() -> None:
    encoded = record_to_dict(PER_MOVE)
    assert encoded["model_top_k"] == [
        {"move": "e2e4", "logprob": -0.1},
        {"move": "d2d4", "logprob": -1.2},
    ]


def test_record_from_dict_rejects_unknown_shape() -> None:
    with pytest.raises(ValueError, match="cannot infer record type"):
        record_from_dict(
            {
                "run_id": "x",
                "event_id": 0,
                "evaluator": "made_up",
                "checkpoint": "x",
            }
        )
