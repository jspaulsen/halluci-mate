"""Round-trip serialization tests for eval record dataclasses."""

from __future__ import annotations

import json

import pytest

from halluci_mate.eval.records import (
    Evaluator,
    PerGameRecord,
    PerLegalRateRecord,
    PerMoveRecord,
    PerPerplexityRecord,
    PerPuzzleRecord,
    Record,
    record_from_dict,
    record_to_dict,
)
from tests.helpers.eval_records import (
    DEFAULT_CHECKPOINT,
    START_FEN,
    make_per_game_record,
    make_per_move_record,
    make_per_puzzle_record,
)

PER_MOVE = make_per_move_record(
    event_id=0,
    run_id="2026-04-19T20-15-00_ckpt-9660_vs-stockfish",
    game_id="g0",
    ply=1,
)

PER_GAME = make_per_game_record(
    event_id=5,
    run_id="2026-04-19T20-15-00_ckpt-9660_vs-stockfish",
    game_id="g0",
)

PER_PUZZLE = make_per_puzzle_record(
    event_id=42,
    run_id="2026-04-19T20-15-00_ckpt-9660_puzzles",
    puzzle_id="abcde",
)

PER_LEGAL_RATE = PerLegalRateRecord(
    run_id="2026-04-19T20-15-00_ckpt-9660_legal-rate",
    event_id=7,
    evaluator=Evaluator.LEGAL_RATE,
    checkpoint=DEFAULT_CHECKPOINT,
    position_id="pos-7",
    fen=START_FEN,
    model_move="e2e4",
    legal=True,
)

PER_PERPLEXITY = PerPerplexityRecord(
    run_id="2026-04-19T20-15-00_ckpt-9660_perplexity",
    event_id=3,
    evaluator=Evaluator.PERPLEXITY,
    checkpoint=DEFAULT_CHECKPOINT,
    position_id="seq-3",
    fen=START_FEN,
    token_logprobs=[-0.1, -0.2, -0.3],
)

ROUND_TRIP_CASES: list[Record] = [
    PER_MOVE,
    PER_GAME,
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


def test_record_discriminator_keys_are_disjoint() -> None:
    """The shape-based discriminator is sound only if its four type-specific
    keys are pairwise disjoint across record classes — otherwise a record
    that picks a colliding key would silently route to the wrong type and
    lose payload. This test pins that invariant for future record additions.
    """
    discriminator_keys = {
        PerMoveRecord: "ply",
        PerGameRecord: "result",
        PerPuzzleRecord: "puzzle_id",
        PerPerplexityRecord: "token_logprobs",
        PerLegalRateRecord: "legal",
    }
    for owner_cls, key in discriminator_keys.items():
        for other_cls in discriminator_keys:
            if other_cls is owner_cls:
                continue
            assert key not in other_cls.model_fields, f"discriminator key {key!r} of {owner_cls.__name__} also appears on {other_cls.__name__}"
