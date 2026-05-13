"""Unit tests for the DPO preference-pair exporter."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from halluci_mate.eval.dpo_export import (
    DEFAULT_QUALITY_THRESHOLD_CP,
    DpoExportError,
    DpoFlavor,
    build_legality_pairs,
    build_quality_pairs,
    export_dpo,
)
from halluci_mate.eval.records import Evaluator
from halluci_mate.eval.runs import RunWriter
from tests.helpers.eval_records import (
    DEFAULT_CHECKPOINT,
    DEFAULT_RUN_ID,
    START_FEN,
    make_per_game_record,
    make_per_move_record,
)

if TYPE_CHECKING:
    from pathlib import Path

    from halluci_mate.eval.records import PerMoveRecord

# Distinct from START_FEN so dedup-by-fen tests can write two records that
# share a position and verify only the first survives.
SECOND_FEN = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"


def _illegal_raw_record(event_id: int, **overrides: object) -> PerMoveRecord:
    """Per-move record where masking rescued an illegal raw sample."""
    fields: dict[str, object] = {
        "mask_used": True,
        "model_move": "e2e4",
        "raw_sample_move": "e2e9",
        "raw_sample_legal": False,
    }
    fields.update(overrides)
    return make_per_move_record(event_id=event_id, **fields)


def _blunder_record(event_id: int, *, cpl: int, **overrides: object) -> PerMoveRecord:
    """Per-move record with populated Stockfish analysis and ``centipawn_loss == cpl``."""
    fields: dict[str, object] = {
        "model_move": "a2a3",
        "sf_best_move": "e2e4",
        "sf_eval_before_cp": 0,
        "sf_eval_after_cp": -cpl,
        "centipawn_loss": cpl,
        "is_blunder": cpl > DEFAULT_QUALITY_THRESHOLD_CP,
    }
    fields.update(overrides)
    return make_per_move_record(event_id=event_id, **fields)


def test_build_legality_pairs_picks_only_illegal_raw_records() -> None:
    records = [
        _illegal_raw_record(event_id=0),
        # Legal raw sample → no pair.
        make_per_move_record(event_id=1),
        # Unmasked illegal raw → both sides illegal, excluded.
        make_per_move_record(event_id=2, mask_used=False, raw_sample_legal=False, raw_sample_move="e2e9", model_move="e2e9"),
        _illegal_raw_record(event_id=3, fen_before=SECOND_FEN, raw_sample_move="d2d9"),
    ]

    pairs = list(build_legality_pairs(records))

    assert len(pairs) == 2
    assert pairs[0].chosen == "e2e4"
    assert pairs[0].rejected == "e2e9"
    assert pairs[1].prompt == SECOND_FEN
    assert pairs[1].rejected == "d2d9"


def test_build_quality_pairs_require_consequential_drops_lost_positions() -> None:
    """``require_consequential`` keeps only moves played from positions
    not yet that-far-down (model-perspective eval-before >= -300 cp)."""
    records = [
        # Consequential — model is still in the game.
        _blunder_record(event_id=0, cpl=300, sf_eval_before_cp=-200),
        # Already lost: eval_before below the threshold.
        _blunder_record(event_id=1, cpl=400, sf_eval_before_cp=-500),
        # Black model: eval is sign-flipped, +500 cp white-relative = -500 model-relative.
        _blunder_record(event_id=2, cpl=400, sf_eval_before_cp=500, model_side="black"),
    ]
    pairs = list(build_quality_pairs(records, threshold=200, require_consequential=True))
    assert len(pairs) == 1
    assert pairs[0].rejected == "a2a3"


def test_build_quality_pairs_exclude_repetition_drops_recurring_positions() -> None:
    """``exclude_repetition`` drops moves whose position-key recurs in
    the same game from both numerator and denominator."""
    pos_a = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    pos_a_bumped = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 5 3"
    pos_b = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"
    records = [
        _blunder_record(event_id=0, cpl=300, game_id="g0", fen_before=pos_a),
        _blunder_record(event_id=1, cpl=300, game_id="g0", fen_before=pos_a_bumped),
        _blunder_record(event_id=2, cpl=300, game_id="g0", fen_before=pos_b),
    ]
    pairs = list(build_quality_pairs(records, threshold=200, exclude_repetition=True))
    assert len(pairs) == 1
    assert pairs[0].prompt == pos_b


def test_build_quality_pairs_filters_compose() -> None:
    """Both filters applied together drop anything that fails either."""
    pos_a = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    records = [
        # In-lost AND repeats → dropped by both.
        _blunder_record(event_id=0, cpl=300, game_id="g0", fen_before=pos_a, sf_eval_before_cp=-700),
        _blunder_record(event_id=1, cpl=300, game_id="g0", fen_before=pos_a, sf_eval_before_cp=-700),
        # Consequential and unique → survives.
        _blunder_record(event_id=2, cpl=300, game_id="g0", fen_before=SECOND_FEN, sf_eval_before_cp=-100),
    ]
    pairs = list(build_quality_pairs(records, threshold=200, require_consequential=True, exclude_repetition=True))
    assert len(pairs) == 1
    assert pairs[0].prompt == SECOND_FEN


def test_export_dpo_passes_quality_filters_through(tmp_path: Path) -> None:
    """End-to-end: the CLI-facing entry point honours both filter flags."""
    pos_a = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    run_dir = tmp_path / "run"
    _seed_run(
        run_dir,
        [
            _blunder_record(event_id=0, cpl=300, sf_eval_before_cp=-700),  # in-lost
            _blunder_record(event_id=1, cpl=300, game_id="gx", fen_before=pos_a, sf_eval_before_cp=-100),
            _blunder_record(event_id=2, cpl=300, game_id="gx", fen_before=pos_a, sf_eval_before_cp=-100),  # repetition with prev
            _blunder_record(event_id=3, cpl=300, fen_before=SECOND_FEN, sf_eval_before_cp=-100),  # survivor
        ],
        analyze=True,
    )
    output = tmp_path / "out.jsonl"
    n = export_dpo(
        run_dir=run_dir,
        output=output,
        flavor=DpoFlavor.QUALITY,
        threshold=200,
        require_consequential=True,
        exclude_repetition=True,
    )
    assert n == 1
    [pair] = _read_jsonl(output)
    assert pair["prompt"] == SECOND_FEN


def test_build_quality_pairs_thresholds_and_skips_missing_analysis() -> None:
    records = [
        _blunder_record(event_id=0, cpl=300),
        # Below threshold.
        _blunder_record(event_id=1, cpl=50),
        # Stockfish agrees with the model → no pair even with nonzero CPL.
        _blunder_record(event_id=2, cpl=300, model_move="e2e4"),
        # Missing analysis fields (non-`--sf-analyze` record).
        make_per_move_record(event_id=3),
    ]

    pairs = list(build_quality_pairs(records, threshold=200))

    assert len(pairs) == 1
    assert pairs[0].chosen == "e2e4"
    assert pairs[0].rejected == "a2a3"


def _seed_run(run_dir: Path, records: list[PerMoveRecord], *, analyze: bool) -> None:
    writer = RunWriter(run_dir)
    writer.write_config(
        {
            "evaluator": Evaluator.VS_STOCKFISH.value,
            "run_id": DEFAULT_RUN_ID,
            "checkpoint": DEFAULT_CHECKPOINT,
            "analyze": analyze,
        }
    )
    with writer:
        for record in records:
            writer.append_record(record)
        # A trailing per-game record exercises the PerMoveRecord filter in `export_dpo`.
        writer.append_record(make_per_game_record(event_id=len(records)))


def _read_jsonl(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_export_dpo_legality_writes_pairs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _seed_run(run_dir, [_illegal_raw_record(event_id=0)], analyze=False)
    output = tmp_path / "out.jsonl"

    n = export_dpo(run_dir=run_dir, output=output, flavor=DpoFlavor.LEGALITY)

    assert n == 1
    [pair] = _read_jsonl(output)
    assert pair == {"prompt": START_FEN, "chosen": "e2e4", "rejected": "e2e9"}


def test_export_dpo_quality_requires_analyze(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _seed_run(run_dir, [_blunder_record(event_id=0, cpl=300)], analyze=False)
    output = tmp_path / "out.jsonl"

    with pytest.raises(DpoExportError, match="--sf-analyze"):
        export_dpo(run_dir=run_dir, output=output, flavor=DpoFlavor.QUALITY)

    assert not output.exists(), "exporter must not write a partial file when the precondition fails"


def test_export_dpo_quality_emits_pairs_above_threshold(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _seed_run(
        run_dir,
        [
            _blunder_record(event_id=0, cpl=300),
            _blunder_record(event_id=1, cpl=50),
        ],
        analyze=True,
    )
    output = tmp_path / "out.jsonl"

    n = export_dpo(run_dir=run_dir, output=output, flavor=DpoFlavor.QUALITY, threshold=200)

    assert n == 1
    [pair] = _read_jsonl(output)
    assert pair["chosen"] == "e2e4"
    assert pair["rejected"] == "a2a3"


def test_export_dpo_both_unions_flavors_and_can_dedup_by_fen(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    # Same FEN on the legality and quality records — dedup should keep the
    # earlier (legality) pair, since `_dedup_by_fen` is first-wins.
    _seed_run(
        run_dir,
        [
            _illegal_raw_record(event_id=0),
            _blunder_record(event_id=1, cpl=300),
        ],
        analyze=True,
    )
    output_with_dups = tmp_path / "with_dups.jsonl"
    output_deduped = tmp_path / "deduped.jsonl"

    n_dup = export_dpo(run_dir=run_dir, output=output_with_dups, flavor=DpoFlavor.BOTH)
    n_dedup = export_dpo(run_dir=run_dir, output=output_deduped, flavor=DpoFlavor.BOTH, dedup_by_fen=True)

    assert n_dup == 2
    assert n_dedup == 1
    [survivor] = _read_jsonl(output_deduped)
    assert survivor["rejected"] == "e2e9", "legality pair (appended first) wins the dedup collision"
