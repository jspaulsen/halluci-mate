"""Unit tests for scripts/sweep_summary.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

import scripts.sweep_summary as sweep_summary
from halluci_mate.eval.records import TopKEntry
from halluci_mate.eval.runs import RunWriter
from tests.helpers.eval_records import make_per_move_record

if TYPE_CHECKING:
    from pathlib import Path


def _write_run(evals_dir: Path, run_id: str, *, search_k: int, search_margin: float, score_rate: float, records: list) -> None:
    writer = RunWriter(evals_dir / run_id)
    writer.write_config(
        {
            "evaluator": "vs_stockfish",
            "run_id": run_id,
            "checkpoint": "jspaulsen/halluci-mate-v2d",
            "games": 30,
            "search": True,
            "search_k": search_k,
            "search_margin": search_margin,
            "search_leaf": "material-king-safety",
        }
    )
    writer.write_metrics(
        {
            "win_rate": {"overall": {"score_rate": score_rate, "wins": 1, "draws": 2, "losses": 27}},
            "centipawn_loss": {"overall": {"median": 14.0, "p95": 178.0}},
            "blunder_rate": {"overall": {"rate": 0.043}},
            "legal_rate": {"overall": {"rate": 0.983}},
            "tactical_oversight_rate": {"overall": {"rate": 0.12}},
        }
    )
    with writer as open_writer:
        for record in records:
            open_writer.append_record(record)


def test_collect_summaries_sorts_by_score_and_parses_knobs(tmp_path: Path) -> None:
    agree = make_per_move_record(0, model_move="e2e4", model_top_k=[TopKEntry(move="e2e4", logprob=-0.1)])
    override = make_per_move_record(1, model_move="d2d4", model_top_k=[TopKEntry(move="g1f3", logprob=-0.2)])
    _write_run(tmp_path, "2026-05-25T01-00-00_v2d-gridk3t10_vs-stockfish", search_k=3, search_margin=1.0, score_rate=0.10, records=[agree, override])
    _write_run(tmp_path, "2026-05-25T02-00-00_v2d-gridk8t05_vs-stockfish", search_k=8, search_margin=0.5, score_rate=0.20, records=[override])
    _write_run(tmp_path, "2026-05-25T03-00-00_other-run_vs-stockfish", search_k=3, search_margin=1.0, score_rate=0.99, records=[agree])

    summaries = sweep_summary.collect_summaries(tmp_path, "v2d-grid")

    # Sorted by score_rate desc; the non-matching "other-run" tag is excluded.
    assert [(cell.search_k, cell.search_margin) for cell in summaries] == [(8, 0.5), (3, 1.0)]
    assert summaries[0].score_rate == 0.20
    assert summaries[1].override_rate == 0.5  # 1 of 2 decisions overridden


def test_override_rate_counts_played_vs_argmax_disagreements() -> None:
    records = [
        make_per_move_record(0, model_move="e2e4", model_top_k=[TopKEntry(move="e2e4", logprob=-0.1)]),  # agree
        make_per_move_record(1, model_move="d2d4", model_top_k=[TopKEntry(move="g1f3", logprob=-0.2)]),  # override
        make_per_move_record(2, model_move="c2c4", model_top_k=[TopKEntry(move="b1c3", logprob=-0.3)]),  # override
    ]
    assert sweep_summary.override_rate(records) == 2 / 3


def test_override_rate_is_none_without_eligible_records() -> None:
    assert sweep_summary.override_rate([]) is None
    assert sweep_summary.override_rate([make_per_move_record(0, model_top_k=[])]) is None
