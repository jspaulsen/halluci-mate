"""Tests for the cross-run comparison helpers in ``halluci_mate.eval.compare``.

The Streamlit UI in ``scripts/compare.py`` is intentionally not exercised here
— these tests cover the pure data layer it sits on top of.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from halluci_mate.eval.compare import (
    discover_runs,
    latest_run_for,
    list_checkpoints,
    load_or_compute_metrics,
    runs_for,
)
from halluci_mate.eval.records import Evaluator
from halluci_mate.eval.runs import CONFIG_FILENAME, METRICS_FILENAME, RECORDS_FILENAME, RunWriter
from tests.helpers.eval_records import make_per_game_record, make_per_move_record

if TYPE_CHECKING:
    from pathlib import Path


def _write_run(
    evals_dir: Path,
    run_id: str,
    checkpoint: str,
    evaluator: Evaluator,
    *,
    extra_config: dict[str, Any] | None = None,
) -> Path:
    run_dir = evals_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {"evaluator": evaluator.value, "checkpoint": checkpoint}
    if extra_config:
        config.update(extra_config)
    (run_dir / CONFIG_FILENAME).write_text(json.dumps(config), encoding="utf-8")
    (run_dir / RECORDS_FILENAME).write_text("", encoding="utf-8")
    return run_dir


def test_discover_runs_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    assert discover_runs(tmp_path / "does-not-exist") == []


def test_discover_runs_parses_run_metadata(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "2026-04-30T01-29-13_model-a-ckpt1_vs-stockfish",
        "runs-v1a/model-a/checkpoint-1",
        Evaluator.VS_STOCKFISH,
    )
    runs = discover_runs(tmp_path)
    assert len(runs) == 1
    entry = runs[0]
    assert entry.checkpoint == "runs-v1a/model-a/checkpoint-1"
    assert entry.evaluator is Evaluator.VS_STOCKFISH
    assert entry.timestamp == datetime(2026, 4, 30, 1, 29, 13)


def test_discover_runs_skips_malformed_directories(tmp_path: Path) -> None:
    (tmp_path / "no-config-here").mkdir()
    bad_json = tmp_path / "bad-config_x_vs-stockfish"
    bad_json.mkdir()
    (bad_json / CONFIG_FILENAME).write_text("{not json", encoding="utf-8")
    missing_fields = tmp_path / "2026-04-30T01-29-13_x_vs-stockfish"
    missing_fields.mkdir()
    (missing_fields / CONFIG_FILENAME).write_text(json.dumps({"evaluator": "vs_stockfish"}), encoding="utf-8")
    unparseable_ts = _write_run(tmp_path, "not-a-timestamp_x_vs-stockfish", "ckpt", Evaluator.VS_STOCKFISH)
    assert unparseable_ts.exists()

    assert discover_runs(tmp_path) == []


def test_list_checkpoints_dedupes_and_sorts(tmp_path: Path) -> None:
    _write_run(tmp_path, "2026-04-30T01-00-00_a_vs-stockfish", "ckpt-b", Evaluator.VS_STOCKFISH)
    _write_run(tmp_path, "2026-04-30T02-00-00_a_legal-rate", "ckpt-b", Evaluator.LEGAL_RATE)
    _write_run(tmp_path, "2026-04-30T03-00-00_a_perplexity", "ckpt-a", Evaluator.PERPLEXITY)
    assert list_checkpoints(discover_runs(tmp_path)) == ["ckpt-a", "ckpt-b"]


def test_latest_run_for_picks_newest_timestamp(tmp_path: Path) -> None:
    _write_run(tmp_path, "2026-04-30T01-00-00_a_vs-stockfish", "ckpt-a", Evaluator.VS_STOCKFISH)
    _write_run(tmp_path, "2026-04-30T03-00-00_a_vs-stockfish", "ckpt-a", Evaluator.VS_STOCKFISH)
    _write_run(tmp_path, "2026-04-30T02-00-00_a_vs-stockfish", "ckpt-a", Evaluator.VS_STOCKFISH)
    runs = discover_runs(tmp_path)
    latest = latest_run_for(runs, "ckpt-a", Evaluator.VS_STOCKFISH)
    assert latest is not None
    assert latest.run_id.startswith("2026-04-30T03-00-00")


def test_latest_run_for_returns_none_when_no_match(tmp_path: Path) -> None:
    _write_run(tmp_path, "2026-04-30T01-00-00_a_vs-stockfish", "ckpt-a", Evaluator.VS_STOCKFISH)
    runs = discover_runs(tmp_path)
    assert latest_run_for(runs, "ckpt-missing", Evaluator.VS_STOCKFISH) is None
    assert latest_run_for(runs, "ckpt-a", Evaluator.PERPLEXITY) is None


def test_runs_for_returns_all_matches_newest_first(tmp_path: Path) -> None:
    _write_run(tmp_path, "2026-04-30T01-00-00_a_vs-stockfish", "ckpt-a", Evaluator.VS_STOCKFISH)
    _write_run(tmp_path, "2026-04-30T03-00-00_b_vs-stockfish", "ckpt-a", Evaluator.VS_STOCKFISH)
    _write_run(tmp_path, "2026-04-30T02-00-00_c_vs-stockfish", "ckpt-a", Evaluator.VS_STOCKFISH)
    _write_run(tmp_path, "2026-04-30T05-00-00_d_vs-stockfish", "ckpt-b", Evaluator.VS_STOCKFISH)
    runs = discover_runs(tmp_path)
    ordered = runs_for(runs, "ckpt-a", Evaluator.VS_STOCKFISH)
    assert [r.run_id for r in ordered] == [
        "2026-04-30T03-00-00_b_vs-stockfish",
        "2026-04-30T02-00-00_c_vs-stockfish",
        "2026-04-30T01-00-00_a_vs-stockfish",
    ]
    assert runs_for(runs, "ckpt-missing", Evaluator.VS_STOCKFISH) == []


def test_load_or_compute_metrics_reads_existing_file(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "2026-04-30T01-00-00_a_vs-stockfish", "ckpt-a", Evaluator.VS_STOCKFISH)
    cached = {"evaluator": "vs_stockfish", "win_rate": {"overall": {"wins": 7}}}
    (run_dir / METRICS_FILENAME).write_text(json.dumps(cached), encoding="utf-8")
    [entry] = discover_runs(tmp_path)
    assert load_or_compute_metrics(entry) == cached


def test_load_or_compute_metrics_falls_back_to_compute_all(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        "2026-04-30T01-00-00_a_vs-stockfish",
        "ckpt-a",
        Evaluator.VS_STOCKFISH,
        extra_config={"stockfish_skill": 3},
    )
    with RunWriter(run_dir) as writer:
        writer.append_record(make_per_move_record(0))
        writer.append_record(make_per_game_record(1, result="1-0"))
    [entry] = discover_runs(tmp_path)
    metrics = load_or_compute_metrics(entry)
    assert metrics["evaluator"] == "vs_stockfish"
    assert metrics["win_rate"]["overall"]["wins"] == 1
