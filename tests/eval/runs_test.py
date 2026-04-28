"""Tests for run-id formatting and run directory writer/reader."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from halluci_mate.eval.records import Evaluator
from halluci_mate.eval.runs import (
    CONFIG_FILENAME,
    GAMES_PGN_FILENAME,
    METRICS_FILENAME,
    RECORDS_FILENAME,
    RunReader,
    RunWriter,
    make_run_id,
)
from tests.eval.conftest import make_per_move_record, make_per_puzzle_record

if TYPE_CHECKING:
    from pathlib import Path

FIXED_NOW = datetime(2026, 4, 19, 20, 15, 0, tzinfo=UTC)


def test_make_run_id_formats_components() -> None:
    run_id = make_run_id("marvelous-deer-608-ckpt9660", Evaluator.VS_STOCKFISH, now=FIXED_NOW)
    assert run_id == "2026-04-19T20-15-00_marvelous-deer-608-ckpt9660_vs-stockfish"


def test_make_run_id_uses_current_utc_time_when_now_omitted() -> None:
    run_id = make_run_id("ckpt", Evaluator.PERPLEXITY)
    timestamp, _, _ = run_id.partition("_")
    parsed = datetime.strptime(timestamp, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=UTC)
    assert abs((datetime.now(UTC) - parsed).total_seconds()) < 60


def test_make_run_id_uses_hyphenated_evaluator_slug() -> None:
    """Underscore evaluator names (`vs_stockfish`) must serialize as hyphens
    in run-ids so the run-id stays splittable on `_`."""
    run_id = make_run_id("ckpt", Evaluator.LEGAL_RATE, now=FIXED_NOW)
    assert run_id.endswith("_legal-rate")


def test_make_run_id_rejects_underscore_in_checkpoint_tag() -> None:
    with pytest.raises(ValueError, match="must not contain '_'"):
        make_run_id("bad_tag", Evaluator.VS_STOCKFISH, now=FIXED_NOW)


def test_run_writer_creates_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "evals" / "run-1"
    RunWriter(run_dir)
    assert run_dir.is_dir()


def test_run_writer_writes_config_metrics_and_pgn(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    writer = RunWriter(run_dir)
    config = {"evaluator": "vs_stockfish", "games": 2}
    metrics = {"win_rate": 0.5}

    writer.write_config(config)
    writer.write_metrics(metrics)
    writer.write_pgn('[Event "x"]\n\n1. e4 *\n')

    reader = RunReader(run_dir)
    assert reader.read_config() == config
    assert reader.read_metrics() == metrics
    assert reader.read_pgn() == '[Event "x"]\n\n1. e4 *\n'
    assert (run_dir / CONFIG_FILENAME).exists()
    assert (run_dir / METRICS_FILENAME).exists()
    assert (run_dir / GAMES_PGN_FILENAME).exists()


def test_run_writer_appends_records_round_trip(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    move_a = make_per_move_record(event_id=0)
    move_b = make_per_move_record(event_id=1)
    puzzle = make_per_puzzle_record(event_id=2)

    with RunWriter(run_dir) as writer:
        writer.append_record(move_a)
        writer.append_record(move_b)
        writer.append_record(puzzle)

    reader = RunReader(run_dir)
    assert reader.read_records() == [move_a, move_b, puzzle]


def test_records_jsonl_is_one_record_per_line(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    with RunWriter(run_dir) as writer:
        writer.append_record(make_per_move_record(event_id=0))
        writer.append_record(make_per_move_record(event_id=1))

    lines = (run_dir / RECORDS_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)


def test_run_writer_holds_records_file_open_for_run_lifetime(tmp_path: Path) -> None:
    """Single open per run, not per record — see PR #10 review."""
    run_dir = tmp_path / "run"
    with RunWriter(run_dir) as writer:
        writer.append_record(make_per_move_record(event_id=0))
        fp = writer._records_fp
        assert fp is not None
        assert not fp.closed
        writer.append_record(make_per_move_record(event_id=1))
        assert writer._records_fp is fp
    assert fp.closed


def test_run_writer_append_record_outside_context_raises(tmp_path: Path) -> None:
    writer = RunWriter(tmp_path / "run")
    with pytest.raises(RuntimeError, match="context manager"):
        writer.append_record(make_per_move_record(event_id=0))
