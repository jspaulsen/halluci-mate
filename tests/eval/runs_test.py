"""Tests for run-id formatting and run directory writer/reader."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from halluci_mate.eval.records import (
    Evaluator,
    PerMoveRecord,
    PerPuzzleRecord,
    Phase,
    Side,
    TopKEntry,
)
from halluci_mate.eval.runs import (
    CONFIG_FILENAME,
    GAMES_PGN_FILENAME,
    METRICS_FILENAME,
    RECORDS_FILENAME,
    RunReader,
    RunWriter,
    make_run_id,
)

if TYPE_CHECKING:
    from pathlib import Path

FIXED_NOW = datetime(2026, 4, 19, 20, 15, 0, tzinfo=UTC)


def _move_record(event_id: int) -> PerMoveRecord:
    return PerMoveRecord(
        run_id="run-x",
        event_id=event_id,
        evaluator=Evaluator.VS_STOCKFISH,
        checkpoint="ckpt",
        game_id=f"g{event_id}",
        ply=event_id,
        phase=Phase.OPENING,
        side_to_move=Side.WHITE,
        model_side=Side.WHITE,
        fen_before="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        legal_moves=["e2e4"],
        model_move="e2e4",
        model_top_k=[TopKEntry(move="e2e4", logprob=-0.1)],
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


def test_make_run_id_formats_components() -> None:
    run_id = make_run_id("marvelous-deer-608-ckpt9660", "vs-stockfish", now=FIXED_NOW)
    assert run_id == "2026-04-19T20-15-00_marvelous-deer-608-ckpt9660_vs-stockfish"


def test_make_run_id_uses_current_utc_time_when_now_omitted() -> None:
    run_id = make_run_id("ckpt", "perplexity")
    timestamp, _, _ = run_id.partition("_")
    parsed = datetime.strptime(timestamp, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=UTC)
    assert abs((datetime.now(UTC) - parsed).total_seconds()) < 60


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
    writer = RunWriter(run_dir)

    move_a = _move_record(0)
    move_b = _move_record(1)
    puzzle = PerPuzzleRecord(
        run_id="run-x",
        event_id=2,
        evaluator=Evaluator.PUZZLES,
        checkpoint="ckpt",
        puzzle_id="p1",
        rating=1500,
        themes=["fork"],
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        solution=["e2e4"],
        model_attempt=["e2e4"],
        solved=True,
    )

    writer.append_record(move_a)
    writer.append_record(move_b)
    writer.append_record(puzzle)

    reader = RunReader(run_dir)
    assert list(reader.read_records()) == [move_a, move_b, puzzle]


def test_records_jsonl_is_one_record_per_line(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    writer = RunWriter(run_dir)
    writer.append_record(_move_record(0))
    writer.append_record(_move_record(1))

    lines = (run_dir / RECORDS_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)
