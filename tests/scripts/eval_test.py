"""Smoke tests for the ``scripts/eval.py`` CLI.

The CLI is exercised in-process via ``main(argv=...)`` against stub
inference + Stockfish engines so the test stays hermetic — no checkpoint
load, no Stockfish binary. The point is to verify that the CLI dispatch,
the post-run metrics aggregation, and the ``report`` subcommand all wire
the run-directory contract together correctly.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import chess
import chess.engine
import pytest

import scripts.eval as eval_cli
from halluci_mate.eval.records import Evaluator, TopKEntry
from halluci_mate.eval.runs import (
    CONFIG_FILENAME,
    METRICS_FILENAME,
    RECORDS_FILENAME,
    RunWriter,
    make_run_id,
)
from halluci_mate.inference import MovePrediction
from tests.eval.conftest import DEFAULT_CHECKPOINT, make_per_game_record, make_per_move_record

if TYPE_CHECKING:
    from pathlib import Path

    from halluci_mate.game import Game


class _StubEngine:
    """Stand-in for ``ChessInferenceEngine``. Plays the first legal move."""

    def __init__(self, *, temperature: float = 0.0, top_k: int = 0) -> None:
        self.temperature = temperature
        self.top_k = top_k

    def predict_with_metadata(
        self,
        game: Game,
        *,
        constrained: bool | None = None,
        record_top_k: int = 5,
    ) -> MovePrediction:
        del constrained, record_top_k
        played = next(iter(game.board.legal_moves))
        return MovePrediction(
            played_move=played,
            model_move_uci=played.uci(),
            raw_sample_move_uci=played.uci(),
            raw_sample_legal=True,
            model_top_k=[TopKEntry(move=played.uci(), logprob=-0.1)],
            mask_used=True,
        )


class _StubStockfish:
    """Stand-in for ``chess.engine.SimpleEngine``. Plays the last legal move."""

    def __init__(self) -> None:
        self.configured: dict[str, Any] = {}
        self.quit_called = False

    def configure(self, options: dict[str, Any]) -> None:
        self.configured = dict(options)

    def play(self, board: chess.Board, limit: chess.engine.Limit) -> chess.engine.PlayResult:
        del limit
        return chess.engine.PlayResult(move=list(board.legal_moves)[-1], ponder=None)

    def quit(self) -> None:
        self.quit_called = True


def _patch_engines(monkeypatch: pytest.MonkeyPatch, stub_stockfish: _StubStockfish) -> None:
    monkeypatch.setattr(eval_cli.ChessInferenceEngine, "from_checkpoint", classmethod(lambda cls, *a, **kw: _StubEngine()))
    monkeypatch.setattr(eval_cli.chess.engine.SimpleEngine, "popen_uci", staticmethod(lambda *_a, **_kw: stub_stockfish))


def test_vs_stockfish_smoke_writes_metrics_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--games 1` produces a populated run directory including metrics.json."""
    stockfish = _StubStockfish()
    _patch_engines(monkeypatch, stockfish)
    evals_dir = tmp_path / "evals"

    eval_cli.main(
        [
            "vs-stockfish",
            "--checkpoint",
            "stub-ckpt",
            "--games",
            "1",
            "--max-plies",
            "4",
            "--halluci-color",
            "white",
            "--evals-dir",
            str(evals_dir),
        ]
    )

    assert stockfish.quit_called, "CLI must close the Stockfish process even on the happy path"

    run_dirs = [p for p in evals_dir.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    for filename in (CONFIG_FILENAME, RECORDS_FILENAME, METRICS_FILENAME):
        assert (run_dir / filename).exists(), f"missing {filename} after vs-stockfish run"

    metrics = json.loads((run_dir / METRICS_FILENAME).read_text(encoding="utf-8"))
    assert metrics["evaluator"] == Evaluator.VS_STOCKFISH.value
    assert metrics["win_rate"]["overall"]["games"] == 1
    assert "by_phase" in metrics["legal_rate"]
    assert "by_model_side" in metrics["legal_rate"]


def test_report_recomputes_metrics_without_touching_records(tmp_path: Path) -> None:
    """`report <run-id>` rewrites metrics.json from existing records.jsonl."""
    evals_dir = tmp_path / "evals"
    run_id = make_run_id("stub-ckpt", Evaluator.VS_STOCKFISH)
    run_dir = evals_dir / run_id

    writer = RunWriter(run_dir)
    writer.write_config(
        {
            "evaluator": Evaluator.VS_STOCKFISH.value,
            "run_id": run_id,
            "checkpoint": DEFAULT_CHECKPOINT,
            "stockfish_skill": 0,
        }
    )
    with writer:
        writer.append_record(make_per_move_record(event_id=0, run_id=run_id))
        writer.append_record(make_per_game_record(event_id=1, run_id=run_id))

    records_bytes_before = (run_dir / RECORDS_FILENAME).read_bytes()

    eval_cli.main(["report", run_id, "--evals-dir", str(evals_dir)])

    metrics = json.loads((run_dir / METRICS_FILENAME).read_text(encoding="utf-8"))
    assert metrics["evaluator"] == Evaluator.VS_STOCKFISH.value
    assert metrics["win_rate"]["overall"]["games"] == 1
    assert metrics["win_rate"]["overall"]["wins"] == 1
    assert metrics["legal_rate"]["overall"]["n"] == 1
    assert metrics["legal_rate"]["overall"]["legal"] == 1

    assert (run_dir / RECORDS_FILENAME).read_bytes() == records_bytes_before


def test_report_missing_run_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run directory not found"):
        eval_cli.main(["report", "does-not-exist", "--evals-dir", str(tmp_path)])


def test_report_recovers_from_corrupt_metrics(tmp_path: Path) -> None:
    """A pre-existing malformed metrics.json must not block recomputation."""
    evals_dir = tmp_path / "evals"
    run_id = make_run_id("stub-ckpt", Evaluator.VS_STOCKFISH)
    run_dir = evals_dir / run_id

    writer = RunWriter(run_dir)
    writer.write_config(
        {
            "evaluator": Evaluator.VS_STOCKFISH.value,
            "run_id": run_id,
            "checkpoint": DEFAULT_CHECKPOINT,
            "stockfish_skill": 0,
        }
    )
    with writer:
        writer.append_record(make_per_move_record(event_id=0, run_id=run_id))
    (run_dir / METRICS_FILENAME).write_text("not json\n", encoding="utf-8")

    eval_cli.main(["report", run_id, "--evals-dir", str(evals_dir)])

    metrics = json.loads((run_dir / METRICS_FILENAME).read_text(encoding="utf-8"))
    assert metrics["evaluator"] == Evaluator.VS_STOCKFISH.value
