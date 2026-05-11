"""Smoke tests for the ``scripts/eval.py`` CLI.

The CLI is exercised in-process via ``main(argv=...)`` against stub
inference + Stockfish engines so the test stays hermetic — no checkpoint
load, no Stockfish binary. The point is to verify that the CLI dispatch,
the post-run metrics aggregation, and the ``report`` subcommand all wire
the run-directory contract together correctly.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import chess
import chess.engine
import pytest
import torch

import scripts.eval as eval_cli
from halluci_mate.chess_tokenizer import ChessTokenizer
from halluci_mate.eval.records import Evaluator, TopKEntry
from halluci_mate.eval.runs import (
    CONFIG_FILENAME,
    METRICS_FILENAME,
    RECORDS_FILENAME,
    RunWriter,
    make_run_id,
)
from halluci_mate.inference import MovePrediction
from tests.helpers.eval_records import DEFAULT_CHECKPOINT, make_per_game_record, make_per_move_record

if TYPE_CHECKING:
    from pathlib import Path

    from halluci_mate.game import Game


class _UniformModel:
    """Returns log-uniform logits at every position; used for perplexity scoring."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def __call__(self, *, input_ids: torch.Tensor) -> Any:
        batch, seq_len = input_ids.shape
        return SimpleNamespace(logits=torch.zeros((batch, seq_len, self.vocab_size)))


class _StubEngine:
    """Stand-in for ``ChessInferenceEngine``. Plays the first legal move.

    Carries ``model`` / ``tokenizer`` / ``device`` attributes so the
    ``perplexity`` evaluator's ``PerplexityScorer`` Protocol is satisfied
    by the same stub used for ``vs-stockfish`` and ``legal-rate``.
    """

    def __init__(self, *, temperature: float = 0.0, top_k: int = 0) -> None:
        self.temperature = temperature
        self.top_k = top_k
        self.tokenizer = ChessTokenizer()
        self.device = torch.device("cpu")
        self.model = _UniformModel(vocab_size=self.tokenizer.vocab_size)

    def predict_with_metadata(
        self,
        game: Game,
        *,
        constrained: bool | None = None,
        record_top_k: int = 5,
    ) -> MovePrediction:
        del record_top_k
        played = next(iter(game.board.legal_moves))
        # ``legal-rate`` invokes us with ``constrained=False`` (no masking, no
        # played move); ``vs-stockfish`` leaves it ``None`` (its default,
        # masking on). Reflect that asymmetry on ``played_move`` / ``mask_used``.
        masked = constrained is not False
        return MovePrediction(
            played_move=played if masked else None,
            model_move_uci=played.uci(),
            raw_sample_move_uci=played.uci(),
            raw_sample_legal=True,
            model_top_k=[TopKEntry(move=played.uci(), logprob=-0.1)],
            mask_used=masked,
        )


class _StubStockfish:
    """Stand-in for ``chess.engine.SimpleEngine``. Plays the last legal move."""

    def __init__(self) -> None:
        self.quit_called = False

    def configure(self, options: dict[str, Any]) -> None:
        del options

    def play(self, board: chess.Board, limit: chess.engine.Limit) -> chess.engine.PlayResult:
        del limit
        return chess.engine.PlayResult(move=list(board.legal_moves)[-1], ponder=None)

    def quit(self) -> None:
        self.quit_called = True


def _patch_engines(monkeypatch: pytest.MonkeyPatch, stub_stockfish: _StubStockfish) -> None:
    monkeypatch.setattr(eval_cli.ChessInferenceEngine, "from_checkpoint", lambda *_a, **_kw: _StubEngine())
    monkeypatch.setattr(eval_cli.chess.engine.SimpleEngine, "popen_uci", lambda *_a, **_kw: stub_stockfish)


def _seed_run(evals_dir: Path, *, with_per_game: bool = True) -> tuple[Path, str]:
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
        if with_per_game:
            writer.append_record(make_per_game_record(event_id=1, run_id=run_id))
    return run_dir, run_id


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
    run_dir, run_id = _seed_run(evals_dir)

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


_TWO_GAME_PGN = """[Event "g1"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0

[Event "g2"]
[Result "0-1"]

1. d4 d5 2. c4 e6 0-1
"""


def _patch_engine_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_cli.ChessInferenceEngine, "from_checkpoint", classmethod(lambda cls, *a, **kw: _StubEngine()))


def test_legal_rate_smoke_with_positions_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`legal-rate --positions` produces a valid run dir + metrics.json."""
    _patch_engine_only(monkeypatch)
    fen_file = tmp_path / "positions.fen"
    fen_file.write_text(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\nrnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2\n",
        encoding="utf-8",
    )
    evals_dir = tmp_path / "evals"

    eval_cli.main(
        [
            "legal-rate",
            "--checkpoint",
            "stub-ckpt",
            "--positions",
            str(fen_file),
            "--evals-dir",
            str(evals_dir),
        ]
    )

    run_dirs = [p for p in evals_dir.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    for filename in (CONFIG_FILENAME, RECORDS_FILENAME, METRICS_FILENAME):
        assert (run_dir / filename).exists(), f"missing {filename} after legal-rate run"

    metrics = json.loads((run_dir / METRICS_FILENAME).read_text(encoding="utf-8"))
    assert metrics["evaluator"] == Evaluator.LEGAL_RATE.value
    # _StubEngine returns played as the legal raw sample → all 2 positions legal.
    assert metrics["legal_rate"]["overall"]["n"] == 2
    assert metrics["legal_rate"]["overall"]["legal"] == 2


def test_legal_rate_smoke_with_pgn_sampling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`legal-rate --sample-from-games` reads a PGN and produces a run dir."""
    _patch_engine_only(monkeypatch)
    pgn_file = tmp_path / "games.pgn"
    pgn_file.write_text(_TWO_GAME_PGN, encoding="utf-8")
    evals_dir = tmp_path / "evals"

    eval_cli.main(
        [
            "legal-rate",
            "--checkpoint",
            "stub-ckpt",
            "--sample-from-games",
            str(pgn_file),
            "--n",
            "3",
            "--seed",
            "1",
            "--evals-dir",
            str(evals_dir),
        ]
    )

    run_dirs = [p for p in evals_dir.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    metrics = json.loads((run_dir / METRICS_FILENAME).read_text(encoding="utf-8"))
    assert metrics["evaluator"] == Evaluator.LEGAL_RATE.value
    assert metrics["legal_rate"]["overall"]["n"] == 3


def test_legal_rate_requires_a_position_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """argparse mutually exclusive group rejects missing source."""
    _patch_engine_only(monkeypatch)
    with pytest.raises(SystemExit):
        eval_cli.main(
            [
                "legal-rate",
                "--checkpoint",
                "stub-ckpt",
                "--evals-dir",
                str(tmp_path / "evals"),
            ]
        )


def test_perplexity_smoke_writes_metrics_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`perplexity --data` produces a valid run dir + metrics.json with NLL/bits."""
    _patch_engine_only(monkeypatch)
    data = tmp_path / "sequences.jsonl"
    data.write_text(
        json.dumps({"id": "g1", "perspective": "white", "moves": ["e2e4", "e7e5"]}) + "\n" + json.dumps({"id": "g2", "perspective": "black", "moves": ["d2d4", "d7d5"]}) + "\n",
        encoding="utf-8",
    )
    evals_dir = tmp_path / "evals"

    eval_cli.main(
        [
            "perplexity",
            "--checkpoint",
            "stub-ckpt",
            "--data",
            str(data),
            "--evals-dir",
            str(evals_dir),
        ]
    )

    run_dirs = [p for p in evals_dir.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    for filename in (CONFIG_FILENAME, RECORDS_FILENAME, METRICS_FILENAME):
        assert (run_dir / filename).exists(), f"missing {filename} after perplexity run"

    metrics = json.loads((run_dir / METRICS_FILENAME).read_text(encoding="utf-8"))
    assert metrics["evaluator"] == Evaluator.PERPLEXITY.value
    assert metrics["num_sequences"] == 2
    # 2 sequences * 2 logprobs each (3 tokens → 2 targets).
    assert metrics["num_tokens"] == 4
    assert metrics["mean_nll"] > 0.0
    assert metrics["bits_per_token"] > 0.0
    assert metrics["perplexity"] > 1.0


def test_perplexity_max_sequences_is_respected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_engine_only(monkeypatch)
    data = tmp_path / "sequences.jsonl"
    data.write_text(
        "".join(json.dumps({"id": f"g{i}", "perspective": "white", "moves": ["e2e4", "e7e5"]}) + "\n" for i in range(5)),
        encoding="utf-8",
    )
    evals_dir = tmp_path / "evals"

    eval_cli.main(
        [
            "perplexity",
            "--checkpoint",
            "stub-ckpt",
            "--data",
            str(data),
            "--max-sequences",
            "2",
            "--evals-dir",
            str(evals_dir),
        ]
    )

    run_dir = next(p for p in evals_dir.iterdir() if p.is_dir())
    metrics = json.loads((run_dir / METRICS_FILENAME).read_text(encoding="utf-8"))
    assert metrics["num_sequences"] == 2


def test_report_recovers_from_corrupt_metrics(tmp_path: Path) -> None:
    """A pre-existing malformed metrics.json must not block recomputation."""
    evals_dir = tmp_path / "evals"
    run_dir, run_id = _seed_run(evals_dir, with_per_game=False)
    (run_dir / METRICS_FILENAME).write_text("not json\n", encoding="utf-8")

    eval_cli.main(["report", run_id, "--evals-dir", str(evals_dir)])

    metrics = json.loads((run_dir / METRICS_FILENAME).read_text(encoding="utf-8"))
    assert metrics["evaluator"] == Evaluator.VS_STOCKFISH.value
