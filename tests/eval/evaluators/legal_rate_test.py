"""Tests for the ``legal_rate`` evaluator.

Exercised end-to-end against a stub engine so the evaluator's record-emission
+ run-directory wiring is tested without loading a real checkpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import chess
import pytest

from halluci_mate.eval.evaluators.legal_rate import (
    LegalRateConfig,
    run_legal_rate,
)
from halluci_mate.eval.records import Evaluator, PerLegalRateRecord
from halluci_mate.eval.runs import CONFIG_FILENAME, RECORDS_FILENAME, RunReader
from halluci_mate.inference import MovePrediction
from tests.eval.conftest import DEFAULT_CHECKPOINT, DEFAULT_RUN_ID

if TYPE_CHECKING:
    from pathlib import Path

    from halluci_mate.game import Game


_TWO_GAME_PGN = """[Event "g1"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0

[Event "g2"]
[Result "0-1"]

1. d4 d5 2. c4 e6 0-1
"""


class _ConstantEngine:
    """Returns a fixed UCI as the unconstrained top-1, with a fixed legality bit."""

    def __init__(self, *, raw_uci: str = "e2e4", raw_legal: bool = True) -> None:
        self._raw_uci = raw_uci
        self._raw_legal = raw_legal
        self.calls = 0

    def predict_with_metadata(
        self,
        game: Game,
        *,
        constrained: bool | None = None,
        record_top_k: int = 5,
    ) -> MovePrediction:
        del game
        self.calls += 1
        # ``legal_rate`` must request unconstrained scoring.
        assert constrained is False
        assert record_top_k == 0
        return MovePrediction(
            played_move=None,
            model_move_uci=self._raw_uci,
            raw_sample_move_uci=self._raw_uci,
            raw_sample_legal=self._raw_legal,
            model_top_k=[],
            mask_used=False,
        )


def _read_legal_records(run_dir: Path) -> list[PerLegalRateRecord]:
    records = RunReader(run_dir).read_records()
    return [r for r in records if isinstance(r, PerLegalRateRecord)]


def test_positions_file_emits_one_record_per_fen(tmp_path: Path) -> None:
    fen_file = tmp_path / "positions.fen"
    fen_file.write_text(
        "\n".join(
            [
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "",  # blank lines must be skipped
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "run"
    config = LegalRateConfig(positions_path=fen_file)
    engine = _ConstantEngine(raw_uci="e2e4", raw_legal=True)

    n = run_legal_rate(
        engine=engine,
        config=config,
        run_dir=run_dir,
        run_id=DEFAULT_RUN_ID,
        checkpoint=DEFAULT_CHECKPOINT,
    )

    assert n == 2
    assert engine.calls == 2
    assert (run_dir / CONFIG_FILENAME).exists()
    assert (run_dir / RECORDS_FILENAME).exists()

    records = _read_legal_records(run_dir)
    assert [r.event_id for r in records] == [0, 1]
    assert [r.position_id for r in records] == ["positions.fen:1", "positions.fen:3"]
    assert all(r.evaluator == Evaluator.LEGAL_RATE for r in records)
    assert all(r.run_id == DEFAULT_RUN_ID for r in records)
    assert all(r.checkpoint == DEFAULT_CHECKPOINT for r in records)
    assert all(r.model_move == "e2e4" for r in records)
    assert all(r.legal is True for r in records)

    config_payload = RunReader(run_dir).read_config()
    assert config_payload["evaluator"] == Evaluator.LEGAL_RATE.value
    assert config_payload["positions_path"] == str(fen_file)
    assert config_payload["sample_from_games_path"] is None


def test_records_legal_false_when_engine_returns_illegal(tmp_path: Path) -> None:
    fen_file = tmp_path / "p.fen"
    fen_file.write_text("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n", encoding="utf-8")

    run_dir = tmp_path / "run"
    config = LegalRateConfig(positions_path=fen_file)
    engine = _ConstantEngine(raw_uci="<UNK>", raw_legal=False)

    run_legal_rate(
        engine=engine,
        config=config,
        run_dir=run_dir,
        run_id=DEFAULT_RUN_ID,
        checkpoint=DEFAULT_CHECKPOINT,
    )

    records = _read_legal_records(run_dir)
    assert len(records) == 1
    assert records[0].legal is False
    assert records[0].model_move == "<UNK>"


def test_sample_from_games_emits_records_with_replayed_prefix(tmp_path: Path) -> None:
    pgn_file = tmp_path / "games.pgn"
    pgn_file.write_text(_TWO_GAME_PGN, encoding="utf-8")
    run_dir = tmp_path / "run"

    config = LegalRateConfig(sample_from_games_path=pgn_file, sample_n=4, seed=7)
    engine = _ConstantEngine(raw_uci="a2a4", raw_legal=False)

    n = run_legal_rate(
        engine=engine,
        config=config,
        run_dir=run_dir,
        run_id=DEFAULT_RUN_ID,
        checkpoint=DEFAULT_CHECKPOINT,
    )

    # Two 4-ply games → 8 candidate positions; reservoir-sampling 4 returns 4 records.
    assert n == 4
    records = _read_legal_records(run_dir)
    assert {r.position_id.split(":")[0] for r in records} <= {"game-0000", "game-0001"}
    # Position IDs must encode the source ply so the sampled positions can be
    # reconstructed from the records.
    assert all(":ply" in r.position_id for r in records)
    # Every recorded FEN must be a valid chess position.
    for r in records:
        chess.Board(r.fen)


def test_sample_from_games_caps_at_total_positions(tmp_path: Path) -> None:
    """``sample_n`` larger than the candidate pool should yield every candidate."""
    pgn_file = tmp_path / "games.pgn"
    pgn_file.write_text(_TWO_GAME_PGN, encoding="utf-8")
    run_dir = tmp_path / "run"

    config = LegalRateConfig(sample_from_games_path=pgn_file, sample_n=1000, seed=0)
    engine = _ConstantEngine()

    n = run_legal_rate(
        engine=engine,
        config=config,
        run_dir=run_dir,
        run_id=DEFAULT_RUN_ID,
        checkpoint=DEFAULT_CHECKPOINT,
    )

    # 4 plies per game * 2 games = 8 candidate positions.
    assert n == 8


def test_sample_from_games_is_seeded(tmp_path: Path) -> None:
    """Same seed → same sampled position IDs across runs."""
    pgn_file = tmp_path / "games.pgn"
    pgn_file.write_text(_TWO_GAME_PGN, encoding="utf-8")

    def run(seed: int, run_dir: Path) -> list[str]:
        config = LegalRateConfig(sample_from_games_path=pgn_file, sample_n=3, seed=seed)
        run_legal_rate(
            engine=_ConstantEngine(),
            config=config,
            run_dir=run_dir,
            run_id=DEFAULT_RUN_ID,
            checkpoint=DEFAULT_CHECKPOINT,
        )
        return [r.position_id for r in _read_legal_records(run_dir)]

    a = run(seed=42, run_dir=tmp_path / "run-a")
    b = run(seed=42, run_dir=tmp_path / "run-b")
    assert a == b


def test_sample_from_games_handles_empty_pgn(tmp_path: Path) -> None:
    """An empty PGN file emits zero records without raising."""
    pgn_file = tmp_path / "empty.pgn"
    pgn_file.write_text("", encoding="utf-8")
    run_dir = tmp_path / "run"

    n = run_legal_rate(
        engine=_ConstantEngine(),
        config=LegalRateConfig(sample_from_games_path=pgn_file, sample_n=10, seed=0),
        run_dir=run_dir,
        run_id=DEFAULT_RUN_ID,
        checkpoint=DEFAULT_CHECKPOINT,
    )

    assert n == 0
    assert _read_legal_records(run_dir) == []
    # Run dir still has a valid config.json (we wrote it before scoring).
    assert (run_dir / CONFIG_FILENAME).exists()


def test_config_rejects_no_source() -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        LegalRateConfig()


def test_config_rejects_both_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        LegalRateConfig(positions_path=tmp_path / "p.fen", sample_from_games_path=tmp_path / "g.pgn")


def test_config_rejects_zero_sample_n(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sample_n"):
        LegalRateConfig(sample_from_games_path=tmp_path / "g.pgn", sample_n=0)


def test_positions_file_malformed_fen_raises_with_lineno(tmp_path: Path) -> None:
    """A malformed FEN line must surface the file path and line number, not just python-chess's bare error."""
    fen_file = tmp_path / "positions.fen"
    fen_file.write_text(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\nnot-a-fen\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"positions\.fen:2: invalid FEN"):
        run_legal_rate(
            engine=_ConstantEngine(),
            config=LegalRateConfig(positions_path=fen_file),
            run_dir=tmp_path / "run",
            run_id=DEFAULT_RUN_ID,
            checkpoint=DEFAULT_CHECKPOINT,
        )
