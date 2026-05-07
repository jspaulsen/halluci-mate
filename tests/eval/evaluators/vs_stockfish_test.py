"""Tests for the ``vs_stockfish`` evaluator.

The evaluator is exercised end-to-end against stub engine and Stockfish
implementations: stubs avoid loading a real checkpoint or shelling out to
the Stockfish binary, while still driving the full record-emission +
run-directory write paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import chess
import chess.engine
import pytest

from halluci_mate.eval.evaluators.vs_stockfish import (
    GameOutcome,
    HalluciColor,
    VsStockfishConfig,
    _color_for_game,
    _phase_for_ply,
    run_vs_stockfish,
)
from halluci_mate.eval.records import Evaluator, PerGameRecord, PerMoveRecord, Phase, Side, TopKEntry
from halluci_mate.eval.runs import CONFIG_FILENAME, GAMES_PGN_FILENAME, RECORDS_FILENAME, RunReader
from halluci_mate.inference import MovePrediction
from tests.eval.conftest import DEFAULT_CHECKPOINT, DEFAULT_RUN_ID

if TYPE_CHECKING:
    from pathlib import Path

    from halluci_mate.game import Game


class _StubEngine:
    """Plays the first legal move; mirrors ``ChessInferenceEngine.predict_with_metadata``."""

    def __init__(self, *, mask_used: bool = True) -> None:
        self._mask_used = mask_used

    def predict_with_metadata(
        self,
        game: Game,
        *,
        constrained: bool | None = None,
        record_top_k: int = 5,
    ) -> MovePrediction:
        del constrained, record_top_k
        legal_moves = list(game.board.legal_moves)
        played = legal_moves[0]
        return MovePrediction(
            played_move=played,
            model_move_uci=played.uci(),
            raw_sample_move_uci=played.uci(),
            raw_sample_legal=True,
            model_top_k=[TopKEntry(move=played.uci(), logprob=-0.1)],
            mask_used=self._mask_used,
        )


class _StubStockfish:
    """Plays the last legal move (so Stockfish picks something other than the model's first-move stub)."""

    def __init__(self) -> None:
        self.configured: dict[str, Any] = {}

    def configure(self, options: dict[str, Any]) -> None:
        self.configured = dict(options)

    def play(self, board: chess.Board, limit: chess.engine.Limit) -> chess.engine.PlayResult:
        del limit
        move = list(board.legal_moves)[-1]
        return chess.engine.PlayResult(move=move, ponder=None)


def _read_records(run_dir: Path) -> list[PerMoveRecord]:
    return [r for r in RunReader(run_dir).read_records() if isinstance(r, PerMoveRecord)]


def _read_game_records(run_dir: Path) -> list[PerGameRecord]:
    return [r for r in RunReader(run_dir).read_records() if isinstance(r, PerGameRecord)]


def test_runs_one_game_end_to_end(tmp_path: Path) -> None:
    """A 1-game run produces a valid HAL-5 directory with well-formed records."""
    run_dir = tmp_path / "run"
    config = VsStockfishConfig(games=1, max_plies=4, halluci_color="white")

    outcomes = run_vs_stockfish(
        engine=_StubEngine(),
        stockfish=_StubStockfish(),
        config=config,
        run_dir=run_dir,
        run_id=DEFAULT_RUN_ID,
        checkpoint=DEFAULT_CHECKPOINT,
    )

    assert (run_dir / CONFIG_FILENAME).exists()
    assert (run_dir / RECORDS_FILENAME).exists()
    assert (run_dir / GAMES_PGN_FILENAME).exists()

    config_payload = RunReader(run_dir).read_config()
    assert config_payload["evaluator"] == Evaluator.VS_STOCKFISH.value
    assert config_payload["run_id"] == DEFAULT_RUN_ID
    assert config_payload["checkpoint"] == DEFAULT_CHECKPOINT
    assert config_payload["games"] == 1

    records = _read_records(run_dir)
    assert len(records) >= 1, "max_plies=4 with halluci=white must yield at least one model record"
    assert [r.event_id for r in records] == list(range(len(records)))
    assert all(r.run_id == DEFAULT_RUN_ID for r in records)
    assert all(r.evaluator == Evaluator.VS_STOCKFISH for r in records)
    assert all(r.game_id == "game-0000" for r in records)
    assert all(r.model_side == Side.WHITE for r in records)
    assert all(r.mask_used is True for r in records)
    assert all(r.model_top_k for r in records)
    assert all(r.sf_best_move is None for r in records)
    assert records[0].prior_opponent_move is None
    if len(records) > 1:
        assert records[1].prior_opponent_move is not None

    assert (run_dir / GAMES_PGN_FILENAME).read_text(encoding="utf-8").strip() != ""
    assert len(outcomes) == 1
    assert outcomes[0].game_id == "game-0000"


def test_alternates_color_across_games(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config = VsStockfishConfig(games=2, max_plies=2, halluci_color="alternate")

    outcomes = run_vs_stockfish(
        engine=_StubEngine(),
        stockfish=_StubStockfish(),
        config=config,
        run_dir=run_dir,
        run_id=DEFAULT_RUN_ID,
        checkpoint=DEFAULT_CHECKPOINT,
    )

    assert [o.halluci_color for o in outcomes] == [chess.WHITE, chess.BLACK]
    records = _read_records(run_dir)
    assert {r.game_id for r in records} == {"game-0000", "game-0001"}
    sides_by_game = {r.game_id: r.model_side for r in records}
    assert sides_by_game["game-0000"] == Side.WHITE
    assert sides_by_game["game-0001"] == Side.BLACK


def test_unconstrained_records_mask_used_false(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config = VsStockfishConfig(games=1, max_plies=2, halluci_color="white", unconstrained=True)

    run_vs_stockfish(
        engine=_StubEngine(mask_used=False),
        stockfish=_StubStockfish(),
        config=config,
        run_dir=run_dir,
        run_id=DEFAULT_RUN_ID,
        checkpoint=DEFAULT_CHECKPOINT,
    )

    records = _read_records(run_dir)
    assert records, "expected at least one record"
    assert all(r.mask_used is False for r in records)


def test_stockfish_skill_is_configured(tmp_path: Path) -> None:
    """The configured skill flows through to Stockfish before any games begin."""
    run_dir = tmp_path / "run"
    config = VsStockfishConfig(games=1, max_plies=2, halluci_color="white", stockfish_skill=7)
    stockfish = _StubStockfish()

    run_vs_stockfish(
        engine=_StubEngine(),
        stockfish=stockfish,
        config=config,
        run_dir=run_dir,
        run_id=DEFAULT_RUN_ID,
        checkpoint=DEFAULT_CHECKPOINT,
    )

    assert stockfish.configured == {"Skill Level": 7}


def test_terminates_on_max_plies(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config = VsStockfishConfig(games=1, max_plies=3, halluci_color="white")

    outcomes = run_vs_stockfish(
        engine=_StubEngine(),
        stockfish=_StubStockfish(),
        config=config,
        run_dir=run_dir,
        run_id=DEFAULT_RUN_ID,
        checkpoint=DEFAULT_CHECKPOINT,
    )

    assert outcomes[0].termination == "max-plies"
    assert outcomes[0].result == "*"


def test_emits_per_game_record(tmp_path: Path) -> None:
    """Each game writes a terminal `PerGameRecord` summarizing its outcome."""
    run_dir = tmp_path / "run"
    config = VsStockfishConfig(games=2, max_plies=2, halluci_color="alternate")

    run_vs_stockfish(
        engine=_StubEngine(),
        stockfish=_StubStockfish(),
        config=config,
        run_dir=run_dir,
        run_id=DEFAULT_RUN_ID,
        checkpoint=DEFAULT_CHECKPOINT,
    )

    game_records = _read_game_records(run_dir)
    assert [g.game_id for g in game_records] == ["game-0000", "game-0001"]
    assert [g.model_side for g in game_records] == [Side.WHITE, Side.BLACK]
    assert all(g.termination == "max-plies" for g in game_records)
    assert all(g.result == "*" for g in game_records)


def test_terminates_on_illegal_move(tmp_path: Path) -> None:
    """When the played move is None, the record is still written and the game ends."""
    run_dir = tmp_path / "run"
    config = VsStockfishConfig(games=1, max_plies=4, halluci_color="white", unconstrained=True)

    class _IllegalEngine:
        def predict_with_metadata(self, game: Game, *, constrained: bool | None = None, record_top_k: int = 5) -> MovePrediction:
            del game, constrained, record_top_k
            return MovePrediction(
                played_move=None,
                model_move_uci="<UNK>",
                raw_sample_move_uci="<UNK>",
                raw_sample_legal=False,
                model_top_k=[],
                mask_used=False,
            )

    outcomes = run_vs_stockfish(
        engine=_IllegalEngine(),
        stockfish=_StubStockfish(),
        config=config,
        run_dir=run_dir,
        run_id=DEFAULT_RUN_ID,
        checkpoint=DEFAULT_CHECKPOINT,
    )

    assert outcomes[0].termination == "illegal-move"
    records = _read_records(run_dir)
    assert len(records) == 1
    assert records[0].raw_sample_legal is False
    assert records[0].model_move == "<UNK>"


@pytest.mark.parametrize(
    ("ply", "expected"),
    [
        (0, Phase.OPENING),
        (19, Phase.OPENING),
        (20, Phase.MIDDLE),
        (59, Phase.MIDDLE),
        (60, Phase.ENDGAME),
        (200, Phase.ENDGAME),
    ],
)
def test_phase_classification_at_boundaries(ply: int, expected: Phase) -> None:
    assert _phase_for_ply(ply) == expected


@pytest.mark.parametrize(
    ("mode", "idx", "expected"),
    [
        ("white", 0, chess.WHITE),
        ("white", 1, chess.WHITE),
        ("black", 0, chess.BLACK),
        ("black", 1, chess.BLACK),
        ("alternate", 0, chess.WHITE),
        ("alternate", 1, chess.BLACK),
        ("alternate", 2, chess.WHITE),
    ],
)
def test_color_for_game(mode: HalluciColor, idx: int, expected: chess.Color) -> None:
    assert _color_for_game(mode, idx) == expected


def test_stockfish_skill_validation_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="stockfish_skill"):
        VsStockfishConfig(stockfish_skill=21)


def test_games_validation_rejects_zero() -> None:
    with pytest.raises(ValueError, match="games"):
        VsStockfishConfig(games=0)


def test_max_plies_validation_rejects_zero() -> None:
    with pytest.raises(ValueError, match="max_plies"):
        VsStockfishConfig(max_plies=0)


def test_record_top_k_validation_rejects_negative() -> None:
    with pytest.raises(ValueError, match="record_top_k"):
        VsStockfishConfig(record_top_k=-1)


def test_movetime_overrides_depth() -> None:
    config = VsStockfishConfig(stockfish_movetime=0.05, stockfish_depth=None)
    limit = config.stockfish_limit()
    assert limit.time == pytest.approx(0.05)
    assert limit.depth is None


def test_depth_used_when_no_movetime() -> None:
    config = VsStockfishConfig(stockfish_movetime=None, stockfish_depth=4)
    limit = config.stockfish_limit()
    assert limit.depth == 4
    assert limit.time is None


def test_outcome_dataclass_shape() -> None:
    """Smoke-check ``GameOutcome`` is constructable from positional args."""
    outcome = GameOutcome(
        game_id="game-0000",
        halluci_color=chess.WHITE,
        result="1-0",
        ply_count=5,
        termination="natural",
        pgn='[Event "x"]\n\n1. e4 *\n',
    )
    assert outcome.game_id == "game-0000"
