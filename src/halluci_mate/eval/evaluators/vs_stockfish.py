"""``vs_stockfish`` evaluator: play N games against Stockfish, emit per-move records.

Built on the HAL-5 record/runs APIs. Caller manages the inference engine and
Stockfish process — keeps the evaluator pure (testable with stubs).

Out of scope (separate tickets): Stockfish per-position analysis
(``--sf-analyze``) and metric aggregation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import chess
import chess.engine
import chess.pgn

from halluci_mate.eval.records import Evaluator, PerMoveRecord, Phase, Side
from halluci_mate.eval.runs import RunWriter
from halluci_mate.game import Game, Perspective

if TYPE_CHECKING:
    from pathlib import Path

    from halluci_mate.inference import Predictor

# Stockfish "Skill Level" UCI option range.
STOCKFISH_SKILL_MIN = 0
STOCKFISH_SKILL_MAX = 20

# Phase boundaries from docs/eval_harness.md §Per-move record.
_OPENING_PLY_MAX = 20
_MIDDLEGAME_PLY_MAX = 60

HalluciColor = Literal["white", "black", "alternate"]
Termination = Literal["natural", "max-plies", "stockfish-resigned", "illegal-move"]


class _StockfishEngine(Protocol):
    """The two methods of ``chess.engine.SimpleEngine`` that the evaluator uses.

    Lifecycle (``popen_uci`` / ``quit``) is the caller's responsibility; this
    protocol covers only the calls made *during* a run, which is also what
    test stubs need to implement.
    """

    def configure(self, options: dict[str, Any]) -> None: ...

    def play(self, board: chess.Board, limit: chess.engine.Limit) -> chess.engine.PlayResult: ...


@dataclass(frozen=True)
class VsStockfishConfig:
    """Knobs for ``run_vs_stockfish``."""

    games: int = 2
    halluci_color: HalluciColor = "alternate"
    stockfish_skill: int = 0
    stockfish_depth: int | None = 1
    stockfish_movetime: float | None = None
    max_plies: int = 400
    temperature: float = 0.0
    top_k: int = 0
    unconstrained: bool = False
    record_top_k: int = 5
    blunder_threshold_cp: int = 200

    def __post_init__(self) -> None:
        if self.games < 1:
            raise ValueError(f"games must be >= 1; got {self.games}")
        if not (STOCKFISH_SKILL_MIN <= self.stockfish_skill <= STOCKFISH_SKILL_MAX):
            raise ValueError(f"stockfish_skill must be in [{STOCKFISH_SKILL_MIN}, {STOCKFISH_SKILL_MAX}]; got {self.stockfish_skill}")
        if self.max_plies < 1:
            raise ValueError(f"max_plies must be >= 1; got {self.max_plies}")
        if self.record_top_k < 0:
            raise ValueError(f"record_top_k must be >= 0; got {self.record_top_k}")
        if self.stockfish_movetime is None and (self.stockfish_depth is None or self.stockfish_depth < 1):
            raise ValueError("must set stockfish_depth >= 1 or stockfish_movetime")

    def stockfish_limit(self) -> chess.engine.Limit:
        if self.stockfish_movetime is not None:
            return chess.engine.Limit(time=self.stockfish_movetime)
        # __post_init__ guarantees depth is set when movetime is None.
        assert self.stockfish_depth is not None
        return chess.engine.Limit(depth=self.stockfish_depth)


@dataclass(frozen=True)
class GameOutcome:
    """Summary of one completed (or aborted) game."""

    game_id: str
    halluci_color: chess.Color
    result: str
    ply_count: int
    termination: Termination
    pgn: str


@dataclass
class _GameState:
    """Mutable per-game scratch space."""

    game: Game
    pgn_root: chess.pgn.Game
    pgn_node: chess.pgn.GameNode
    prior_opponent_move: str | None = None
    termination: Termination = "natural"


def run_vs_stockfish(
    *,
    engine: Predictor,
    stockfish: _StockfishEngine,
    config: VsStockfishConfig,
    run_dir: Path,
    run_id: str,
    checkpoint: str,
) -> list[GameOutcome]:
    """Play ``config.games`` games and write a HAL-5 run directory under ``run_dir``.

    Caller manages the lifecycle of ``engine`` and ``stockfish``. Stockfish's
    ``Skill Level`` is configured here before any games begin.
    """
    stockfish.configure({"Skill Level": config.stockfish_skill})
    limit = config.stockfish_limit()

    writer = RunWriter(run_dir)
    writer.write_config(_build_config_payload(config=config, run_id=run_id, checkpoint=checkpoint))

    outcomes: list[GameOutcome] = []
    event_id = 0
    with writer:
        for game_index in range(config.games):
            color = _color_for_game(config.halluci_color, game_index)
            outcome, event_id = _play_one_game(
                engine=engine,
                stockfish=stockfish,
                stockfish_limit=limit,
                config=config,
                writer=writer,
                run_id=run_id,
                checkpoint=checkpoint,
                game_index=game_index,
                halluci_color=color,
                event_id_start=event_id,
            )
            outcomes.append(outcome)

    # `write_pgn` writes a whole file; it does not use the records-jsonl
    # handle held by the `with writer:` block above.
    writer.write_pgn("\n\n".join(o.pgn for o in outcomes) + "\n")
    return outcomes


def _build_config_payload(*, config: VsStockfishConfig, run_id: str, checkpoint: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "evaluator": Evaluator.VS_STOCKFISH.value,
        "run_id": run_id,
        "checkpoint": checkpoint,
    }
    payload.update(asdict(config))
    return payload


def _play_one_game(
    *,
    engine: Predictor,
    stockfish: _StockfishEngine,
    stockfish_limit: chess.engine.Limit,
    config: VsStockfishConfig,
    writer: RunWriter,
    run_id: str,
    checkpoint: str,
    game_index: int,
    halluci_color: chess.Color,
    event_id_start: int,
) -> tuple[GameOutcome, int]:
    state = _new_game_state(halluci_color=halluci_color)
    game_id = f"game-{game_index:04d}"
    model_side = _side_for_color(halluci_color)
    event_id = event_id_start

    while not state.game.board.is_game_over(claim_draw=True):
        if state.game.board.ply() >= config.max_plies:
            state.termination = "max-plies"
            break

        if state.game.board.turn == halluci_color:
            _handle_model_turn(
                state=state,
                engine=engine,
                config=config,
                writer=writer,
                run_id=run_id,
                checkpoint=checkpoint,
                game_id=game_id,
                model_side=model_side,
                event_id=event_id,
            )
            event_id += 1
            if state.termination == "illegal-move":
                break
        else:
            stockfish_move = stockfish.play(state.game.board, stockfish_limit).move
            if stockfish_move is None:
                state.termination = "stockfish-resigned"
                break
            _apply_move(state, stockfish_move)
            state.prior_opponent_move = stockfish_move.uci()

    pgn = _finalize_pgn(state=state, halluci_color=halluci_color)

    state.game.reset_cache()

    return (
        GameOutcome(
            game_id=game_id,
            halluci_color=halluci_color,
            result=pgn.headers["Result"],
            ply_count=state.game.board.ply(),
            termination=state.termination,
            pgn=str(pgn),
        ),
        event_id,
    )


def _new_game_state(*, halluci_color: chess.Color) -> _GameState:
    halluci_is_white = halluci_color == chess.WHITE
    game = Game(board=chess.Board(), perspective=Perspective.WHITE if halluci_is_white else Perspective.BLACK)

    pgn_root = chess.pgn.Game()
    pgn_root.headers["White"], pgn_root.headers["Black"] = ("halluci-mate", "Stockfish") if halluci_is_white else ("Stockfish", "halluci-mate")

    return _GameState(game=game, pgn_root=pgn_root, pgn_node=pgn_root)


def _handle_model_turn(
    *,
    state: _GameState,
    engine: Predictor,
    config: VsStockfishConfig,
    writer: RunWriter,
    run_id: str,
    checkpoint: str,
    game_id: str,
    model_side: Side,
    event_id: int,
) -> None:
    board = state.game.board
    ply = board.ply()

    prediction = engine.predict_with_metadata(
        state.game,
        constrained=not config.unconstrained,
        record_top_k=config.record_top_k,
    )

    record = PerMoveRecord(
        run_id=run_id,
        event_id=event_id,
        evaluator=Evaluator.VS_STOCKFISH,
        checkpoint=checkpoint,
        game_id=game_id,
        ply=ply,
        phase=_phase_for_ply(ply),
        side_to_move=_side_for_color(board.turn),
        model_side=model_side,
        fen_before=board.fen(),
        legal_moves=[move.uci() for move in board.legal_moves],
        model_move=prediction.model_move_uci,
        model_top_k=list(prediction.model_top_k),
        mask_used=prediction.mask_used,
        raw_sample_move=prediction.raw_sample_move_uci,
        raw_sample_legal=prediction.raw_sample_legal,
        prior_opponent_move=state.prior_opponent_move,
        sf_best_move=None,
        sf_eval_before_cp=None,
        sf_eval_after_cp=None,
        centipawn_loss=None,
        is_blunder=None,
    )
    writer.append_record(record)

    if prediction.played_move is None:
        state.termination = "illegal-move"
        return

    _apply_move(state, prediction.played_move)


def _apply_move(state: _GameState, move: chess.Move) -> None:
    state.game.push_move(move)
    state.pgn_node = state.pgn_node.add_variation(move)


def _finalize_pgn(*, state: _GameState, halluci_color: chess.Color) -> chess.pgn.Game:
    result = state.game.board.result(claim_draw=True) if state.termination == "natural" else "*"
    state.pgn_root.headers["Result"] = result
    state.pgn_root.headers["Termination"] = state.termination
    state.pgn_root.headers["HalluciColor"] = "white" if halluci_color == chess.WHITE else "black"
    return state.pgn_root


def _color_for_game(mode: HalluciColor, game_index: int) -> chess.Color:
    if mode == "white":
        return chess.WHITE
    if mode == "black":
        return chess.BLACK
    return chess.WHITE if game_index % 2 == 0 else chess.BLACK


def _phase_for_ply(ply: int) -> Phase:
    if ply < _OPENING_PLY_MAX:
        return Phase.OPENING
    if ply < _MIDDLEGAME_PLY_MAX:
        return Phase.MIDDLE
    return Phase.ENDGAME


def _side_for_color(color: chess.Color) -> Side:
    return Side.WHITE if color == chess.WHITE else Side.BLACK
