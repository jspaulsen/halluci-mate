"""``vs_stockfish`` evaluator: play N games against Stockfish, emit per-move records.

Built on the HAL-5 record/runs APIs. Caller manages the inference engine and
Stockfish process — keeps the evaluator pure (testable with stubs).

Opt-in per-position Stockfish analysis (``--sf-analyze``) populates the
``sf_*`` / ``centipawn_loss`` / ``is_blunder`` fields; metric aggregation
lives in ``halluci_mate.eval.metrics``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import chess
import chess.engine
import chess.pgn

from halluci_mate.eval.records import Evaluator, GameResult, PerGameRecord, PerMoveRecord, Phase, Side, Termination
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

# Centipawn value used to clamp mate scores when ``--sf-analyze`` is on, so
# `sf_eval_*_cp` stays a finite int. Missed forced mates still produce very
# large `centipawn_loss` values but the on-disk schema remains diff-friendly.
_MATE_SCORE_CP = 100_000

HalluciColor = Literal["white", "black", "alternate"]


class _StockfishEngine(Protocol):
    """The methods of ``chess.engine.SimpleEngine`` that the evaluator uses.

    Lifecycle (``popen_uci`` / ``quit``) is the caller's responsibility; this
    protocol covers only the calls made *during* a run, which is also what
    test stubs need to implement. ``analyse`` is only invoked when
    ``VsStockfishConfig.analyze`` is true.
    """

    def configure(self, options: dict[str, Any]) -> None: ...

    def play(self, board: chess.Board, limit: chess.engine.Limit) -> chess.engine.PlayResult: ...

    def analyse(self, board: chess.Board, limit: chess.engine.Limit) -> chess.engine.InfoDict: ...


@dataclass(frozen=True)
class VsStockfishConfig:
    """Knobs for ``run_vs_stockfish``."""

    games: int = 2
    halluci_color: HalluciColor = "alternate"
    stockfish_skill: int = 0
    stockfish_depth: int | None = 1
    stockfish_movetime: float | None = None
    max_plies: int = 400
    unconstrained: bool = False
    record_top_k: int = 5
    analyze: bool = False
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
    result: GameResult
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
    extra_config: dict[str, object] | None = None,
) -> list[GameOutcome]:
    """Play ``config.games`` games and write a HAL-5 run directory under ``run_dir``.

    Caller manages the lifecycle of ``engine`` and ``stockfish``. Stockfish's
    ``Skill Level`` is configured here before any games begin.

    ``extra_config`` is merged into the on-disk ``config.json`` payload — the
    CLI uses this to record the engine's effective sampling parameters
    (``temperature`` / ``top_k``) without duplicating them on
    ``VsStockfishConfig``, where the evaluator would never read them.
    """
    stockfish.configure({"Skill Level": config.stockfish_skill})
    limit = config.stockfish_limit()

    writer = RunWriter(run_dir)
    writer.write_config(_build_config_payload(config=config, run_id=run_id, checkpoint=checkpoint, extra=extra_config))

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
            # `_play_one_game` returns the next-free event_id (one past the
            # last per-move record for this game); the per-game record
            # consumes that slot, then we bump for the next game's first move.
            writer.append_record(
                PerGameRecord(
                    run_id=run_id,
                    event_id=event_id,
                    evaluator=Evaluator.VS_STOCKFISH,
                    checkpoint=checkpoint,
                    game_id=outcome.game_id,
                    model_side=_side_for_color(color),
                    result=outcome.result,
                    termination=outcome.termination,
                    ply_count=outcome.ply_count,
                )
            )
            event_id += 1

    # `write_pgn` writes a whole file; it does not use the records-jsonl
    # handle held by the `with writer:` block above.
    writer.write_pgn("\n\n".join(o.pgn for o in outcomes) + "\n")
    return outcomes


def _build_config_payload(*, config: VsStockfishConfig, run_id: str, checkpoint: str, extra: dict[str, object] | None) -> dict[str, object]:
    payload: dict[str, object] = {
        "evaluator": Evaluator.VS_STOCKFISH.value,
        "run_id": run_id,
        "checkpoint": checkpoint,
    }
    payload.update(asdict(config))
    if extra:
        payload.update(extra)
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
                stockfish=stockfish,
                stockfish_limit=stockfish_limit,
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
            # `_finalize_pgn` writes one of the four PGN result strings — Pydantic
            # re-validates this when it lands on `PerGameRecord.result`.
            result=cast("GameResult", pgn.headers["Result"]),
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
    stockfish: _StockfishEngine,
    stockfish_limit: chess.engine.Limit,
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
    mover_color = board.turn
    fen_before = board.fen()
    legal_uci_before = [move.uci() for move in board.legal_moves]

    before_info = stockfish.analyse(board, stockfish_limit) if config.analyze else None
    sf_best_move = _pv_first_move_uci(before_info) if before_info is not None else None
    eval_before_white = _white_relative_cp(before_info) if before_info is not None else None

    prediction = engine.predict_with_metadata(
        state.game,
        constrained=not config.unconstrained,
        record_top_k=config.record_top_k,
    )

    eval_after_white: int | None = None
    centipawn_loss: int | None = None
    is_blunder: bool | None = None
    if eval_before_white is not None and prediction.played_move is not None:
        # Analyse the resulting position on a board copy so move application
        # stays at the bottom of the function alongside the non-analyze path.
        after_board = board.copy(stack=False)
        after_board.push(prediction.played_move)
        eval_after_white = _white_relative_cp(stockfish.analyse(after_board, stockfish_limit))
        centipawn_loss = _centipawn_loss_stm(eval_before_white, eval_after_white, mover_color)
        is_blunder = centipawn_loss > config.blunder_threshold_cp

    record = PerMoveRecord(
        run_id=run_id,
        event_id=event_id,
        evaluator=Evaluator.VS_STOCKFISH,
        checkpoint=checkpoint,
        game_id=game_id,
        ply=ply,
        phase=_phase_for_ply(ply),
        side_to_move=_side_for_color(mover_color),
        model_side=model_side,
        fen_before=fen_before,
        legal_moves=legal_uci_before,
        model_move=prediction.model_move_uci,
        model_top_k=list(prediction.model_top_k),
        mask_used=prediction.mask_used,
        raw_sample_move=prediction.raw_sample_move_uci,
        raw_sample_legal=prediction.raw_sample_legal,
        prior_opponent_move=state.prior_opponent_move,
        sf_best_move=sf_best_move,
        sf_eval_before_cp=eval_before_white,
        sf_eval_after_cp=eval_after_white,
        centipawn_loss=centipawn_loss,
        is_blunder=is_blunder,
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


def _white_relative_cp(info: chess.engine.InfoDict) -> int:
    """Return the white-relative centipawn evaluation, with mates clamped.

    Mate scores serialize as `int | None` in the record schema, so they have
    to collapse to a finite int here. See `_MATE_SCORE_CP`.
    """
    score = info["score"].white().score(mate_score=_MATE_SCORE_CP)
    # `score(mate_score=...)` only returns `None` for `MateGiven`, which
    # Stockfish never emits on a non-terminal position. Be defensive.
    assert score is not None
    return score


def _pv_first_move_uci(info: chess.engine.InfoDict) -> str | None:
    """Return the first move of the principal variation as UCI, or `None` if absent."""
    pv = info.get("pv")
    if not pv:
        return None
    return pv[0].uci()


def _centipawn_loss_stm(eval_before_white: int, eval_after_white: int, mover_color: chess.Color) -> int:
    """Compute side-to-move-relative centipawn loss.

    `sf_eval_*_cp` are stored white-relative on disk; CPL is computed from
    the mover's perspective per `docs/eval_harness.md` §Per-move record.
    """
    sign = 1 if mover_color == chess.WHITE else -1
    before_stm = sign * eval_before_white
    after_stm = sign * eval_after_white
    return max(0, before_stm - after_stm)
