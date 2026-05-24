"""Full-strength Stockfish self-play with a MultiPV diversity 'wobble'.

Seeds from a human opening prefix, then plays both sides with Stockfish,
sampling among near-equal moves so a fixed-budget (otherwise deterministic)
engine produces a diverse corpus. Games are adjudicated (resign / draw / cap)
to stay finite and avoid dead-drawn shuffles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import chess
import chess.engine

if TYPE_CHECKING:
    import random

# Mate scores collapse to a finite cp so adjudication/selection arithmetic
# never sees ``None`` (mirrors vs_stockfish ``_MATE_SCORE_CP``).
_MATE_SCORE_CP = 100_000

Outcome = Literal["white", "black", "draw"]
Termination = Literal["natural", "adjudicated-win", "adjudicated-draw", "max-plies"]


@dataclass
class _AdjudicationState:
    """Rolling consecutive-ply counters for resign/draw adjudication."""

    decisive_sign: int = 0
    decisive_run: int = 0
    draw_run: int = 0

    def update(self, white_cp: int, ply: int, config: SelfPlayConfig) -> tuple[Termination, Outcome] | None:
        """Feed the best-line white-relative eval; return a verdict or ``None``."""
        sign = 1 if white_cp >= config.resign_cp else -1 if white_cp <= -config.resign_cp else 0
        if sign != 0 and sign == self.decisive_sign:
            self.decisive_run += 1
        else:
            self.decisive_sign = sign
            self.decisive_run = 1 if sign != 0 else 0
        if self.decisive_run >= config.resign_plies:
            return ("adjudicated-win", "white" if sign > 0 else "black")

        if ply >= config.draw_min_ply and abs(white_cp) < config.draw_cp:
            self.draw_run += 1
        else:
            self.draw_run = 0
        if self.draw_run >= config.draw_plies:
            return ("adjudicated-draw", "draw")
        return None


@dataclass(frozen=True)
class SelfPlayConfig:
    """Knobs for one self-play generation run. ``depth`` is the cost dial."""

    depth: int = 18
    multipv: int = 4
    wobble_cp: int = 30
    wobble_temp: float = 50.0
    seed_plies: int = 12
    resign_cp: int = 700
    resign_plies: int = 4
    draw_cp: int = 15
    draw_plies: int = 8
    draw_min_ply: int = 60
    max_plies: int = 200

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1; got {self.depth}")
        if self.multipv < 1:
            raise ValueError(f"multipv must be >= 1; got {self.multipv}")
        if self.wobble_cp < 0:
            raise ValueError(f"wobble_cp must be >= 0; got {self.wobble_cp}")
        if self.wobble_temp <= 0:
            raise ValueError(f"wobble_temp must be > 0; got {self.wobble_temp}")
        if self.seed_plies < 0:
            raise ValueError(f"seed_plies must be >= 0; got {self.seed_plies}")
        if self.max_plies < 1:
            raise ValueError(f"max_plies must be >= 1; got {self.max_plies}")


def _stm_cp(info: chess.engine.InfoDict, turn: chess.Color) -> int:
    """Side-to-move-relative centipawns for one analysis line (mates clamped)."""
    score = info["score"].pov(turn).score(mate_score=_MATE_SCORE_CP)
    assert score is not None  # unreachable: score(mate_score=...) never returns None when mate_score is set
    return score


def _best_white_cp(infos: list[chess.engine.InfoDict]) -> int:
    """White-relative centipawns of the best line (``infos[0]``), mates clamped."""
    score = infos[0]["score"].white().score(mate_score=_MATE_SCORE_CP)
    assert score is not None  # unreachable: score(mate_score=...) never returns None when mate_score is set
    return score


def _pv_first_move(info: chess.engine.InfoDict) -> chess.Move | None:
    """First move of an analysis line's principal variation, or ``None``."""
    pv = info.get("pv")
    if not pv:
        return None
    return pv[0]


def _softmax(values: list[float]) -> list[float]:
    """Numerically-stable softmax over ``values``."""
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def select_wobble_move(
    infos: list[chess.engine.InfoDict],
    turn: chess.Color,
    config: SelfPlayConfig,
    rng: random.Random,
) -> chess.Move:
    """Sample a near-best move from MultiPV analysis (the diversity 'wobble').

    Keeps lines within ``config.wobble_cp`` of the best side-to-move score,
    then softmax-samples by ``cp / config.wobble_temp`` (lower temp favors the
    best move; larger temp approaches uniform). ``rng`` makes this reproducible.
    """
    candidates = [(move, _stm_cp(info, turn)) for info in infos if (move := _pv_first_move(info)) is not None]
    if not candidates:
        raise ValueError("analysis returned no candidate moves")
    best_cp = max(cp for _, cp in candidates)
    in_band = [(move, cp) for move, cp in candidates if best_cp - cp <= config.wobble_cp]
    weights = _softmax([cp / config.wobble_temp for _, cp in in_band])
    return rng.choices([move for move, _ in in_band], weights=weights, k=1)[0]


# White-relative result strings -> outcome label; "*" (no result) -> draw.
_RESULT_TO_OUTCOME: dict[str, Outcome] = {"1-0": "white", "0-1": "black", "1/2-1/2": "draw"}


class _AnalysisEngine(Protocol):
    """The single ``chess.engine`` call ``play_seed`` makes (multipv analyse)."""

    def analyse(self, board: chess.Board, limit: chess.engine.Limit, *, multipv: int) -> list[chess.engine.InfoDict]: ...


@dataclass(frozen=True)
class SelfPlayGame:
    """One generated game: full move list (incl. seed prefix) + outcome."""

    seed_source: str
    seed_plies: int
    moves_uci: list[str]
    outcome: Outcome
    termination: Termination


# A SeedOpening is (seed_source, seed_moves_uci) -- see distill.seeds.
SeedOpening = tuple[str, list[str]]


def play_seed(engine: _AnalysisEngine, seed: SeedOpening, config: SelfPlayConfig, rng: random.Random) -> SelfPlayGame:
    """Play one full-strength game from a human opening seed."""
    seed_source, seed_moves = seed
    prefix = seed_moves[: config.seed_plies]
    board = chess.Board()
    for uci in prefix:
        board.push(chess.Move.from_uci(uci))
    moves = list(prefix)

    adjudicator = _AdjudicationState()
    limit = chess.engine.Limit(depth=config.depth)
    termination, outcome = _play_loop(engine, board, moves, adjudicator, config, rng, limit)
    return SelfPlayGame(seed_source=seed_source, seed_plies=len(prefix), moves_uci=moves, outcome=outcome, termination=termination)


def _play_loop(
    engine: _AnalysisEngine,
    board: chess.Board,
    moves: list[str],
    adjudicator: _AdjudicationState,
    config: SelfPlayConfig,
    rng: random.Random,
    limit: chess.engine.Limit,
) -> tuple[Termination, Outcome]:
    """Mutate ``board``/``moves`` until termination; return (termination, outcome)."""
    while True:
        if board.is_game_over(claim_draw=True):
            return "natural", _RESULT_TO_OUTCOME.get(board.result(claim_draw=True), "draw")
        if board.ply() >= config.max_plies:
            return "max-plies", "draw"
        infos = engine.analyse(board, limit, multipv=config.multipv)
        verdict = adjudicator.update(_best_white_cp(infos), board.ply(), config)
        if verdict is not None:
            return verdict
        move = select_wobble_move(infos, board.turn, config, rng)
        board.push(move)
        moves.append(move.uci())
