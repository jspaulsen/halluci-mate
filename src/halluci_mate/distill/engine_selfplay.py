"""Full-strength Stockfish self-play with a MultiPV diversity 'wobble'.

Seeds from a human opening prefix, then plays both sides with Stockfish,
sampling among near-equal moves so a fixed-budget (otherwise deterministic)
engine produces a diverse corpus. Games are adjudicated (resign / draw / cap)
to stay finite and avoid dead-drawn shuffles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess

if TYPE_CHECKING:
    import chess.engine

# Mate scores collapse to a finite cp so adjudication/selection arithmetic
# never sees ``None`` (mirrors vs_stockfish ``_MATE_SCORE_CP``).
_MATE_SCORE_CP = 100_000


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
