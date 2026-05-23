"""Leaf evaluators for inference-time search.

A ``LeafEvaluator`` scores a board from a given side's point of view (higher is
better for that side). ``MaterialEvaluator`` is the v1 implementation:
piece-count with checkmate / draw terminal handling.
"""

from __future__ import annotations

from typing import Protocol

import chess

# Classic piece weights. Kings are omitted: they have no material value and are
# always present, so including them would only add a constant 0 to the sum.
PIECE_VALUES: dict[chess.PieceType, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}

# Finite sentinel for a decisive (checkmate) leaf: large enough to dominate any
# material swing, but a plain float so scores stay ordinary numbers.
MATE_VALUE = 1_000_000.0


class LeafEvaluator(Protocol):
    """Scores a leaf board from ``pov``'s perspective (higher = better for pov)."""

    def evaluate(self, board: chess.Board, *, pov: chess.Color) -> float: ...


class MaterialEvaluator:
    """Piece-count leaf eval with checkmate / draw terminal handling."""

    def evaluate(self, board: chess.Board, *, pov: chess.Color) -> float:
        if board.is_checkmate():
            # The side to move has been checkmated.
            return -MATE_VALUE if board.turn == pov else MATE_VALUE
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        white = _material(board, chess.WHITE)
        black = _material(board, chess.BLACK)
        score = white - black
        return score if pov == chess.WHITE else -score


def _material(board: chess.Board, color: chess.Color) -> float:
    return sum(value * len(board.pieces(piece_type, color)) for piece_type, value in PIECE_VALUES.items())
