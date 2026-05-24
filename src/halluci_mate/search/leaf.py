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


# King-safety weights, in pawn-equivalents. Kept well below a minor piece (3.0)
# so material always dominates; these only color otherwise-comparable lines.
W_KING_SHIELD = 0.15  # bonus per friendly pawn shielding the king
W_KING_OPEN_FILE = 0.25  # penalty per open/semi-open file on or next to the king


class MaterialKingSafetyEvaluator:
    """Material plus a small king-safety term (pawn shield + open files)."""

    def __init__(self) -> None:
        self._material = MaterialEvaluator()

    def evaluate(self, board: chess.Board, *, pov: chess.Color) -> float:
        if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
            return self._material.evaluate(board, pov=pov)  # terminal: material handles mate/draw
        base = self._material.evaluate(board, pov=pov)
        safety = _king_safety(board, chess.WHITE) - _king_safety(board, chess.BLACK)
        return base + (safety if pov == chess.WHITE else -safety)


def _king_safety(board: chess.Board, color: chess.Color) -> float:
    return W_KING_SHIELD * _king_shield(board, color) - W_KING_OPEN_FILE * _king_open_files(board, color)


def _king_shield(board: chess.Board, color: chess.Color) -> int:
    """Count friendly pawns on the three files around the king, on the two ranks in front of it."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    forward = 1 if color == chess.WHITE else -1
    pawn = chess.Piece(chess.PAWN, color)
    shield = 0
    for df in (-1, 0, 1):
        file = king_file + df
        if not 0 <= file <= 7:
            continue
        for dr in (1, 2):
            rank = king_rank + forward * dr
            if 0 <= rank <= 7 and board.piece_at(chess.square(file, rank)) == pawn:
                shield += 1
    return shield


def _king_open_files(board: chess.Board, color: chess.Color) -> int:
    """Count files on/next to the king that hold no friendly pawn (semi-open or open)."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    king_file = chess.square_file(king_sq)
    pawn = chess.Piece(chess.PAWN, color)
    exposed = 0
    for df in (-1, 0, 1):
        file = king_file + df
        if not 0 <= file <= 7:
            continue
        if not any(board.piece_at(chess.square(file, rank)) == pawn for rank in range(8)):
            exposed += 1
    return exposed
