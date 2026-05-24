"""Tests for the material leaf evaluator."""

from __future__ import annotations

import chess
import pytest

from halluci_mate.search.leaf import MATE_VALUE, MaterialEvaluator, MaterialKingSafetyEvaluator


def test_start_position_is_balanced() -> None:
    evaluator = MaterialEvaluator()
    board = chess.Board()
    assert evaluator.evaluate(board, pov=chess.WHITE) == 0.0
    assert evaluator.evaluate(board, pov=chess.BLACK) == 0.0


def test_material_advantage_is_pov_signed() -> None:
    evaluator = MaterialEvaluator()
    # Standard start position with the black queen removed: White is +9.
    board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert evaluator.evaluate(board, pov=chess.WHITE) == 9.0
    assert evaluator.evaluate(board, pov=chess.BLACK) == -9.0


def test_checkmate_is_decisive_for_the_mating_side() -> None:
    evaluator = MaterialEvaluator()
    board = chess.Board()
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:  # fool's mate; White is mated
        board.push_uci(uci)
    assert board.is_checkmate()
    assert board.turn == chess.WHITE
    assert evaluator.evaluate(board, pov=chess.WHITE) == -MATE_VALUE
    assert evaluator.evaluate(board, pov=chess.BLACK) == MATE_VALUE


def test_stalemate_is_zero() -> None:
    evaluator = MaterialEvaluator()
    board = chess.Board("k7/8/1Q6/8/8/8/8/7K b - - 0 1")
    assert board.is_stalemate()
    assert evaluator.evaluate(board, pov=chess.WHITE) == 0.0
    assert evaluator.evaluate(board, pov=chess.BLACK) == 0.0


def test_insufficient_material_is_zero() -> None:
    evaluator = MaterialEvaluator()
    board = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1")  # K vs K
    assert board.is_insufficient_material()
    assert evaluator.evaluate(board, pov=chess.WHITE) == 0.0
    assert evaluator.evaluate(board, pov=chess.BLACK) == 0.0


# Equal material (3 pawns each). White's king is fully shielded (f2,g2,h2);
# Black's pawns are on the queenside (a7,b7,c7), leaving f/g/h open by its king.
_KING_SAFETY_FEN = "6k1/ppp5/8/8/8/8/5PPP/6K1 w - - 0 1"


def test_king_safety_rewards_shield_and_penalizes_open_files() -> None:
    board = chess.Board(_KING_SAFETY_FEN)
    assert MaterialEvaluator().evaluate(board, pov=chess.WHITE) == 0.0  # material is equal
    evaluator = MaterialKingSafetyEvaluator()
    white = evaluator.evaluate(board, pov=chess.WHITE)
    assert white > 0.0  # White's safer king scores higher
    assert evaluator.evaluate(board, pov=chess.BLACK) == pytest.approx(-white)  # POV-antisymmetric


def test_king_safety_eval_is_symmetric_at_the_start() -> None:
    assert MaterialKingSafetyEvaluator().evaluate(chess.Board(), pov=chess.WHITE) == 0.0


def test_king_safety_eval_is_decisive_on_checkmate() -> None:
    board = chess.Board()
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:  # fool's mate; White is mated
        board.push_uci(uci)
    assert MaterialKingSafetyEvaluator().evaluate(board, pov=chess.WHITE) == -MATE_VALUE
