"""Tests for SearchPredictor (the Predictor-protocol wrapper around search)."""

from __future__ import annotations

import chess
import pytest

from halluci_mate.game import Game, Perspective
from halluci_mate.search.leaf import MaterialEvaluator
from halluci_mate.search.predictor import SearchPredictor
from tests.helpers.search_policy import ScriptedPolicy

# Back-rank position: a1a8 is mate; policy ranks the quiet g1f1 first.
_BACK_RANK_FEN = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
_BACK_RANK_KEY = "6k1/5ppp/8/8/8/8/5PPP/R5K1"


def _white_game(fen: str = _BACK_RANK_FEN) -> Game:
    return Game(board=chess.Board(fen), perspective=Perspective.WHITE)


# Equal-material position; Qxd5 wins a truly free pawn (no recapture) -> +1 edge
# over the quiet Qd1. Lets us probe the margin gate with a small, known edge.
_FREE_PAWN_FEN = "6k1/6pp/8/3p4/8/8/3Q2PP/6K1 w - - 0 1"
_FREE_PAWN_KEY = "6k1/6pp/8/3p4/8/8/3Q2PP/6K1"


def test_gate_overrides_when_edge_clears_margin() -> None:
    policy = ScriptedPolicy({_FREE_PAWN_KEY: [("d2d1", -0.1), ("d2d5", -0.5)]})
    # quiescence=False: score the position at depth-2 only so the +1 pawn edge is
    # visible without quiescence reaching the pawn on subsequent plies.
    predictor = SearchPredictor(policy=policy, leaf=MaterialEvaluator(), k=2, margin=0.5, quiescence=False)
    assert predictor.predict(_white_game(_FREE_PAWN_FEN)) == chess.Move.from_uci("d2d5")


def test_gate_keeps_argmax_when_edge_below_margin() -> None:
    policy = ScriptedPolicy({_FREE_PAWN_KEY: [("d2d1", -0.1), ("d2d5", -0.5)]})
    predictor = SearchPredictor(policy=policy, leaf=MaterialEvaluator(), k=2, margin=2.0, quiescence=False)
    assert predictor.predict(_white_game(_FREE_PAWN_FEN)) == chess.Move.from_uci("d2d1")


def test_negative_margin_raises() -> None:
    with pytest.raises(ValueError, match="margin must be >= 0"):
        SearchPredictor(policy=ScriptedPolicy(), leaf=MaterialEvaluator(), k=2, margin=-1.0)


def test_prediction_replaces_move_but_preserves_policy_metadata() -> None:
    policy = ScriptedPolicy({_BACK_RANK_KEY: [("g1f1", -0.1), ("a1a8", -0.5)]}, mask_used=False)
    predictor = SearchPredictor(policy=policy, leaf=MaterialEvaluator(), k=2)

    prediction = predictor.predict_with_metadata(_white_game(), record_top_k=1)

    # model_move is the search pick (the mate), not the policy argmax.
    assert prediction.model_move_uci == "a1a8"
    assert prediction.played_move == chess.Move.from_uci("a1a8")
    # The rest passes through from the policy's own root prediction.
    assert prediction.raw_sample_move_uci == "g1f1"
    assert prediction.mask_used is False
    assert [entry.move for entry in prediction.model_top_k] == ["g1f1"]  # sliced to record_top_k=1


def test_predict_returns_the_chosen_move() -> None:
    policy = ScriptedPolicy({_BACK_RANK_KEY: [("g1f1", -0.1), ("a1a8", -0.5)]})
    predictor = SearchPredictor(policy=policy, leaf=MaterialEvaluator(), k=2)
    assert predictor.predict(_white_game()) == chess.Move.from_uci("a1a8")


def test_k_below_one_raises() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        SearchPredictor(policy=ScriptedPolicy(), leaf=MaterialEvaluator(), k=0)
