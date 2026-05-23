"""Tests for depth-2 minimax search over the policy's top-K."""

from __future__ import annotations

import chess
import pytest

from halluci_mate.game import Game, Perspective
from halluci_mate.search.leaf import MaterialEvaluator
from halluci_mate.search.minimax import run_search
from tests.helpers.search_policy import ScriptedPolicy

# Back-rank position: Ra1-a8 is mate; g1f1 is a quiet alternative.
_BACK_RANK_FEN = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
_BACK_RANK_KEY = "6k1/5ppp/8/8/8/8/5PPP/R5K1"

# Trap position: White Qxd5 wins a pawn but hangs the queen to exd5.
_TRAP_FEN = "6k1/5ppp/4p3/3p4/8/8/6PP/3Q2K1 w - - 0 1"
_TRAP_ROOT_KEY = "6k1/5ppp/4p3/3p4/8/8/6PP/3Q2K1"
_TRAP_AFTER_QXD5_KEY = "6k1/5ppp/4p3/3Q4/8/8/6PP/6K1"

# Quiet equal-material position for tie-break checks.
_QUIET_FEN = "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1"
_QUIET_KEY = "6k1/5ppp/8/8/8/8/5PPP/6K1"


def _white_game(fen: str) -> Game:
    return Game(board=chess.Board(fen), perspective=Perspective.WHITE)


def test_picks_mate_over_quiet_move_and_overrides_policy_argmax() -> None:
    # Sanity: a1a8 really is mate from this position.
    sanity = chess.Board(_BACK_RANK_FEN)
    sanity.push_uci("a1a8")
    assert sanity.is_checkmate()

    # Policy ranks the quiet move first; search must override to the mate.
    policy = ScriptedPolicy({_BACK_RANK_KEY: [("g1f1", -0.1), ("a1a8", -0.5)]})
    result = run_search(policy, MaterialEvaluator(), _white_game(_BACK_RANK_FEN), k=2)

    assert result.policy_argmax == chess.Move.from_uci("g1f1")
    assert result.chosen == chess.Move.from_uci("a1a8")


def test_avoids_hanging_a_piece_via_opponent_min() -> None:
    policy = ScriptedPolicy(
        {
            _TRAP_ROOT_KEY: [("d1d5", -0.1), ("d1d2", -0.5)],  # policy greedily prefers the capture
            _TRAP_AFTER_QXD5_KEY: [("e6d5", -0.1)],            # opponent recaptures the queen
        }
    )
    result = run_search(policy, MaterialEvaluator(), _white_game(_TRAP_FEN), k=2)

    assert result.policy_argmax == chess.Move.from_uci("d1d5")
    assert result.chosen == chess.Move.from_uci("d1d2")


def test_candidate_that_is_terminal_after_my_move_is_scored_without_a_reply() -> None:
    # Only the mate is offered; pushing it ends the game (no reply layer).
    policy = ScriptedPolicy({_BACK_RANK_KEY: [("a1a8", -0.1)]})
    result = run_search(policy, MaterialEvaluator(), _white_game(_BACK_RANK_FEN), k=1)
    assert result.chosen == chess.Move.from_uci("a1a8")


def test_tie_break_prefers_logprob_then_uci() -> None:
    leaf = MaterialEvaluator()

    # Equal score (both quiet, material 0) and equal logprob -> lower UCI wins.
    by_uci = ScriptedPolicy({_QUIET_KEY: [("g1h1", -0.5), ("g1f1", -0.5)]})
    assert run_search(by_uci, leaf, _white_game(_QUIET_FEN), k=2).chosen == chess.Move.from_uci("g1f1")

    # Equal score, different logprob -> higher logprob wins.
    by_logprob = ScriptedPolicy({_QUIET_KEY: [("g1f1", -0.9), ("g1h1", -0.1)]})
    assert run_search(by_logprob, leaf, _white_game(_QUIET_FEN), k=2).chosen == chess.Move.from_uci("g1h1")


def test_k_below_one_raises() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        run_search(ScriptedPolicy(), MaterialEvaluator(), _white_game(_QUIET_FEN), k=0)
