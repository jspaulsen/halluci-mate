"""Depth-2 minimax: the LM proposes root candidates, board search verifies them.

The model's top-K supplies the root candidate moves; the opponent's reply is
searched full-width over *all* legal moves (the leaf is board-only, so the LM is
not queried below the root), and each candidate is scored by the opponent's
worst-for-us reply. See
``docs/superpowers/specs/2026-05-23-search-strength-v2-design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess

from halluci_mate.game import Perspective

if TYPE_CHECKING:
    from halluci_mate.eval.records import TopKEntry
    from halluci_mate.game import Game
    from halluci_mate.inference import MovePrediction, Predictor
    from halluci_mate.search.leaf import LeafEvaluator

# Default quiescence search depth cap (plies beyond the depth-2 root/reply).
DEFAULT_QDEPTH = 4


@dataclass(frozen=True)
class CandidateScore:
    """One root candidate move with its policy logprob and minimax score."""

    move: chess.Move
    policy_logprob: float
    score: float


@dataclass(frozen=True)
class SearchResult:
    """Full output of a search; the seam for future search->policy distillation."""

    candidates: list[CandidateScore]  # score-sorted, best first
    chosen: chess.Move
    policy_argmax: chess.Move
    root_prediction: MovePrediction


def run_search(policy: Predictor, leaf: LeafEvaluator, game: Game, *, k: int, quiescence: bool = True, qdepth: int = DEFAULT_QDEPTH) -> SearchResult:
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if qdepth < 0:
        raise ValueError(f"qdepth must be >= 0; got {qdepth}")
    pov = chess.WHITE if game.perspective == Perspective.WHITE else chess.BLACK
    root = policy.predict_with_metadata(game, constrained=True, record_top_k=k)
    candidates = _legal_candidates(root.model_top_k, game.board)
    if not candidates:
        raise ValueError(f"policy returned no legal top-K candidates for {game.board.fen()}")
    scored = [_score_candidate(leaf, game.board, move, logprob, pov, quiescence=quiescence, qdepth=qdepth) for move, logprob in candidates]
    scored.sort(key=lambda candidate: (-candidate.score, -candidate.policy_logprob, candidate.move.uci()))
    return SearchResult(candidates=scored, chosen=scored[0].move, policy_argmax=candidates[0][0], root_prediction=root)


def _score_candidate(leaf: LeafEvaluator, board_before: chess.Board, move: chess.Move, logprob: float, pov: chess.Color, *, quiescence: bool, qdepth: int) -> CandidateScore:
    board_after = board_before.copy(stack=False)  # no move history needed below the root
    board_after.push(move)
    if board_after.is_game_over():
        return CandidateScore(move=move, policy_logprob=logprob, score=leaf.evaluate(board_after, pov=pov))
    # Opponent minimizes our score over ALL legal replies (full-width, board-only).
    score = min(_reply_value(leaf, board_after, reply, pov, quiescence=quiescence, qdepth=qdepth) for reply in board_after.legal_moves)
    return CandidateScore(move=move, policy_logprob=logprob, score=score)


def _reply_value(leaf: LeafEvaluator, board_after: chess.Board, reply: chess.Move, pov: chess.Color, *, quiescence: bool, qdepth: int) -> float:
    child = board_after.copy(stack=False)
    child.push(reply)
    if quiescence:
        return quiesce(child, leaf, pov, qdepth)
    return leaf.evaluate(child, pov=pov)


def quiesce(board: chess.Board, leaf: LeafEvaluator, pov: chess.Color, qdepth: int) -> float:
    """Fixed-POV minimax that extends only forcing moves to a quiet leaf.

    The leaf is always scored from ``pov``, so a node maximizes when
    ``board.turn == pov`` and minimizes otherwise. Captures are always
    extended; when the side to move is in check, all legal moves (evasions)
    are extended. ``qdepth`` caps the recursion for guaranteed termination.
    """
    legal = list(board.legal_moves)
    if not legal:
        return leaf.evaluate(board, pov=pov)  # checkmate or stalemate
    stand_pat = leaf.evaluate(board, pov=pov)
    if qdepth == 0:
        return stand_pat
    in_check = board.is_check()
    if in_check:
        forcing = legal
    else:
        forcing = [move for move in legal if board.is_capture(move)]
        if not forcing:
            return stand_pat  # quiet position
    maximizing = board.turn == pov
    best = (float("-inf") if maximizing else float("inf")) if in_check else stand_pat
    for move in forcing:
        child = board.copy(stack=False)
        child.push(move)
        value = quiesce(child, leaf, pov, qdepth - 1)
        best = max(best, value) if maximizing else min(best, value)
    return best


def _legal_candidates(top_k: list[TopKEntry], board: chess.Board) -> list[tuple[chess.Move, float]]:
    """Parse top-K UCI entries to legal moves, dropping any that don't apply."""
    return [(move, entry.logprob) for entry in top_k if (move := _parse_legal(entry.move, board)) is not None]


def _parse_legal(uci: str, board: chess.Board) -> chess.Move | None:
    try:
        move = chess.Move.from_uci(uci)
    except chess.InvalidMoveError:
        return None
    return move if move in board.legal_moves else None
