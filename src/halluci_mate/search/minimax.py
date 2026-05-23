"""Depth-2 minimax search over the policy's legal top-K.

For each of the policy's top-K candidate moves, ``run_search`` asks the policy
for the opponent's top-K replies, scores each resulting leaf with a
``LeafEvaluator``, takes the opponent's worst-for-us reply (min), and finally
the best candidate (argmax). Leaf scoring is board-only, so the cost is up to
``1 + k`` forwards (a candidate that ends the game skips its reply pass). See
``docs/superpowers/specs/2026-05-23-inference-search-design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess

from halluci_mate.game import Game, Perspective

if TYPE_CHECKING:
    from halluci_mate.eval.records import TopKEntry
    from halluci_mate.inference import MovePrediction, Predictor
    from halluci_mate.search.leaf import LeafEvaluator


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


def run_search(policy: Predictor, leaf: LeafEvaluator, game: Game, *, k: int) -> SearchResult:
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    pov = chess.WHITE if game.perspective == Perspective.WHITE else chess.BLACK
    root = policy.predict_with_metadata(game, constrained=True, record_top_k=k)
    candidates = _legal_candidates(root.model_top_k, game.board)
    if not candidates:
        raise ValueError(f"policy returned no legal top-K candidates for {game.board.fen()}")
    scored = [_score_candidate(policy, leaf, game, move, logprob, pov, k) for move, logprob in candidates]
    scored.sort(key=lambda candidate: (-candidate.score, -candidate.policy_logprob, candidate.move.uci()))
    return SearchResult(candidates=scored, chosen=scored[0].move, policy_argmax=candidates[0][0], root_prediction=root)


def _score_candidate(policy: Predictor, leaf: LeafEvaluator, game: Game, move: chess.Move, logprob: float, pov: chess.Color, k: int) -> CandidateScore:
    board_after = game.board.copy(stack=True)  # keep move history so the branch tokenizes correctly
    board_after.push(move)
    if board_after.is_game_over():
        return CandidateScore(move=move, policy_logprob=logprob, score=leaf.evaluate(board_after, pov=pov))
    branch = Game(board=board_after, perspective=game.perspective)
    replies = policy.predict_with_metadata(branch, constrained=True, record_top_k=k)
    reply_moves = [reply for reply, _ in _legal_candidates(replies.model_top_k, board_after)]
    return CandidateScore(move=move, policy_logprob=logprob, score=_min_reply_score(board_after, reply_moves, leaf, pov))


def _min_reply_score(board_after: chess.Board, reply_moves: list[chess.Move], leaf: LeafEvaluator, pov: chess.Color) -> float:
    if not reply_moves:  # defensive: non-terminal board always has legal replies
        return leaf.evaluate(board_after, pov=pov)
    scores: list[float] = []
    for reply in reply_moves:
        leaf_board = board_after.copy(stack=False)  # material eval needs no history
        leaf_board.push(reply)
        scores.append(leaf.evaluate(leaf_board, pov=pov))
    return min(scores)  # opponent chooses the reply that minimises our score


def _legal_candidates(top_k: list[TopKEntry], board: chess.Board) -> list[tuple[chess.Move, float]]:
    """Parse top-K UCI entries to legal moves, dropping any that don't apply."""
    return [(move, entry.logprob) for entry in top_k if (move := _parse_legal(entry.move, board)) is not None]


def _parse_legal(uci: str, board: chess.Board) -> chess.Move | None:
    try:
        move = chess.Move.from_uci(uci)
    except chess.InvalidMoveError:
        return None
    return move if move in board.legal_moves else None
