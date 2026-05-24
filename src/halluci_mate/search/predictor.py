"""SearchPredictor: wrap a policy in depth-2 search, exposing the Predictor API.

Implements ``inference.Predictor`` so it is a drop-in for ``ChessInferenceEngine``
in the eval harness and in third-party play loops. A margin gate decides the
played move: search overrides the policy argmax only when its score beats the
argmax by ``margin`` pawn-equivalents (``margin`` large -> always keep argmax,
i.e. the bare policy; ``margin=0`` -> always trust search).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from halluci_mate.search.minimax import DEFAULT_QDEPTH, run_search

if TYPE_CHECKING:
    import chess

    from halluci_mate.game import Game
    from halluci_mate.inference import MovePrediction, Predictor
    from halluci_mate.search.leaf import LeafEvaluator
    from halluci_mate.search.minimax import SearchResult


class SearchPredictor:
    """Selects moves by depth-2 minimax over a wrapped policy's legal top-K."""

    def __init__(self, policy: Predictor, leaf: LeafEvaluator, *, k: int, margin: float = 0.0, quiescence: bool = True, qdepth: int = DEFAULT_QDEPTH) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}")
        if margin < 0:
            raise ValueError(f"margin must be >= 0; got {margin}")
        if qdepth < 0:
            raise ValueError(f"qdepth must be >= 0; got {qdepth}")
        self.policy = policy
        self.leaf = leaf
        self.k = k
        self.margin = margin
        self.quiescence = quiescence
        self.qdepth = qdepth

    def predict_with_metadata(self, game: Game, *, constrained: bool | None = None, record_top_k: int = 5) -> MovePrediction:
        # Search always reasons over the legal (masked) top-K; ``constrained`` is
        # accepted for Predictor parity and ignored.
        del constrained
        result = self._search(game)
        played = self._gated_move(result)
        root = result.root_prediction
        return replace(
            root,
            played_move=played,
            model_move_uci=played.uci(),
            model_top_k=root.model_top_k[:record_top_k],
        )

    def predict(self, game: Game, constrained: bool | None = None) -> chess.Move:
        # The chosen move always comes from the legal top-K, so this never raises.
        del constrained
        return self._gated_move(self._search(game))

    def _search(self, game: Game) -> SearchResult:
        return run_search(self.policy, self.leaf, game, k=self.k, quiescence=self.quiescence, qdepth=self.qdepth)

    def _gated_move(self, result: SearchResult) -> chess.Move:
        best = result.candidates[0]  # highest minimax score (score-sorted)
        argmax = next(candidate for candidate in result.candidates if candidate.move == result.policy_argmax)
        return best.move if (best.score - argmax.score) >= self.margin else argmax.move
