"""SearchPredictor: wrap a policy in depth-2 search, exposing the Predictor API.

Implements ``inference.Predictor`` so it is a drop-in for ``ChessInferenceEngine``
in the eval harness and in third-party play loops.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from halluci_mate.search.minimax import run_search

if TYPE_CHECKING:
    import chess

    from halluci_mate.game import Game
    from halluci_mate.inference import MovePrediction, Predictor
    from halluci_mate.search.leaf import LeafEvaluator


class SearchPredictor:
    """Selects moves by depth-2 minimax over a wrapped policy's legal top-K."""

    def __init__(self, policy: Predictor, leaf: LeafEvaluator, *, k: int) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}")
        self.policy = policy
        self.leaf = leaf
        self.k = k

    def predict_with_metadata(self, game: Game, *, constrained: bool | None = None, record_top_k: int = 5) -> MovePrediction:
        # Search always reasons over the legal (masked) top-K; ``constrained`` is
        # accepted for Predictor parity and ignored.
        del constrained
        result = run_search(self.policy, self.leaf, game, k=self.k)
        root = result.root_prediction
        return replace(
            root,
            played_move=result.chosen,
            model_move_uci=result.chosen.uci(),
            model_top_k=root.model_top_k[:record_top_k],
        )

    def predict(self, game: Game, constrained: bool | None = None) -> chess.Move:
        # The chosen move always comes from the legal top-K, so this never raises.
        del constrained
        return run_search(self.policy, self.leaf, game, k=self.k).chosen
