"""Scripted ``Predictor`` stub for search tests.

Returns a top-K keyed by board *placement* (``board_fen()``, ignoring clocks
and side-to-move so keys stay stable); falls back to the first legal moves in
board order. Implements the ``inference.Predictor`` protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import chess

from halluci_mate.eval.records import TopKEntry
from halluci_mate.inference import MovePrediction

if TYPE_CHECKING:
    from halluci_mate.game import Game


class ScriptedPolicy:
    def __init__(
        self,
        script: dict[str, list[tuple[str, float]]] | None = None,
        *,
        fallback_k: int = 3,
        mask_used: bool = True,
    ) -> None:
        self._script = script or {}
        self._fallback_k = fallback_k
        self._mask_used = mask_used

    def _entries(self, board: chess.Board, record_top_k: int) -> list[TopKEntry]:
        key = board.board_fen()
        if key in self._script:
            return [TopKEntry(move=uci, logprob=logprob) for uci, logprob in self._script[key]]
        moves = list(board.legal_moves)[: max(record_top_k, self._fallback_k)]
        return [TopKEntry(move=move.uci(), logprob=-float(i)) for i, move in enumerate(moves)]

    def predict_with_metadata(self, game: Game, *, constrained: bool | None = None, record_top_k: int = 5) -> MovePrediction:
        del constrained
        entries = self._entries(game.board, record_top_k)
        top_uci = entries[0].move
        return MovePrediction(
            played_move=chess.Move.from_uci(top_uci),
            model_move_uci=top_uci,
            raw_sample_move_uci=top_uci,
            raw_sample_legal=True,
            model_top_k=entries[:record_top_k] if record_top_k > 0 else [],
            mask_used=self._mask_used,
        )

    def predict(self, game: Game, constrained: bool | None = None) -> chess.Move:
        del constrained
        # run_search only uses predict_with_metadata; this exists to satisfy the
        # Predictor protocol and is not script-driven.
        return next(iter(game.board.legal_moves))
