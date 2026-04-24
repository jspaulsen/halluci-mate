import random

import chess

from halluci_mate.game import Game, Perspective
from halluci_mate.inference import ChessInferenceEngine

CHECKPOINT = "jspaulsen/halluci-mate-v1a"
SEED = 4042


def _play_game(engine: ChessInferenceEngine, rng: random.Random, *, reset_cache_each_turn: bool) -> list[chess.Move]:
    """Play 3 engine predictions interleaved with 2 random-legal opponent moves.

    Shared driver for the round-trip test and the KV-cache equivalence test.
    When ``reset_cache_each_turn`` is True, the engine re-forwards the full
    history on every ``predict`` — the pre-KV-cache behavior.
    """
    game = Game(board=chess.Board(), perspective=Perspective.WHITE)

    for turn in range(3):
        if reset_cache_each_turn:
            game.reset_cache()
        game.push_move(engine.predict(game))
        if turn < 2:
            game.push_move(rng.choice(list(game.board.legal_moves)))

    return list(game.board.move_stack)


class TestChessInterface:
    def test_chess_inference(self):
        engine = ChessInferenceEngine.from_checkpoint(CHECKPOINT, device="cpu")
        _play_game(engine, random.Random(SEED), reset_cache_each_turn=False)

    def test_kv_cache_matches_uncached(self):
        """Cache-on and cache-off must produce identical move sequences.

        Guards against numerical drift from chunked vs single forward passes.
        """
        engine = ChessInferenceEngine.from_checkpoint(CHECKPOINT, device="cpu")
        with_cache = _play_game(engine, random.Random(SEED), reset_cache_each_turn=False)
        without_cache = _play_game(engine, random.Random(SEED), reset_cache_each_turn=True)
        assert with_cache == without_cache
