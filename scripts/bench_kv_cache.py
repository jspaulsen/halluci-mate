"""Wall-clock benchmark for KV-cache fast path vs full re-forward.

Plays a 40-move game (40 engine predictions, interleaved with random-legal
opponent replies) with the cache on, then with ``game.reset_cache()`` before
every predict — the pre-HAL-13 behavior. Prints both timings and the speedup.

Usage: ``uv run python scripts/bench_kv_cache.py``
"""

from __future__ import annotations

import random
import time

import chess

from halluci_mate.game import Game, Perspective
from halluci_mate.inference import ChessInferenceEngine

CHECKPOINT = "jspaulsen/halluci-mate-v1a"
SEED = 4042
MOVES_PER_SIDE = 40


def _play(engine: ChessInferenceEngine, *, reset_each_turn: bool) -> float:
    rng = random.Random(SEED)
    game = Game(board=chess.Board(), perspective=Perspective.WHITE)
    start = time.perf_counter()
    for _ in range(MOVES_PER_SIDE):
        if reset_each_turn:
            game.reset_cache()
        game.push_move(engine.predict(game))
        if game.board.is_game_over():
            break
        game.push_move(rng.choice(list(game.board.legal_moves)))
        if game.board.is_game_over():
            break
    return time.perf_counter() - start


def main() -> None:
    engine = ChessInferenceEngine.from_checkpoint(CHECKPOINT, device="cpu")

    # Warm up to amortize first-call overhead (CUDA init, lazy imports, etc.)
    warmup = Game(board=chess.Board(), perspective=Perspective.WHITE)
    engine.predict(warmup)

    with_cache = _play(engine, reset_each_turn=False)
    without_cache = _play(engine, reset_each_turn=True)

    print(f"With KV cache:         {with_cache:7.2f}s")
    print(f"Without (reset/turn):  {without_cache:7.2f}s")
    print(f"Speedup:               {without_cache / with_cache:6.2f}x")


if __name__ == "__main__":
    main()
