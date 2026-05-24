"""End-to-end distillation pipeline test against a real Stockfish binary.

Skipped automatically when no ``stockfish`` binary is on PATH. Run explicitly:
    uv run pytest tests/integration/distill_pipeline_test.py -v
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import chess.engine
import pytest

from halluci_mate.data_preparation import create_tokenizer
from halluci_mate.distill.engine_selfplay import SelfPlayConfig
from halluci_mate.distill.generate import run_generation
from halluci_mate.distill.raw_shards import EngineMeta, read_raw_games
from halluci_mate.distill.tokenize import tokenize_engine_shards

if TYPE_CHECKING:
    from pathlib import Path

_STOCKFISH = shutil.which("stockfish")
pytestmark = pytest.mark.skipif(_STOCKFISH is None, reason="stockfish binary not on PATH")

_SEEDS = [("g-ruy", ["e2e4", "e7e5", "g1f3", "b8c6"]), ("g-fr", ["e2e4", "e7e6", "d2d4", "d7d5"])]


def test_generate_then_tokenize(tmp_path: Path) -> None:
    config = SelfPlayConfig(depth=6, multipv=3, max_plies=24, draw_min_ply=10, draw_plies=4, seed_plies=4)
    raw_dir = tmp_path / "_raw"
    engine = chess.engine.SimpleEngine.popen_uci(_STOCKFISH)
    try:
        engine.configure({"Threads": 1, "Hash": 16})
        meta = EngineMeta(depth=config.depth, multipv=config.multipv, wobble_cp=config.wobble_cp, sf_version="sf", nnue="nnue")
        written = run_generation(seeds=_SEEDS, engine=engine, config=config, meta=meta, raw_dir=raw_dir, shard_size=10)
    finally:
        engine.quit()

    assert written == 2
    games = list(read_raw_games(raw_dir))
    assert len(games) == 2
    # The seed prefix is a hard copy in play_seed; the engine only plays from
    # ply seed_plies onward, so these first moves are deterministic.
    assert all(g.moves_uci[:4] == seed[1] for g, seed in zip(games, _SEEDS, strict=True))

    shard_dir = tmp_path / "_tok"
    shard_dir.mkdir()
    total = tokenize_engine_shards(raw_dir, create_tokenizer(), shard_dir)
    assert total >= 2
    assert list(shard_dir.glob("shard_*.parquet"))
