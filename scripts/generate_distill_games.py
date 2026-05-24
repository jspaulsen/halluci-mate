"""Stage 1: generate a Stockfish self-play distillation corpus (raw game shards).

Streams high-Elo Rapid+Classical Lichess openings as seeds, plays both sides
with full-strength Stockfish + a MultiPV wobble, and writes resumable raw
parquet shards. Re-run with the same args to resume an interrupted job.

Usage:
    uv run python scripts/generate_distill_games.py --num-games 100000 --raw-dir data/distill/_raw
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Annotated

import chess.engine
import typer
from datasets import load_dataset

from halluci_mate.distill.engine_selfplay import SelfPlayConfig
from halluci_mate.distill.generate import run_generation
from halluci_mate.distill.raw_shards import EngineMeta
from halluci_mate.distill.seeds import iter_seed_openings
from halluci_mate.logging_setup import configure_script_logging

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/distill/_raw")
DEFAULT_NUM_GAMES = 1_000_000
DEFAULT_SHARD_SIZE = 10_000
DEFAULT_MIN_ELO = 2000
DEFAULT_MAX_RATING_DIFF = 30
DEFAULT_MAX_ELO_GAP = 200
# Stockfish engine resource defaults: one thread per process (workers run many
# engines in parallel) and a modest per-engine transposition table.
DEFAULT_THREADS = 1
DEFAULT_HASH_MB = 256


def _engine_meta(engine: chess.engine.SimpleEngine, config: SelfPlayConfig) -> EngineMeta:
    # The NNUE net name is a UCI *option* default (EvalFile), not part of
    # ``engine.id`` (which carries only name/author) -- record it for provenance.
    nnue = str(engine.options["EvalFile"].default) if "EvalFile" in engine.options else "unknown"
    return EngineMeta(
        depth=config.depth,
        multipv=config.multipv,
        wobble_cp=config.wobble_cp,
        sf_version=engine.id.get("name", "unknown"),
        nnue=nnue,
    )


def main(
    stockfish_path: Annotated[str, typer.Option(help="Path to the Stockfish binary")] = "stockfish",
    num_games: Annotated[int, typer.Option(help="Total games to generate (cap applies across resumes)")] = DEFAULT_NUM_GAMES,
    raw_dir: Annotated[Path, typer.Option(help="Output directory for raw game shards")] = DEFAULT_RAW_DIR,
    shard_size: Annotated[int, typer.Option(help="Games per shard file")] = DEFAULT_SHARD_SIZE,
    depth: Annotated[int, typer.Option(help="Stockfish search depth (cost dial)")] = SelfPlayConfig.depth,
    threads: Annotated[int, typer.Option(help="Stockfish Threads per engine")] = DEFAULT_THREADS,
    hash_mb: Annotated[int, typer.Option(help="Stockfish Hash (MB) per engine")] = DEFAULT_HASH_MB,
    min_elo: Annotated[int, typer.Option(help="Minimum Elo for both seed players")] = DEFAULT_MIN_ELO,
    max_rating_diff: Annotated[int, typer.Option(help="Max |RatingDiff| for either seed player")] = DEFAULT_MAX_RATING_DIFF,
    max_elo_gap: Annotated[int, typer.Option(help="Max |WhiteElo - BlackElo| for the seed")] = DEFAULT_MAX_ELO_GAP,
) -> None:
    """Generate the Stage 1 raw self-play corpus."""
    configure_script_logging(__name__)
    config = SelfPlayConfig(depth=depth)

    stream = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)
    seeds = iter_seed_openings(stream, config.seed_plies, min_elo, max_rating_diff, max_elo_gap)
    capped_seeds = itertools.islice(seeds, num_games)

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        engine.configure({"Threads": threads, "Hash": hash_mb})
        meta = _engine_meta(engine, config)
        run_generation(seeds=capped_seeds, engine=engine, config=config, meta=meta, raw_dir=raw_dir, shard_size=shard_size)
    finally:
        engine.quit()


if __name__ == "__main__":
    typer.run(main)
