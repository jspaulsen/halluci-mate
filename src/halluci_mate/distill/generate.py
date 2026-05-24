"""Serial, resumable driver: seeds -> play_seed -> raw shards.

Game ids are the global seed index (zero-padded) so a resumed run continues
the same numbering. The per-game RNG is seeded from the game id, making the
wobble reproducible independent of processing order.
"""

from __future__ import annotations

import itertools
import logging
import random
from typing import TYPE_CHECKING

from halluci_mate.distill.engine_selfplay import play_seed
from halluci_mate.distill.raw_shards import ShardWriter

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from halluci_mate.distill.engine_selfplay import SeedOpening, SelfPlayConfig, _AnalysisEngine
    from halluci_mate.distill.raw_shards import EngineMeta

logger = logging.getLogger(__name__)


def run_generation(
    *,
    seeds: Iterable[SeedOpening],
    engine: _AnalysisEngine,
    config: SelfPlayConfig,
    meta: EngineMeta,
    raw_dir: Path,
    shard_size: int,
) -> int:
    """Play games for ``seeds`` into ``raw_dir``; return total games written.

    On resume, ``ShardWriter.games_written`` says how many leading seeds were
    already processed; those are skipped so the run continues deterministically.
    """
    with ShardWriter(raw_dir, meta, shard_size=shard_size) as writer:
        start = writer.games_written
        if start:
            logger.info("Resuming: skipping %d already-written games", start)
        for index, seed in enumerate(itertools.islice(seeds, start, None), start=start):
            game_id = f"game-{index:08d}"
            game = play_seed(engine, seed, config, random.Random(game_id))
            writer.add(game_id, game)
    # ShardWriter.__exit__ (on clean exit) flushes any remaining buffered games,
    # bumping games_written. Read the cursor here so that final flush is counted.
    produced = writer.games_written - start
    logger.info("Generation complete: %d new games (%d total)", produced, start + produced)
    return produced
