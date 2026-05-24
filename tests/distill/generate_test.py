from __future__ import annotations

from typing import TYPE_CHECKING

import chess
import chess.engine

from halluci_mate.distill.engine_selfplay import SelfPlayConfig
from halluci_mate.distill.generate import run_generation
from halluci_mate.distill.raw_shards import EngineMeta, read_raw_games

if TYPE_CHECKING:
    from pathlib import Path

META = EngineMeta(depth=4, multipv=2, wobble_cp=30, sf_version="test", nnue="t")


class _BalancedEngine:
    def analyse(self, board: chess.Board, limit: chess.engine.Limit, *, multipv: int) -> list[chess.engine.InfoDict]:
        del limit
        return [{"score": chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE), "pv": [m]} for m in list(board.legal_moves)[:multipv]]


def _seeds(n: int) -> list[tuple[str, list[str]]]:
    return [(f"s{i}", ["e2e4", "e7e5"]) for i in range(n)]


def test_run_generation_writes_all_games(tmp_path: Path) -> None:
    config = SelfPlayConfig(seed_plies=2, max_plies=4, draw_min_ply=999, multipv=2, wobble_cp=10_000)
    written = run_generation(seeds=_seeds(5), engine=_BalancedEngine(), config=config, meta=META, raw_dir=tmp_path, shard_size=2)
    assert written == 5
    games = list(read_raw_games(tmp_path))
    assert [g.game_id for g in games] == [f"game-{i:08d}" for i in range(5)]


def test_run_generation_resumes_without_duplicates(tmp_path: Path) -> None:
    config = SelfPlayConfig(seed_plies=2, max_plies=4, draw_min_ply=999, multipv=2, wobble_cp=10_000)
    run_generation(seeds=_seeds(2), engine=_BalancedEngine(), config=config, meta=META, raw_dir=tmp_path, shard_size=2)
    # Resume: pass the full seed list again; first 2 must be skipped via the cursor.
    written = run_generation(seeds=_seeds(5), engine=_BalancedEngine(), config=config, meta=META, raw_dir=tmp_path, shard_size=2)
    assert written == 3
    ids = [g.game_id for g in read_raw_games(tmp_path)]
    assert ids == [f"game-{i:08d}" for i in range(5)]
    assert len(ids) == len(set(ids))


def test_run_generation_resumes_from_partial_trailing_shard(tmp_path: Path) -> None:
    # The operationally likely case: a run killed mid-shard leaves a partial
    # trailing shard. Resume must derive the next shard index from the existing
    # shards (not overwrite the partial) and append without duplicates.
    config = SelfPlayConfig(seed_plies=2, max_plies=4, draw_min_ply=999, multipv=2, wobble_cp=10_000)
    # 3 games, shard_size=2 -> shard_00000 (2 games) + shard_00001 (1 game, partial).
    assert run_generation(seeds=_seeds(3), engine=_BalancedEngine(), config=config, meta=META, raw_dir=tmp_path, shard_size=2) == 3
    assert len(list(tmp_path.glob("shard_*.parquet"))) == 2
    # Resume: 2 new games land in a fresh shard_00002, leaving the partial intact.
    assert run_generation(seeds=_seeds(5), engine=_BalancedEngine(), config=config, meta=META, raw_dir=tmp_path, shard_size=2) == 2
    ids = [g.game_id for g in read_raw_games(tmp_path)]
    assert ids == [f"game-{i:08d}" for i in range(5)]
    assert len(ids) == len(set(ids))
