from __future__ import annotations

from pathlib import Path  # noqa: TC003 — used at runtime (pytest tmp_path type annotation)

import pytest

from halluci_mate.distill.engine_selfplay import SelfPlayGame
from halluci_mate.distill.raw_shards import EngineMeta, RawGameRow, ShardWriter, read_raw_games

META = EngineMeta(depth=18, multipv=4, wobble_cp=30, sf_version="test", nnue="test.nnue")


def _game(source: str) -> SelfPlayGame:
    return SelfPlayGame(seed_source=source, seed_plies=1, moves_uci=["e2e4", "e7e5"], outcome="draw", termination="max-plies")


def test_row_roundtrips_through_model() -> None:
    row = RawGameRow(game_id="g0", seed_source="s", seed_plies=1, moves_uci=["e2e4"], outcome="draw", termination="natural", **META.as_columns())
    assert RawGameRow.model_validate(row.model_dump()) == row


def test_shardwriter_writes_full_and_partial_shards(tmp_path: Path) -> None:
    with ShardWriter(tmp_path, META, shard_size=2) as writer:
        for i in range(5):
            writer.add(f"g{i}", _game(f"s{i}"))
    shards = sorted(tmp_path.glob("shard_*.parquet"))
    assert len(shards) == 3  # 2 + 2 + 1
    games = list(read_raw_games(tmp_path))
    assert [g.game_id for g in games] == ["g0", "g1", "g2", "g3", "g4"]
    assert all(g.outcome == "draw" for g in games)


def test_shardwriter_resumes_from_manifest(tmp_path: Path) -> None:
    with ShardWriter(tmp_path, META, shard_size=2) as writer:
        writer.add("g0", _game("s0"))
        writer.add("g1", _game("s1"))
    # Reopen: cursor should report 2 already written, next shard index 1.
    writer2 = ShardWriter(tmp_path, META, shard_size=2)
    assert writer2.games_written == 2
    with writer2:
        writer2.add("g2", _game("s2"))
    assert sorted(p.name for p in tmp_path.glob("shard_*.parquet")) == ["shard_00000.parquet", "shard_00001.parquet"]
    assert [g.game_id for g in read_raw_games(tmp_path)] == ["g0", "g1", "g2"]


def test_shardwriter_no_partial_file_left_on_clean_exit(tmp_path: Path) -> None:
    with ShardWriter(tmp_path, META, shard_size=2) as writer:
        writer.add("g0", _game("s0"))
    assert list(tmp_path.glob("*.tmp")) == []


def test_resume_does_not_overwrite_orphan_shard_when_manifest_lags(tmp_path: Path) -> None:
    # Simulate the one crash window: a shard was published but the process died
    # before the manifest cursor caught up. Deleting the manifest models a
    # cursor that undercounts the on-disk shards. Resume must derive the next
    # index from the existing shard files and NOT overwrite the orphan.
    with ShardWriter(tmp_path, META, shard_size=1) as writer:
        writer.add("g0", _game("s0"))  # publishes shard_00000 + manifest
    (tmp_path / "manifest.json").unlink()  # manifest lost / behind disk

    resumed = ShardWriter(tmp_path, META, shard_size=1)
    assert resumed.games_written == 0  # cursor reset, but _next_index reads the glob
    with resumed:
        resumed.add("g1", _game("s1"))

    assert sorted(p.name for p in tmp_path.glob("shard_*.parquet")) == ["shard_00000.parquet", "shard_00001.parquet"]
    assert [g.game_id for g in read_raw_games(tmp_path)] == ["g0", "g1"]


def test_buffer_discarded_on_exception(tmp_path: Path) -> None:
    # A crash mid-session must drop the unflushed buffer: no short shard, no
    # manifest, so the cursor never credits games that were never published.
    with pytest.raises(RuntimeError, match="boom"), ShardWriter(tmp_path, META, shard_size=10) as writer:
        writer.add("g0", _game("s0"))
        raise RuntimeError("boom")
    assert list(tmp_path.glob("shard_*.parquet")) == []
    assert not (tmp_path / "manifest.json").exists()
