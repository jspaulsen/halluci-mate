"""Durable raw-game shard layer for the distillation generator.

Games are written as atomic parquet shards plus a ``manifest.json`` holding a
resumable ``games_written`` cursor. Re-tokenizing/re-splitting reads these
shards and never re-runs Stockfish.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 — used at runtime
from typing import TYPE_CHECKING, Literal, Self

from datasets import Dataset, load_dataset
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

    from halluci_mate.distill.engine_selfplay import SelfPlayGame

MANIFEST_FILENAME = "manifest.json"
_SHARD_GLOB = "shard_*.parquet"
# Zero-pad width for shard indices so lexicographic == numeric sort (supports up to 1e5 shards).
_SHARD_INDEX_WIDTH = 5


@dataclass(frozen=True)
class EngineMeta:
    """Engine provenance recorded on every row (Stockfish output is build-dependent)."""

    depth: int
    multipv: int
    wobble_cp: int
    sf_version: str
    nnue: str

    def as_columns(self) -> dict[str, object]:
        return {"engine_depth": self.depth, "engine_multipv": self.multipv, "engine_wobble_cp": self.wobble_cp, "engine_sf_version": self.sf_version, "engine_nnue": self.nnue}


class RawGameRow(BaseModel):
    """One on-disk game row (flattened engine columns for robust parquet typing)."""

    model_config = ConfigDict(frozen=True)

    game_id: str
    seed_source: str
    seed_plies: int
    moves_uci: list[str]
    outcome: Literal["white", "black", "draw"]
    termination: Literal["natural", "adjudicated-win", "adjudicated-draw", "max-plies"]
    engine_depth: int
    engine_multipv: int
    engine_wobble_cp: int
    engine_sf_version: str
    engine_nnue: str


def game_to_row(game_id: str, game: SelfPlayGame, meta: EngineMeta) -> RawGameRow:
    return RawGameRow(
        game_id=game_id,
        seed_source=game.seed_source,
        seed_plies=game.seed_plies,
        moves_uci=game.moves_uci,
        outcome=game.outcome,
        termination=game.termination,
        **meta.as_columns(),
    )


def read_raw_games(raw_dir: Path) -> Iterator[RawGameRow]:
    """Yield validated rows across all shards in ``raw_dir`` (shard order)."""
    shards = sorted(str(p) for p in raw_dir.glob(_SHARD_GLOB))
    if not shards:
        return
    data = load_dataset("parquet", data_files=shards, split="train")
    for row in data:
        yield RawGameRow.model_validate(row)


class ShardWriter:
    """Buffer games and flush atomic parquet shards with a resumable manifest."""

    def __init__(self, raw_dir: Path, meta: EngineMeta, shard_size: int) -> None:
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._meta = meta
        self._shard_size = shard_size
        self._buffer: list[dict[str, object]] = []
        self._next_index = len(list(raw_dir.glob(_SHARD_GLOB)))
        self.games_written = self._read_cursor()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        # Flush remaining games only on a clean exit so a crash doesn't write a
        # short shard that would corrupt the games_written cursor on resume.
        if exc_type is None and self._buffer:
            self._flush()

    def add(self, game_id: str, game: SelfPlayGame) -> None:
        self._buffer.append(game_to_row(game_id, game, self._meta).model_dump())
        if len(self._buffer) >= self._shard_size:
            self._flush()

    def _flush(self) -> None:
        path = self.raw_dir / f"shard_{self._next_index:0{_SHARD_INDEX_WIDTH}d}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        Dataset.from_list(self._buffer).to_parquet(str(tmp))
        os.replace(tmp, path)  # atomic publish
        self._next_index += 1
        self.games_written += len(self._buffer)
        self._buffer.clear()
        self._write_cursor()

    def _read_cursor(self) -> int:
        manifest = self.raw_dir / MANIFEST_FILENAME
        if not manifest.exists():
            return 0
        return int(json.loads(manifest.read_text(encoding="utf-8"))["games_written"])

    def _write_cursor(self) -> None:
        manifest = self.raw_dir / MANIFEST_FILENAME
        tmp = manifest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"games_written": self.games_written}) + "\n", encoding="utf-8")
        os.replace(tmp, manifest)
