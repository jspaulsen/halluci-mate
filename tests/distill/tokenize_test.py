from __future__ import annotations

from pathlib import Path  # noqa: TC003 — used at runtime (pytest tmp_path type annotation)

from datasets import Dataset

from halluci_mate.data_preparation import create_tokenizer
from halluci_mate.distill.raw_shards import RawGameRow
from halluci_mate.distill.tokenize import process_engine_game, tokenize_engine_shards


def _raw_row(game_id: str, outcome: str) -> dict:
    return {
        "game_id": game_id,
        "seed_source": "s",
        "seed_plies": 2,
        "moves_uci": ["e2e4", "e7e5", "g1f3"],
        "outcome": outcome,
        "termination": "natural",
        "engine_depth": 18,
        "engine_multipv": 4,
        "engine_wobble_cp": 30,
        "engine_sf_version": "t",
        "engine_nnue": "t",
    }


def _row(game_id: str, outcome: str) -> RawGameRow:
    return RawGameRow.model_validate(_raw_row(game_id, outcome))


def test_process_engine_game_emits_winner_perspective_sequence() -> None:
    examples = process_engine_game(_row("g0", "white"), create_tokenizer())
    assert len(examples) == 1  # decisive -> one perspective
    assert examples[0]["result"] == "white"
    assert examples[0]["opening_family"] == "e4"
    assert examples[0]["termination_type"] == "decisive"
    assert "elo_bucket" not in examples[0]
    assert len(examples[0]["input_ids"]) > 0


def test_process_engine_game_draw_emits_two_perspectives() -> None:
    assert len(process_engine_game(_row("g1", "draw"), create_tokenizer())) == 2


def test_tokenize_engine_shards_roundtrips(tmp_path: Path) -> None:
    raw_dir = tmp_path / "_raw"
    raw_dir.mkdir()
    Dataset.from_list([_raw_row(f"g{i}", "draw") for i in range(3)]).to_parquet(str(raw_dir / "shard_00000.parquet"))
    out_shards = tmp_path / "_tok"
    out_shards.mkdir()
    total = tokenize_engine_shards(raw_dir, create_tokenizer(), out_shards)
    assert total == 6  # 3 draws x 2 perspectives


def test_tokenize_engine_shards_dedups_repeated_game_ids(tmp_path: Path) -> None:
    raw_dir = tmp_path / "_raw"
    raw_dir.mkdir()
    Dataset.from_list([_raw_row("dup", "white"), _raw_row("dup", "white")]).to_parquet(str(raw_dir / "shard_00000.parquet"))
    out_shards = tmp_path / "_tok"
    out_shards.mkdir()
    assert tokenize_engine_shards(raw_dir, create_tokenizer(), out_shards) == 1  # second "dup" skipped
