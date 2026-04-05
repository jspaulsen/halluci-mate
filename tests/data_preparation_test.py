"""Tests for the data preparation pipeline."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — used at runtime via tmp_path

from datasets import Dataset

from halluci_mate.data_preparation import build_stratified_splits, create_tokenizer, process_game, strip_metadata, write_shard

SAMPLE_MOVETEXT = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6"
SHORT_MOVETEXT = "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#"


def _make_sample(
    movetext: str = SAMPLE_MOVETEXT,
    result: str = "1-0",
    white_elo: str = "1500",
    black_elo: str = "1500",
    termination: str = "Normal",
) -> dict:
    return {
        "movetext": movetext,
        "Result": result,
        "WhiteElo": white_elo,
        "BlackElo": black_elo,
        "Termination": termination,
    }


TOKENIZER = create_tokenizer()


def _make_shard_example(i: int, elo_bucket: str = "<1200", result: str = "white") -> dict:
    return {
        "input_ids": [i],
        "attention_mask": [1],
        "elo_bucket": elo_bucket,
        "result": result,
        "opening_family": "e4",
        "termination_type": "decisive",
    }


def test_process_game_returns_metadata_columns() -> None:
    examples = process_game(_make_sample(), TOKENIZER)
    assert len(examples) >= 1
    for ex in examples:
        assert "input_ids" in ex
        assert "attention_mask" in ex
        assert "elo_bucket" in ex
        assert "result" in ex
        assert "opening_family" in ex
        assert "termination_type" in ex


def test_process_game_no_min_moves_filter() -> None:
    """Short games with valid termination should NOT be filtered out."""
    examples = process_game(_make_sample(movetext=SHORT_MOVETEXT), TOKENIZER)
    assert len(examples) >= 1


def test_process_game_missing_elo_defaults_to_unknown() -> None:
    examples = process_game(_make_sample(white_elo="?"), TOKENIZER)
    assert len(examples) >= 1
    assert examples[0]["elo_bucket"] == "unknown"


def test_process_game_elo_classification() -> None:
    examples = process_game(_make_sample(white_elo="2200", black_elo="2400"), TOKENIZER)
    assert examples[0]["elo_bucket"] == "2000+"


def test_process_game_opening_family() -> None:
    examples = process_game(_make_sample(), TOKENIZER)
    assert examples[0]["opening_family"] == "e4"


def test_process_game_result_white_win() -> None:
    examples = process_game(_make_sample(result="1-0"), TOKENIZER)
    assert examples[0]["result"] == "white"


def test_process_game_draw_produces_two_examples() -> None:
    examples = process_game(_make_sample(result="1/2-1/2"), TOKENIZER)
    assert len(examples) == 2
    assert examples[0]["termination_type"] == "draw"


def test_process_game_invalid_movetext_returns_empty() -> None:
    examples = process_game(_make_sample(movetext=""), TOKENIZER)
    assert examples == []


def test_process_game_invalid_result_returns_empty() -> None:
    examples = process_game(_make_sample(result="*"), TOKENIZER)
    assert examples == []


def test_write_shard(tmp_path: Path) -> None:
    examples = [_make_shard_example(0), _make_shard_example(1, elo_bucket="2000+", result="black")]
    path = write_shard(examples, tmp_path, 0)
    assert path.exists()
    ds = Dataset.from_parquet(str(path))
    assert len(ds) == 2


def test_build_stratified_splits_correct_sizes(tmp_path: Path) -> None:
    """Eval and test sets should be exactly the requested sizes."""
    examples = [
        _make_shard_example(i, elo_bucket="<1200" if i < 200 else "2000+", result="white" if i % 2 == 0 else "black")
        for i in range(400)
    ]
    write_shard(examples, tmp_path, 0)

    eval_size = 20
    test_size = 5
    train, eval_ds, test_ds = build_stratified_splits(tmp_path, eval_size, test_size, seed=42)
    assert len(eval_ds) == eval_size
    assert len(test_ds) == test_size
    assert len(train) + len(eval_ds) + len(test_ds) == 400


def test_build_stratified_splits_no_overlap(tmp_path: Path) -> None:
    """Train, eval, and test sets should have no overlapping examples."""
    examples = [_make_shard_example(i) for i in range(200)]
    write_shard(examples, tmp_path, 0)

    train, eval_ds, test_ds = build_stratified_splits(tmp_path, 20, 5, seed=42)
    train_ids = {tuple(x) for x in train["input_ids"]}
    eval_ids = {tuple(x) for x in eval_ds["input_ids"]}
    test_ids = {tuple(x) for x in test_ds["input_ids"]}
    assert not train_ids & eval_ids
    assert not train_ids & test_ids
    assert not eval_ids & test_ids


def test_build_stratified_splits_covers_strata(tmp_path: Path) -> None:
    """Each stratum should have representation in the eval set."""
    strata = [("<1200", "white"), ("<1200", "black"), ("2000+", "white"), ("2000+", "black")]
    examples = [
        _make_shard_example(stratum_idx * 50 + j, elo_bucket=bucket, result=result)
        for stratum_idx, (bucket, result) in enumerate(strata)
        for j in range(50)
    ]
    write_shard(examples, tmp_path, 0)

    _, eval_ds, _ = build_stratified_splits(tmp_path, 40, 10, seed=42)
    eval_buckets = set(eval_ds["elo_bucket"])
    eval_results = set(eval_ds["result"])
    assert "<1200" in eval_buckets
    assert "2000+" in eval_buckets
    assert "white" in eval_results
    assert "black" in eval_results


def test_strip_metadata_removes_columns() -> None:
    ds = Dataset.from_list([
        {"input_ids": [1], "attention_mask": [1], "elo_bucket": "x", "result": "y", "opening_family": "z", "termination_type": "v"},
    ])
    stripped = strip_metadata(ds)
    assert set(stripped.column_names) == {"input_ids", "attention_mask"}
