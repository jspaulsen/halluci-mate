"""Tests for the ``perplexity`` evaluator.

Exercises the full path from a jsonl input file through tokenization,
scoring, and record-emission. Uses a stub model whose logits are
crafted so the expected logprobs are fixed and easy to assert.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
import torch

from halluci_mate.chess_tokenizer import ChessTokenizer
from halluci_mate.eval.evaluators.perplexity import (
    PerplexityConfig,
    run_perplexity,
)
from halluci_mate.eval.records import Evaluator, PerPerplexityRecord
from halluci_mate.eval.runs import CONFIG_FILENAME, RECORDS_FILENAME, RunReader
from tests.eval.conftest import DEFAULT_CHECKPOINT, DEFAULT_RUN_ID

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class _ModelOutput:
    logits: torch.Tensor


class _UniformModel:
    """Returns log-uniform logits over the full vocabulary at every position.

    Under uniform logits, every target token has logprob = -log(vocab_size).
    Crafted this way so test expectations are deterministic without depending
    on specific token IDs.
    """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.calls = 0
        self.last_input_shape: tuple[int, ...] | None = None

    def __call__(self, *, input_ids: torch.Tensor) -> _ModelOutput:
        self.calls += 1
        self.last_input_shape = tuple(input_ids.shape)
        batch, seq_len = input_ids.shape
        # Uniform logits → log_softmax = -log(vocab) for every token.
        logits = torch.zeros((batch, seq_len, self.vocab_size))
        return _ModelOutput(logits=logits)


class _StubScorer:
    """Wraps a model + tokenizer + device for the ``PerplexityScorer`` Protocol."""

    def __init__(self, model: _UniformModel, tokenizer: ChessTokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cpu")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _read_perplexity_records(run_dir: Path) -> list[PerPerplexityRecord]:
    return [r for r in RunReader(run_dir).read_records() if isinstance(r, PerPerplexityRecord)]


def test_emits_one_record_per_sequence(tmp_path: Path) -> None:
    data = tmp_path / "sequences.jsonl"
    _write_jsonl(
        data,
        [
            {"id": "g1", "perspective": "white", "moves": ["e2e4", "e7e5"]},
            {"id": "g2", "perspective": "black", "moves": ["d2d4", "d7d5", "c2c4"], "is_draw": True},
        ],
    )

    tokenizer = ChessTokenizer()
    model = _UniformModel(vocab_size=tokenizer.vocab_size)
    run_dir = tmp_path / "run"

    n = run_perplexity(
        engine=_StubScorer(model, tokenizer),
        config=PerplexityConfig(data_path=data),
        run_dir=run_dir,
        run_id=DEFAULT_RUN_ID,
        checkpoint=DEFAULT_CHECKPOINT,
    )

    assert n == 2
    assert (run_dir / CONFIG_FILENAME).exists()
    assert (run_dir / RECORDS_FILENAME).exists()

    records = _read_perplexity_records(run_dir)
    assert [r.event_id for r in records] == [0, 1]
    assert [r.position_id for r in records] == ["g1", "g2"]
    assert all(r.evaluator == Evaluator.PERPLEXITY for r in records)
    assert all(r.run_id == DEFAULT_RUN_ID for r in records)
    assert all(r.checkpoint == DEFAULT_CHECKPOINT for r in records)

    # First sequence: <WHITE> e2e4 e7e5 — 3 tokens → 2 logprobs.
    assert len(records[0].token_logprobs) == 2
    # Second sequence with is_draw=True: <BLACK> d2d4 d7d5 c2c4 <DRAW> — 5 tokens → 4 logprobs.
    assert len(records[1].token_logprobs) == 4

    # Uniform model → every token logprob equals -log(vocab_size).
    expected = -math.log(tokenizer.vocab_size)
    for r in records:
        for lp in r.token_logprobs:
            assert lp == pytest.approx(expected, rel=1e-6)


def test_max_sequences_caps_iteration(tmp_path: Path) -> None:
    data = tmp_path / "sequences.jsonl"
    _write_jsonl(
        data,
        [{"id": f"g{i}", "perspective": "white", "moves": ["e2e4", "e7e5"]} for i in range(10)],
    )

    tokenizer = ChessTokenizer()
    model = _UniformModel(vocab_size=tokenizer.vocab_size)
    run_dir = tmp_path / "run"

    n = run_perplexity(
        engine=_StubScorer(model, tokenizer),
        config=PerplexityConfig(data_path=data, max_sequences=3),
        run_dir=run_dir,
        run_id=DEFAULT_RUN_ID,
        checkpoint=DEFAULT_CHECKPOINT,
    )

    assert n == 3
    records = _read_perplexity_records(run_dir)
    assert [r.position_id for r in records] == ["g0", "g1", "g2"]
    # The model must not be invoked beyond the cap.
    assert model.calls == 3


def test_skips_blank_lines(tmp_path: Path) -> None:
    data = tmp_path / "sequences.jsonl"
    rows_text = (
        "\n"  # leading blank
        + json.dumps({"id": "g1", "perspective": "white", "moves": ["e2e4"]})
        + "\n\n"
        + json.dumps({"id": "g2", "perspective": "white", "moves": ["d2d4"]})
        + "\n"
    )
    data.write_text(rows_text, encoding="utf-8")

    tokenizer = ChessTokenizer()
    run_dir = tmp_path / "run"

    n = run_perplexity(
        engine=_StubScorer(_UniformModel(vocab_size=tokenizer.vocab_size), tokenizer),
        config=PerplexityConfig(data_path=data),
        run_dir=run_dir,
        run_id=DEFAULT_RUN_ID,
        checkpoint=DEFAULT_CHECKPOINT,
    )

    assert n == 2


def test_rejects_unknown_perspective(tmp_path: Path) -> None:
    data = tmp_path / "sequences.jsonl"
    _write_jsonl(data, [{"id": "g1", "perspective": "draw", "moves": ["e2e4"]}])
    tokenizer = ChessTokenizer()

    with pytest.raises(ValueError, match="perspective"):
        run_perplexity(
            engine=_StubScorer(_UniformModel(vocab_size=tokenizer.vocab_size), tokenizer),
            config=PerplexityConfig(data_path=data),
            run_dir=tmp_path / "run",
            run_id=DEFAULT_RUN_ID,
            checkpoint=DEFAULT_CHECKPOINT,
        )


def test_rejects_missing_required_fields(tmp_path: Path) -> None:
    data = tmp_path / "sequences.jsonl"
    _write_jsonl(data, [{"id": "g1", "moves": ["e2e4"]}])  # no "perspective"
    tokenizer = ChessTokenizer()

    with pytest.raises(ValueError, match="perspective"):
        run_perplexity(
            engine=_StubScorer(_UniformModel(vocab_size=tokenizer.vocab_size), tokenizer),
            config=PerplexityConfig(data_path=data),
            run_dir=tmp_path / "run",
            run_id=DEFAULT_RUN_ID,
            checkpoint=DEFAULT_CHECKPOINT,
        )


def test_config_payload_round_trips_data_path(tmp_path: Path) -> None:
    data = tmp_path / "sequences.jsonl"
    _write_jsonl(data, [{"id": "g1", "perspective": "white", "moves": ["e2e4"]}])
    tokenizer = ChessTokenizer()
    run_dir = tmp_path / "run"

    run_perplexity(
        engine=_StubScorer(_UniformModel(vocab_size=tokenizer.vocab_size), tokenizer),
        config=PerplexityConfig(data_path=data, max_sequences=1),
        run_dir=run_dir,
        run_id=DEFAULT_RUN_ID,
        checkpoint=DEFAULT_CHECKPOINT,
    )

    payload = RunReader(run_dir).read_config()
    assert payload["evaluator"] == Evaluator.PERPLEXITY.value
    assert payload["data_path"] == str(data)
    assert payload["max_sequences"] == 1


def test_config_rejects_zero_max_sequences(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_sequences"):
        PerplexityConfig(data_path=tmp_path / "x.jsonl", max_sequences=0)


def test_rejects_non_dict_jsonl_row(tmp_path: Path) -> None:
    """A jsonl line that decodes to a non-object (list/scalar) must fail with a clear message."""
    data = tmp_path / "sequences.jsonl"
    data.write_text("[1, 2, 3]\n", encoding="utf-8")
    tokenizer = ChessTokenizer()

    with pytest.raises(ValueError, match="must be a JSON object"):
        run_perplexity(
            engine=_StubScorer(_UniformModel(vocab_size=tokenizer.vocab_size), tokenizer),
            config=PerplexityConfig(data_path=data),
            run_dir=tmp_path / "run",
            run_id=DEFAULT_RUN_ID,
            checkpoint=DEFAULT_CHECKPOINT,
        )


def test_rejects_move_not_in_vocab(tmp_path: Path) -> None:
    """A UCI string outside the tokenizer vocabulary must fail with the offending row id."""
    data = tmp_path / "sequences.jsonl"
    _write_jsonl(data, [{"id": "g1", "perspective": "white", "moves": ["e2e4", "z9z9"]}])
    tokenizer = ChessTokenizer()

    with pytest.raises(ValueError, match=r"g1.*z9z9"):
        run_perplexity(
            engine=_StubScorer(_UniformModel(vocab_size=tokenizer.vocab_size), tokenizer),
            config=PerplexityConfig(data_path=data),
            run_dir=tmp_path / "run",
            run_id=DEFAULT_RUN_ID,
            checkpoint=DEFAULT_CHECKPOINT,
        )
