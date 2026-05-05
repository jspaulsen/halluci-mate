"""Tests for `halluci_mate.eval.metrics`.

Driven by synthetic record fixtures (not real eval runs), per the HAL-7
acceptance criteria.
"""

from __future__ import annotations

import json
import math

import pytest

from halluci_mate.eval.metrics import (
    LegalRateBucket,
    WinRateBucket,
    WinRateStats,
    compute_all,
    compute_legal_rate,
    compute_win_rate,
)
from halluci_mate.eval.records import Evaluator, Phase, Record, Side
from tests.eval.conftest import (
    make_per_game_record,
    make_per_legal_rate_record,
    make_per_move_record,
    make_per_perplexity_record,
)

DEFAULT_CONFIG: dict[str, object] = {
    "evaluator": Evaluator.VS_STOCKFISH.value,
    "stockfish_skill": 5,
}


def _move(event_id: int, **overrides: object) -> Record:
    return make_per_move_record(event_id=event_id, **overrides)


def _game(event_id: int, **overrides: object) -> Record:
    return make_per_game_record(event_id=event_id, **overrides)


def test_compute_legal_rate_counts_raw_sample_legal() -> None:
    records = [
        _move(0, raw_sample_legal=True),
        _move(1, raw_sample_legal=True),
        _move(2, raw_sample_legal=False),
        _move(3, raw_sample_legal=True),
    ]
    assert compute_legal_rate(records) == pytest.approx(0.75)


def test_compute_legal_rate_returns_zero_with_no_records() -> None:
    assert compute_legal_rate([]) == 0.0


def test_compute_legal_rate_ignores_non_per_move_records() -> None:
    records: list[Record] = [
        _move(0, raw_sample_legal=False),
        _game(1, result="1-0"),
    ]
    assert compute_legal_rate(records) == 0.0


def test_compute_win_rate_classifies_outcomes() -> None:
    """`1-0` as White and `0-1` as Black are wins; `*` is unfinished."""
    records = [
        _game(0, game_id="g0", model_side=Side.WHITE, result="1-0"),
        _game(1, game_id="g1", model_side=Side.WHITE, result="0-1"),
        _game(2, game_id="g2", model_side=Side.BLACK, result="0-1"),
        _game(3, game_id="g3", model_side=Side.BLACK, result="1/2-1/2"),
        _game(4, game_id="g4", model_side=Side.WHITE, result="*"),
    ]
    stats = compute_win_rate(records)
    assert isinstance(stats, WinRateStats)

    overall = stats.overall
    assert overall.games == 5
    assert overall.wins == 2
    assert overall.losses == 1
    assert overall.draws == 1
    assert overall.unfinished == 1
    # Scored = 4 (one unfinished). Wins=2 → 0.5; score = (2 + 0.5*1)/4 = 0.625.
    assert overall.win_rate == pytest.approx(0.5)
    assert overall.score_rate == pytest.approx(0.625)


def test_compute_win_rate_stratifies_by_model_side() -> None:
    records = [
        _game(0, game_id="g0", model_side=Side.WHITE, result="1-0"),
        _game(1, game_id="g1", model_side=Side.WHITE, result="0-1"),
        _game(2, game_id="g2", model_side=Side.BLACK, result="0-1"),
        _game(3, game_id="g3", model_side=Side.BLACK, result="1/2-1/2"),
    ]
    stats = compute_win_rate(records)
    white = stats.by_model_side["white"]
    black = stats.by_model_side["black"]
    assert white.games == 2
    assert white.wins == 1
    assert white.losses == 1
    assert black.games == 2
    assert black.wins == 1
    assert black.draws == 1
    assert black.win_rate == pytest.approx(0.5)
    assert black.score_rate == pytest.approx(0.75)


def test_compute_win_rate_empty_buckets_have_zero_rates() -> None:
    """No records of a given side ⇒ both rates collapse to 0.0 (games=0 makes the case unambiguous)."""
    stats = compute_win_rate([])
    assert stats.overall == WinRateBucket(games=0, wins=0, losses=0, draws=0, unfinished=0, win_rate=0.0, score_rate=0.0)
    assert stats.by_model_side["white"].games == 0
    assert stats.by_model_side["black"].games == 0


def test_compute_all_dispatches_on_evaluator() -> None:
    records: list[Record] = [
        _move(0, model_side=Side.WHITE, phase=Phase.OPENING, raw_sample_legal=True),
        _move(1, model_side=Side.WHITE, phase=Phase.OPENING, raw_sample_legal=False),
        _move(2, model_side=Side.BLACK, phase=Phase.MIDDLE, raw_sample_legal=True),
        _game(3, game_id="g0", model_side=Side.WHITE, result="1-0"),
        _game(4, game_id="g1", model_side=Side.BLACK, result="1/2-1/2"),
    ]
    metrics = compute_all(records, DEFAULT_CONFIG)

    assert metrics["evaluator"] == Evaluator.VS_STOCKFISH.value
    assert metrics["stockfish_skill"] == 5

    assert metrics["win_rate"]["overall"]["games"] == 2
    assert metrics["win_rate"]["overall"]["wins"] == 1
    assert metrics["win_rate"]["overall"]["draws"] == 1
    assert metrics["win_rate"]["by_model_side"]["white"]["wins"] == 1
    assert metrics["win_rate"]["by_model_side"]["black"]["draws"] == 1

    legal = metrics["legal_rate"]
    assert legal["overall"] == {"n": 3, "legal": 2, "rate": pytest.approx(2 / 3)}
    assert legal["by_phase"]["opening"] == {"n": 2, "legal": 1, "rate": 0.5}
    assert legal["by_phase"]["middle"]["rate"] == pytest.approx(1.0)
    assert legal["by_phase"]["endgame"] == {"n": 0, "legal": 0, "rate": 0.0}
    assert legal["by_model_side"]["white"]["rate"] == pytest.approx(0.5)
    assert legal["by_model_side"]["black"]["rate"] == pytest.approx(1.0)


def test_compute_all_output_is_json_serializable() -> None:
    """The aggregate dict is what gets written to `metrics.json` — round-trip
    through json to pin that the schema only contains primitives."""
    records: list[Record] = [_move(0), _game(1, result="1-0")]
    metrics = compute_all(records, DEFAULT_CONFIG)
    encoded = json.dumps(metrics)
    decoded = json.loads(encoded)
    assert decoded["evaluator"] == Evaluator.VS_STOCKFISH.value


def test_compute_all_schema_is_stable_for_diffing() -> None:
    """Same evaluator + same record shapes ⇒ same top-level + nested key set,
    even when the underlying counts differ. This is the "diff across runs"
    contract from HAL-7's acceptance criteria.
    """
    records_a: list[Record] = [_move(0), _game(1, result="1-0")]
    records_b: list[Record] = [
        _move(0, raw_sample_legal=False),
        _move(1, raw_sample_legal=False),
        _game(2, result="0-1"),
    ]
    keys_a = _structure_keys(compute_all(records_a, DEFAULT_CONFIG))
    keys_b = _structure_keys(compute_all(records_b, DEFAULT_CONFIG))
    assert keys_a == keys_b


def test_compute_all_rejects_unknown_evaluator() -> None:
    with pytest.raises(ValueError, match="unsupported evaluator"):
        compute_all([], {"evaluator": "puzzles"})


def test_compute_legal_rate_counts_per_legal_rate_records() -> None:
    """The same metric works over `PerLegalRateRecord.legal` for `legal_rate` runs."""
    records: list[Record] = [
        make_per_legal_rate_record(0, legal=True),
        make_per_legal_rate_record(1, legal=False),
        make_per_legal_rate_record(2, legal=True),
        make_per_legal_rate_record(3, legal=True),
    ]
    assert compute_legal_rate(records) == pytest.approx(0.75)


def test_compute_legal_rate_combines_record_shapes() -> None:
    """Mixed records: per-move + per-legal-rate share the same denominator."""
    records: list[Record] = [
        make_per_move_record(0, raw_sample_legal=True),
        make_per_move_record(1, raw_sample_legal=False),
        make_per_legal_rate_record(2, legal=True),
        make_per_legal_rate_record(3, legal=False),
    ]
    assert compute_legal_rate(records) == pytest.approx(0.5)


def test_compute_all_legal_rate_evaluator() -> None:
    records: list[Record] = [
        make_per_legal_rate_record(0, legal=True),
        make_per_legal_rate_record(1, legal=False),
        make_per_legal_rate_record(2, legal=True),
    ]
    config = {"evaluator": Evaluator.LEGAL_RATE.value}
    metrics = compute_all(records, config)

    assert metrics["evaluator"] == Evaluator.LEGAL_RATE.value
    assert metrics["legal_rate"] == {"overall": {"n": 3, "legal": 2, "rate": pytest.approx(2 / 3)}}


def test_compute_all_legal_rate_handles_empty_records() -> None:
    metrics = compute_all([], {"evaluator": Evaluator.LEGAL_RATE.value})
    assert metrics["legal_rate"] == {"overall": {"n": 0, "legal": 0, "rate": 0.0}}


def test_compute_all_perplexity_aggregates_token_logprobs() -> None:
    """Mean NLL is `-mean(logprobs)`; bits/token is NLL / ln 2."""
    records: list[Record] = [
        make_per_perplexity_record(0, token_logprobs=[-1.0, -2.0]),
        make_per_perplexity_record(1, token_logprobs=[-3.0, -4.0]),
    ]
    config = {"evaluator": Evaluator.PERPLEXITY.value}
    metrics = compute_all(records, config)

    expected_mean_nll = (1.0 + 2.0 + 3.0 + 4.0) / 4
    assert metrics["evaluator"] == Evaluator.PERPLEXITY.value
    assert metrics["num_sequences"] == 2
    assert metrics["num_tokens"] == 4
    assert metrics["mean_nll"] == pytest.approx(expected_mean_nll)
    assert metrics["bits_per_token"] == pytest.approx(expected_mean_nll / math.log(2))
    assert metrics["perplexity"] == pytest.approx(math.exp(expected_mean_nll))


def test_compute_all_perplexity_handles_empty_records() -> None:
    metrics = compute_all([], {"evaluator": Evaluator.PERPLEXITY.value})
    assert metrics["num_sequences"] == 0
    assert metrics["num_tokens"] == 0
    assert metrics["mean_nll"] == 0.0
    assert metrics["bits_per_token"] == 0.0
    assert metrics["perplexity"] == 0.0


def test_compute_all_legal_rate_output_json_serializable() -> None:
    metrics = compute_all([make_per_legal_rate_record(0)], {"evaluator": Evaluator.LEGAL_RATE.value})
    json.loads(json.dumps(metrics))


def test_compute_all_perplexity_output_json_serializable() -> None:
    metrics = compute_all([make_per_perplexity_record(0)], {"evaluator": Evaluator.PERPLEXITY.value})
    json.loads(json.dumps(metrics))


def test_legal_rate_bucket_rate_is_zero_when_empty() -> None:
    """Pinning the empty-bucket convention so it is not silently changed."""
    bucket = LegalRateBucket(n=0, legal=0, rate=0.0)
    assert bucket.rate == 0.0


def _structure_keys(value: object) -> object:
    """Reduce a metrics dict to its key skeleton for diff-stability checks."""
    if isinstance(value, dict):
        return {key: _structure_keys(child) for key, child in value.items()}
    return None
