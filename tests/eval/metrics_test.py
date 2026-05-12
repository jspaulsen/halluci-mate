"""Tests for `halluci_mate.eval.metrics`.

Driven by synthetic record fixtures (not real eval runs), per the HAL-7
acceptance criteria.
"""

from __future__ import annotations

import json
import math

import pytest

from halluci_mate.eval.metrics import (
    BlunderBucket,
    CplBucket,
    LegalRateBucket,
    WinRateBucket,
    WinRateStats,
    compute_all,
    compute_blunder_rate,
    compute_centipawn_loss,
    compute_legal_rate,
    compute_win_rate,
)
from halluci_mate.eval.records import Evaluator, PerGameRecord, PerMoveRecord, Phase, Record, Side
from tests.helpers.eval_records import (
    make_per_game_record,
    make_per_legal_rate_record,
    make_per_move_record,
    make_per_perplexity_record,
)

DEFAULT_CONFIG: dict[str, object] = {
    "evaluator": Evaluator.VS_STOCKFISH.value,
    "stockfish_skill": 5,
}


def _move(event_id: int, **overrides: object) -> PerMoveRecord:
    return make_per_move_record(event_id=event_id, **overrides)


def _game(event_id: int, **overrides: object) -> PerGameRecord:
    return make_per_game_record(event_id=event_id, **overrides)


def test_compute_legal_rate_counts_raw_sample_legal() -> None:
    moves = [
        _move(0, raw_sample_legal=True),
        _move(1, raw_sample_legal=True),
        _move(2, raw_sample_legal=False),
        _move(3, raw_sample_legal=True),
    ]
    stats = compute_legal_rate(moves)
    assert stats.overall == LegalRateBucket(n=4, legal=3, rate=0.75)


def test_compute_legal_rate_returns_zero_buckets_with_no_records() -> None:
    stats = compute_legal_rate([])
    zero = LegalRateBucket(n=0, legal=0, rate=0.0)
    assert stats.overall == zero
    assert stats.by_phase == {phase.value: zero for phase in Phase}
    assert stats.by_model_side == {side.value: zero for side in Side}


def test_compute_legal_rate_stratifies_by_phase_and_side() -> None:
    moves = [
        _move(0, model_side=Side.WHITE, phase=Phase.OPENING, raw_sample_legal=True),
        _move(1, model_side=Side.WHITE, phase=Phase.OPENING, raw_sample_legal=False),
        _move(2, model_side=Side.BLACK, phase=Phase.MIDDLE, raw_sample_legal=True),
    ]
    stats = compute_legal_rate(moves)
    assert stats.by_phase["opening"] == LegalRateBucket(n=2, legal=1, rate=0.5)
    assert stats.by_phase["middle"] == LegalRateBucket(n=1, legal=1, rate=1.0)
    assert stats.by_phase["endgame"] == LegalRateBucket(n=0, legal=0, rate=0.0)
    assert stats.by_model_side["white"].rate == pytest.approx(0.5)
    assert stats.by_model_side["black"].rate == pytest.approx(1.0)


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


def test_compute_all_rejects_unknown_evaluator() -> None:
    with pytest.raises(ValueError, match="unsupported evaluator"):
        compute_all([], {"evaluator": "puzzles"})


def test_compute_all_returns_zero_buckets_with_no_records() -> None:
    """Empty-records input yields the same shape as a populated run, with
    zeroed counts and rates. Pinned so a downstream `metrics.json` diff
    against an empty-vs-populated run does not blow up on missing keys."""
    metrics = compute_all([], DEFAULT_CONFIG)

    assert metrics["evaluator"] == Evaluator.VS_STOCKFISH.value
    assert metrics["stockfish_skill"] == 5

    overall_win = metrics["win_rate"]["overall"]
    assert overall_win["games"] == 0
    assert overall_win["win_rate"] == 0.0
    assert overall_win["score_rate"] == 0.0
    assert metrics["win_rate"]["by_model_side"]["white"]["games"] == 0
    assert metrics["win_rate"]["by_model_side"]["black"]["games"] == 0

    legal = metrics["legal_rate"]
    assert legal["overall"] == {"n": 0, "legal": 0, "rate": 0.0}
    assert legal["by_phase"]["opening"] == {"n": 0, "legal": 0, "rate": 0.0}
    assert legal["by_model_side"]["white"] == {"n": 0, "legal": 0, "rate": 0.0}


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


def test_compute_centipawn_loss_overall_and_by_phase() -> None:
    moves = [
        _move(0, phase=Phase.OPENING, centipawn_loss=10, is_blunder=False),
        _move(1, phase=Phase.OPENING, centipawn_loss=20, is_blunder=False),
        _move(2, phase=Phase.MIDDLE, centipawn_loss=300, is_blunder=True),
        _move(3, phase=Phase.MIDDLE, centipawn_loss=500, is_blunder=True),
        _move(4, phase=Phase.ENDGAME, centipawn_loss=50, is_blunder=False),
    ]
    stats = compute_centipawn_loss(moves)
    assert stats.overall.n == 5
    assert stats.overall.mean == pytest.approx((10 + 20 + 300 + 500 + 50) / 5)
    assert stats.overall.median == pytest.approx(50.0)
    assert stats.by_phase["opening"].n == 2
    assert stats.by_phase["opening"].mean == pytest.approx(15.0)
    assert stats.by_phase["middle"].n == 2
    assert stats.by_phase["middle"].mean == pytest.approx(400.0)
    assert stats.by_phase["endgame"].n == 1
    # n=1 collapse: p95 == the sole sample (statistics.quantiles needs >=2).
    assert stats.by_phase["endgame"].p95 == pytest.approx(50.0)


def test_compute_centipawn_loss_skips_none_records() -> None:
    moves = [
        _move(0, centipawn_loss=None, is_blunder=None),
        _move(1, centipawn_loss=100, is_blunder=False),
    ]
    stats = compute_centipawn_loss(moves)
    assert stats.overall.n == 1
    assert stats.overall.mean == pytest.approx(100.0)


def test_compute_centipawn_loss_empty_returns_zero_buckets() -> None:
    stats = compute_centipawn_loss([])
    zero = CplBucket(n=0, mean=0.0, median=0.0, p95=0.0)
    assert stats.overall == zero
    assert stats.by_phase == {phase.value: zero for phase in Phase}


def test_compute_blunder_rate_overall_and_by_phase() -> None:
    moves = [
        _move(0, phase=Phase.OPENING, centipawn_loss=10, is_blunder=False),
        _move(1, phase=Phase.OPENING, centipawn_loss=300, is_blunder=True),
        _move(2, phase=Phase.MIDDLE, centipawn_loss=500, is_blunder=True),
        _move(3, phase=Phase.MIDDLE, centipawn_loss=50, is_blunder=False),
    ]
    stats = compute_blunder_rate(moves)
    assert stats.overall == BlunderBucket(n=4, blunders=2, rate=0.5)
    assert stats.by_phase["opening"] == BlunderBucket(n=2, blunders=1, rate=0.5)
    assert stats.by_phase["middle"] == BlunderBucket(n=2, blunders=1, rate=0.5)
    assert stats.by_phase["endgame"] == BlunderBucket(n=0, blunders=0, rate=0.0)


def test_compute_blunder_rate_empty_returns_zero_buckets() -> None:
    stats = compute_blunder_rate([])
    zero = BlunderBucket(n=0, blunders=0, rate=0.0)
    assert stats.overall == zero
    assert stats.by_phase == {phase.value: zero for phase in Phase}


def test_compute_all_vs_stockfish_emits_cpl_and_blunder_when_present() -> None:
    """`--sf-analyze`-on records make the two new top-level keys appear."""
    records: list[Record] = [
        _move(0, centipawn_loss=10, is_blunder=False),
        _move(1, centipawn_loss=400, is_blunder=True),
        _game(2, result="1-0"),
    ]
    metrics = compute_all(records, DEFAULT_CONFIG)
    assert "centipawn_loss" in metrics
    assert "blunder_rate" in metrics
    assert metrics["centipawn_loss"]["overall"]["n"] == 2
    assert metrics["blunder_rate"]["overall"] == {"n": 2, "blunders": 1, "rate": pytest.approx(0.5)}


def test_compute_all_vs_stockfish_omits_cpl_and_blunder_when_absent() -> None:
    """`--sf-analyze`-off runs (all-None sf fields) must not emit the new blocks.

    Pinned so `metrics.json` diffs between analyze-on and analyze-off runs
    stay structurally honest — an all-zeros block would falsely suggest the
    analysis ran.
    """
    records: list[Record] = [_move(0), _game(1, result="1-0")]
    metrics = compute_all(records, DEFAULT_CONFIG)
    assert "centipawn_loss" not in metrics
    assert "blunder_rate" not in metrics
