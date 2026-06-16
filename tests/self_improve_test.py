"""Unit tests for the pure self-improvement loop logic (no GPU / Stockfish needed)."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest

from halluci_mate.eval.records import Evaluator, PerMoveRecord, Phase, Side, TopKEntry
from halluci_mate.self_improve import (
    CollapseConfig,
    RatchetConfig,
    RatchetDecision,
    RatchetMetrics,
    SelfImproveError,
    anchor_sample_count,
    diversity_collapsed,
    diversity_stats,
    draw_fraction,
    extract_ratchet_metrics,
    mix_anchor_pairs,
    ratchet_decision,
    select_best_checkpoint,
    validate_anchor_pairs,
)

if TYPE_CHECKING:
    from pathlib import Path

RATCHET = RatchetConfig(score_rate_epsilon=0.01, legal_rate_floor=0.98, legal_rate_max_regression=0.005)
COLLAPSE = CollapseConfig(min_distinct_move_ratio=0.05, entropy_floor=0.05, entropy_drop_frac=0.5)


def _move(model_move: str, top_k: list[tuple[str, float]]) -> PerMoveRecord:
    """Build a minimal PerMoveRecord; only model_move / model_top_k matter for diversity."""
    return PerMoveRecord(
        run_id="r",
        event_id=0,
        evaluator=Evaluator.VS_STOCKFISH,
        checkpoint="ckpt",
        game_id="g1",
        ply=0,
        phase=Phase.OPENING,
        side_to_move=Side.WHITE,
        model_side=Side.WHITE,
        fen_before="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        legal_moves=[model_move],
        model_move=model_move,
        model_top_k=[TopKEntry(move=m, logprob=lp) for m, lp in top_k],
        mask_used=True,
        raw_sample_move=model_move,
        raw_sample_legal=True,
        prior_opponent_move=None,
        sf_best_move=None,
        sf_eval_before_cp=None,
        sf_eval_after_cp=None,
        centipawn_loss=None,
        is_blunder=None,
    )


def _metrics_json(score_rate: float, legal_rate: float, cpl_mean: float | None) -> dict:
    payload: dict = {
        "win_rate": {"overall": {"score_rate": score_rate}},
        "legal_rate": {"overall": {"rate": legal_rate}},
    }
    if cpl_mean is not None:
        payload["centipawn_loss"] = {"overall": {"mean": cpl_mean}}
    return payload


# --- extract_ratchet_metrics ------------------------------------------------


def test_extract_ratchet_metrics_with_cpl() -> None:
    metrics = extract_ratchet_metrics(_metrics_json(0.42, 0.99, 55.0))
    assert metrics == RatchetMetrics(score_rate=0.42, legal_rate=0.99, cpl_mean=55.0)


def test_extract_ratchet_metrics_without_cpl() -> None:
    metrics = extract_ratchet_metrics(_metrics_json(0.42, 0.99, None))
    assert metrics.cpl_mean is None


def test_extract_ratchet_metrics_missing_key_raises() -> None:
    with pytest.raises(SelfImproveError):
        extract_ratchet_metrics({"legal_rate": {"overall": {"rate": 0.99}}})


# --- ratchet_decision -------------------------------------------------------


def test_ratchet_keeps_on_score_improvement() -> None:
    champion = RatchetMetrics(score_rate=0.40, legal_rate=0.99, cpl_mean=60.0)
    candidate = RatchetMetrics(score_rate=0.45, legal_rate=0.99, cpl_mean=60.0)
    assert ratchet_decision(candidate, champion, RATCHET).keep is True


def test_ratchet_reverts_on_score_regression() -> None:
    champion = RatchetMetrics(score_rate=0.45, legal_rate=0.99, cpl_mean=60.0)
    candidate = RatchetMetrics(score_rate=0.40, legal_rate=0.99, cpl_mean=60.0)
    assert ratchet_decision(candidate, champion, RATCHET).keep is False


def test_ratchet_vetoes_below_legal_floor() -> None:
    champion = RatchetMetrics(score_rate=0.40, legal_rate=0.99, cpl_mean=60.0)
    candidate = RatchetMetrics(score_rate=0.99, legal_rate=0.90, cpl_mean=10.0)
    decision = ratchet_decision(candidate, champion, RATCHET)
    assert decision == RatchetDecision(keep=False, reason=decision.reason)
    assert "floor" in decision.reason


def test_ratchet_vetoes_legal_regression() -> None:
    champion = RatchetMetrics(score_rate=0.40, legal_rate=0.995, cpl_mean=60.0)
    candidate = RatchetMetrics(score_rate=0.99, legal_rate=0.985, cpl_mean=10.0)  # above floor but regressed > 0.005
    assert ratchet_decision(candidate, champion, RATCHET).keep is False


def test_ratchet_tie_breaks_on_cpl_improvement() -> None:
    champion = RatchetMetrics(score_rate=0.40, legal_rate=0.99, cpl_mean=60.0)
    candidate = RatchetMetrics(score_rate=0.405, legal_rate=0.99, cpl_mean=50.0)  # within score epsilon, lower cpl
    assert ratchet_decision(candidate, champion, RATCHET).keep is True


def test_ratchet_tie_reverts_when_cpl_worse() -> None:
    champion = RatchetMetrics(score_rate=0.40, legal_rate=0.99, cpl_mean=50.0)
    candidate = RatchetMetrics(score_rate=0.405, legal_rate=0.99, cpl_mean=60.0)
    assert ratchet_decision(candidate, champion, RATCHET).keep is False


def test_ratchet_tie_reverts_when_cpl_missing() -> None:
    champion = RatchetMetrics(score_rate=0.40, legal_rate=0.99, cpl_mean=None)
    candidate = RatchetMetrics(score_rate=0.405, legal_rate=0.99, cpl_mean=None)
    assert ratchet_decision(candidate, champion, RATCHET).keep is False


# --- anchor_sample_count / mix_anchor_pairs ---------------------------------


def test_anchor_sample_count_half_fraction() -> None:
    assert anchor_sample_count(10, 0.5) == 10  # 0.5/(0.5) * 10


def test_anchor_sample_count_zero_disables() -> None:
    assert anchor_sample_count(10, 0.0) == 0


def test_anchor_sample_count_invalid_fraction_raises() -> None:
    with pytest.raises(SelfImproveError):
        anchor_sample_count(10, 1.0)


def test_anchor_sample_count_zero_online() -> None:
    assert anchor_sample_count(0, 0.5) == 0  # nothing to anchor against


def test_mix_anchor_zero_fraction_copies_online(tmp_path: Path) -> None:
    online = tmp_path / "online.jsonl"
    online.write_text("a\nb\nc\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    result = mix_anchor_pairs(online_path=online, anchor_path=None, fraction=0.0, seed=1, out_path=out)
    assert result.online == 3
    assert result.anchor == 0
    assert sorted(out.read_text(encoding="utf-8").split()) == ["a", "b", "c"]


def test_mix_anchor_blends_fraction(tmp_path: Path) -> None:
    online = tmp_path / "online.jsonl"
    online.write_text("\n".join(f"o{i}" for i in range(10)) + "\n", encoding="utf-8")
    anchor = tmp_path / "anchor.jsonl"
    anchor.write_text("\n".join(f"a{i}" for i in range(50)) + "\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    result = mix_anchor_pairs(online_path=online, anchor_path=anchor, fraction=0.5, seed=1, out_path=out)
    assert result.online == 10
    assert result.anchor == 10
    assert result.total == 20


def test_mix_anchor_caps_at_available(tmp_path: Path) -> None:
    online = tmp_path / "online.jsonl"
    online.write_text("\n".join(f"o{i}" for i in range(10)) + "\n", encoding="utf-8")
    anchor = tmp_path / "anchor.jsonl"
    anchor.write_text("a0\na1\n", encoding="utf-8")  # fewer than the target 10
    out = tmp_path / "out.jsonl"
    result = mix_anchor_pairs(online_path=online, anchor_path=anchor, fraction=0.5, seed=1, out_path=out)
    assert result.anchor == 2


# --- validate_anchor_pairs --------------------------------------------------

VALID_ANCHOR_LINE = '{"moves_uci":["e2e4"],"model_side":"white","chosen":"d2d4","rejected":"a2a3"}\n'


def test_validate_anchor_pairs_accepts_export_dpo_schema(tmp_path: Path) -> None:
    path = tmp_path / "anchor.jsonl"
    path.write_text(VALID_ANCHOR_LINE, encoding="utf-8")
    validate_anchor_pairs(path)  # does not raise


def test_validate_anchor_pairs_missing_keys_raises(tmp_path: Path) -> None:
    path = tmp_path / "anchor.jsonl"
    path.write_text('{"chosen":"d2d4","rejected":"a2a3"}\n', encoding="utf-8")  # no moves_uci / model_side
    with pytest.raises(SelfImproveError, match="missing export-dpo keys"):
        validate_anchor_pairs(path)


def test_validate_anchor_pairs_not_jsonl_raises(tmp_path: Path) -> None:
    path = tmp_path / "anchor.jsonl"
    path.write_text("not json at all\n", encoding="utf-8")
    with pytest.raises(SelfImproveError, match="not JSONL"):
        validate_anchor_pairs(path)


def test_validate_anchor_pairs_non_object_line_raises(tmp_path: Path) -> None:
    path = tmp_path / "anchor.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")  # valid JSON, but not a pair object
    with pytest.raises(SelfImproveError, match="must be a JSON object"):
        validate_anchor_pairs(path)


def test_validate_anchor_pairs_empty_raises(tmp_path: Path) -> None:
    path = tmp_path / "anchor.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(SelfImproveError, match="empty"):
        validate_anchor_pairs(path)


# --- diversity_stats / diversity_collapsed ----------------------------------


def test_diversity_stats_distinct_ratio_and_entropy() -> None:
    records = [
        _move("e2e4", [("e2e4", 0.0), ("d2d4", 0.0)]),  # uniform 2-way -> entropy ln2
        _move("d2d4", [("d2d4", 0.0), ("e2e4", 0.0)]),
    ]
    stats = diversity_stats(records)
    assert stats.n_moves == 2
    assert stats.distinct_move_ratio == pytest.approx(1.0)
    assert stats.mean_top1_entropy == pytest.approx(math.log(2))


def test_diversity_stats_repeated_move_low_ratio() -> None:
    records = [_move("e2e4", [("e2e4", 0.0)]) for _ in range(5)]
    stats = diversity_stats(records)
    assert stats.distinct_move_ratio == pytest.approx(0.2)
    assert stats.mean_top1_entropy == pytest.approx(0.0)  # single-entry top-k -> 0


def test_diversity_stats_empty() -> None:
    stats = diversity_stats([])
    assert stats.n_moves == 0
    assert stats.distinct_move_ratio == 0.0


def test_diversity_collapsed_distinct_floor() -> None:
    current = diversity_stats([_move("e2e4", [("e2e4", 0.0)]) for _ in range(100)])
    baseline = diversity_stats([_move(f"m{i}", [("a", 0.0), ("b", -1.0)]) for i in range(100)])
    collapsed, reason = diversity_collapsed(current, baseline, COLLAPSE)
    assert collapsed is True
    assert "distinct_move_ratio" in reason


def test_diversity_collapsed_entropy_drop() -> None:
    baseline = diversity_stats([_move(f"m{i}", [("a", 0.0), ("b", 0.0)]) for i in range(10)])  # high entropy
    current = diversity_stats([_move(f"m{i}", [("a", 0.0), ("b", -5.0)]) for i in range(10)])  # peaked -> low entropy
    collapsed, reason = diversity_collapsed(current, baseline, COLLAPSE)
    assert collapsed is True
    assert "entropy" in reason


def test_diversity_within_bounds() -> None:
    stats = diversity_stats([_move(f"m{i}", [("a", 0.0), ("b", 0.0)]) for i in range(10)])
    collapsed, _ = diversity_collapsed(stats, stats, COLLAPSE)
    assert collapsed is False


# --- select_best_checkpoint (the v2c-lesson fix) -----------------------------


def _make_checkpoints(tmp_path: Path, steps: list[int], accuracies: dict[int, float] | None) -> None:
    """Create checkpoint dirs; write the full eval log into the highest-step trainer_state.json."""
    run = tmp_path / "mlflow-run"
    for step in steps:
        (run / f"checkpoint-{step}").mkdir(parents=True)
    log_history = [] if accuracies is None else [{"step": s, "eval_rewards/accuracies": a} for s, a in accuracies.items()]
    latest = run / f"checkpoint-{max(steps)}"
    (latest / "trainer_state.json").write_text(json.dumps({"log_history": log_history}), encoding="utf-8")


def test_select_best_checkpoint_prefers_best_accuracy_not_last(tmp_path: Path) -> None:
    # Final step (300) is NOT the best — step 200 has the highest eval accuracy.
    _make_checkpoints(tmp_path, [100, 200, 300], {100: 0.6, 200: 0.8, 300: 0.7})
    assert select_best_checkpoint(tmp_path).name == "checkpoint-200"


def test_select_best_checkpoint_falls_back_to_last_without_eval(tmp_path: Path) -> None:
    _make_checkpoints(tmp_path, [100, 200, 300], accuracies=None)  # no eval metrics logged
    assert select_best_checkpoint(tmp_path).name == "checkpoint-300"


def test_select_best_checkpoint_tie_breaks_on_earlier_step(tmp_path: Path) -> None:
    # Steps 100 and 300 tie on accuracy; the earlier checkpoint wins.
    _make_checkpoints(tmp_path, [100, 200, 300], {100: 0.8, 200: 0.7, 300: 0.8})
    assert select_best_checkpoint(tmp_path).name == "checkpoint-100"


def test_select_best_checkpoint_none_raises(tmp_path: Path) -> None:
    with pytest.raises(SelfImproveError):
        select_best_checkpoint(tmp_path)


def test_select_best_checkpoint_unparseable_raises(tmp_path: Path) -> None:
    (tmp_path / "run" / "checkpoint-final").mkdir(parents=True)
    with pytest.raises(SelfImproveError):
        select_best_checkpoint(tmp_path)


# --- draw_fraction / empty-run guard ----------------------------------------


def test_draw_fraction_excludes_unfinished() -> None:
    overall = {"games": 12, "wins": 4, "losses": 3, "draws": 3, "unfinished": 2}  # scored = 10
    assert draw_fraction(overall) == pytest.approx(0.3)


def test_draw_fraction_no_scored_games() -> None:
    assert draw_fraction({"games": 2, "wins": 0, "losses": 0, "draws": 0, "unfinished": 2}) == 0.0


def test_diversity_collapsed_empty_run_is_not_collapse() -> None:
    empty = diversity_stats([])
    baseline = diversity_stats([_move(f"m{i}", [("a", 0.0), ("b", 0.0)]) for i in range(10)])
    collapsed, reason = diversity_collapsed(empty, baseline, COLLAPSE)
    assert collapsed is False
    assert "empty generation" in reason
