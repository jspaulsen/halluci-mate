"""Pure logic for the autonomous self-improvement loop (Stage A).

The orchestrator in ``scripts/self_improve.py`` closes the loop:
generate (vs-stockfish) -> select (export-dpo) -> anchor -> fine-tune
(train_dpo) -> evaluate on a fixed held-out battery -> keep-or-revert.

This module holds the GPU/Stockfish/subprocess-free pieces that decision-gate
and shape data for that loop, so they can be unit-tested without a GPU,
Stockfish, or a checkpoint. The decision functions are pure; the data-shaping
ones (``mix_anchor_pairs``, ``validate_anchor_pairs``, ``select_best_checkpoint``,
``write_json``) read/write JSONL or checkpoint metadata — deterministic
transforms over the filesystem as data, not effectful orchestration:

* ``extract_ratchet_metrics`` — pull the three ratchet signals out of a
  ``metrics.json`` payload.
* ``ratchet_decision`` — the keep-or-revert rule (the loop's keystone).
* ``mix_anchor_pairs`` — blend a fraction of fixed anchor pairs into the
  online pairs to resist model collapse / MAD.
* ``diversity_stats`` / ``diversity_collapsed`` — read move diversity off a
  generation run's records to detect mode collapse.
* ``select_best_checkpoint`` — pick the behaviorally-best checkpoint under
  ``train_dpo``'s non-deterministic MLflow-named output directory.
* ``validate_anchor_pairs`` — fail fast on a malformed anchor file before a
  full generate+train cycle is spent.

The held-out battery is measured *off* the data-generation signal (a
different, fixed Stockfish skill/depth, greedy decoding), so gains that do
not generalize cannot pass the ratchet — see ``docs`` / the loop plan.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from halluci_mate.eval.records import PerMoveRecord

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from halluci_mate.eval.records import Record

# train_dpo writes checkpoints to ``<output_dir>/<mlflow-run-name>/checkpoint-<step>``;
# the run name is non-deterministic, so we glob one level down for the step dirs.
CHECKPOINT_GLOB = "*/checkpoint-*"
CHECKPOINT_PREFIX = "checkpoint-"
TRAINER_STATE_FILENAME = "trainer_state.json"
# Behavioral metric train_dpo logs to trainer_state.json; the v2c lesson is
# that this (not eval_loss) tracks vs-stockfish strength, so we select on it.
EVAL_ACCURACY_KEY = "eval_rewards/accuracies"
# Keys train_dpo's _load_pairs requires per anchor JSONL line; validated up
# front so a malformed anchor file fails before a full generate+train cycle.
ANCHOR_REQUIRED_KEYS = ("moves_uci", "model_side", "chosen", "rejected")


class SelfImproveError(RuntimeError):
    """Raised when the self-improvement loop cannot proceed (bad inputs, missing artifacts)."""


@dataclass(frozen=True)
class RatchetMetrics:
    """The three held-out signals the ratchet gates on.

    ``cpl_mean`` is ``None`` when the eval run did not populate centipawn
    loss (``--sf-analyze`` off); the held-out battery always runs with
    analysis on, so in practice it is present and used only as the tie-break.
    """

    score_rate: float
    legal_rate: float
    cpl_mean: float | None


@dataclass(frozen=True)
class RatchetConfig:
    """Thresholds for ``ratchet_decision``."""

    score_rate_epsilon: float
    legal_rate_floor: float
    legal_rate_max_regression: float


@dataclass(frozen=True)
class RatchetDecision:
    """Outcome of one ratchet comparison: keep the candidate or revert to the champion."""

    keep: bool
    reason: str


@dataclass(frozen=True)
class AnchorMixResult:
    """Line counts from ``mix_anchor_pairs`` (online + sampled anchor = total written)."""

    online: int
    anchor: int
    total: int


@dataclass(frozen=True)
class DiversityStats:
    """Move-diversity summary over a generation run's per-move records.

    ``distinct_move_ratio`` is unique played moves over all played moves;
    ``mean_top1_entropy`` is the mean (nats) of the renormalized entropy of
    each move's captured top-K distribution. Both fall as the policy
    collapses onto a few engine-mimicking lines.
    """

    n_moves: int
    distinct_move_ratio: float
    mean_top1_entropy: float


@dataclass(frozen=True)
class CollapseConfig:
    """Absolute floors and relative-drop fractions for ``diversity_collapsed``."""

    min_distinct_move_ratio: float
    entropy_floor: float
    entropy_drop_frac: float


def extract_ratchet_metrics(metrics: dict[str, Any]) -> RatchetMetrics:
    """Pull the ratchet signals out of a vs-stockfish ``metrics.json`` payload.

    ``win_rate`` and ``legal_rate`` are always present for a vs-stockfish
    run (``metrics.py`` emits them unconditionally), so they are indexed
    directly — a missing key is a schema regression we want to surface, not
    hide behind a default. ``centipawn_loss`` is only emitted when the run
    populated CPL (so the block itself is optional), but when it *is* present
    a malformed shape is a schema regression too, so its access is guarded
    alongside the others.
    """
    try:
        score_rate = float(metrics["win_rate"]["overall"]["score_rate"])
        legal_rate = float(metrics["legal_rate"]["overall"]["rate"])
        cpl_block = metrics.get("centipawn_loss")
        cpl_mean = float(cpl_block["overall"]["mean"]) if cpl_block is not None else None
    except (KeyError, TypeError) as exc:
        raise SelfImproveError(f"metrics.json missing required ratchet keys: {exc}") from exc
    return RatchetMetrics(score_rate=score_rate, legal_rate=legal_rate, cpl_mean=cpl_mean)


def ratchet_decision(candidate: RatchetMetrics, champion: RatchetMetrics, config: RatchetConfig) -> RatchetDecision:
    """Decide whether to keep ``candidate`` over the current ``champion``.

    Order: a legality veto first (a checkpoint that plays more illegal moves
    is rejected regardless of score), then the primary score-rate gate, then
    a centipawn-loss tie-break inside the score-rate epsilon band. Anything
    that is not a clear improvement reverts — the ratchet only ever moves
    uphill on the held-out signal.
    """
    if candidate.legal_rate < config.legal_rate_floor:
        return RatchetDecision(keep=False, reason=f"legal_rate {candidate.legal_rate:.4f} below floor {config.legal_rate_floor:.4f}")
    if candidate.legal_rate < champion.legal_rate - config.legal_rate_max_regression:
        return RatchetDecision(keep=False, reason=f"legal_rate regressed {champion.legal_rate:.4f} -> {candidate.legal_rate:.4f}")

    delta = candidate.score_rate - champion.score_rate
    if delta > config.score_rate_epsilon:
        return RatchetDecision(keep=True, reason=f"score_rate improved {champion.score_rate:.4f} -> {candidate.score_rate:.4f}")
    if delta < -config.score_rate_epsilon:
        return RatchetDecision(keep=False, reason=f"score_rate regressed {champion.score_rate:.4f} -> {candidate.score_rate:.4f}")

    if candidate.cpl_mean is not None and champion.cpl_mean is not None and candidate.cpl_mean < champion.cpl_mean:
        return RatchetDecision(keep=True, reason=f"score_rate tie; cpl improved {champion.cpl_mean:.1f} -> {candidate.cpl_mean:.1f}")
    return RatchetDecision(keep=False, reason=f"no improvement (score_rate delta {delta:+.4f} within epsilon)")


def anchor_sample_count(online_n: int, fraction: float) -> int:
    """Number of anchor lines to reach ``fraction`` of the combined set.

    ``fraction`` is the share of the *combined* output the anchor should
    make up: ``anchor / (online + anchor) = fraction``, so
    ``anchor = fraction / (1 - fraction) * online``. ``fraction == 0``
    disables anchoring; ``fraction`` outside ``[0, 1)`` is a usage error.
    """
    if not 0.0 <= fraction < 1.0:
        raise SelfImproveError(f"fresh-data fraction must be in [0, 1); got {fraction}")
    if fraction == 0.0 or online_n == 0:
        return 0
    return round(fraction / (1.0 - fraction) * online_n)


def mix_anchor_pairs(*, online_path: Path, anchor_path: Path | None, fraction: float, seed: int, out_path: Path) -> AnchorMixResult:
    """Write ``out_path`` = online pairs + a seeded sample of anchor pairs, shuffled.

    Each JSONL line is treated opaquely (a complete pair object), so no
    parsing or schema coupling is needed here. When fewer anchor lines exist
    than the target sample count, all of them are used — the caller can
    detect the shortfall from the returned counts.
    """
    online_lines = _read_nonempty_lines(online_path)
    if fraction == 0.0 or anchor_path is None:
        _write_lines(out_path, online_lines)
        return AnchorMixResult(online=len(online_lines), anchor=0, total=len(online_lines))

    target = anchor_sample_count(len(online_lines), fraction)
    anchor_lines = _read_nonempty_lines(anchor_path)
    rng = random.Random(seed)
    sampled = anchor_lines if target >= len(anchor_lines) else rng.sample(anchor_lines, target)
    combined = [*online_lines, *sampled]
    rng.shuffle(combined)
    _write_lines(out_path, combined)
    return AnchorMixResult(online=len(online_lines), anchor=len(sampled), total=len(combined))


def validate_anchor_pairs(anchor_path: Path) -> None:
    """Raise unless ``anchor_path``'s first line is an export-dpo pair object.

    Only the first non-empty line is inspected — a cheap guard against pointing
    the loop at the wrong file (raw records, a metrics blob) before a whole
    generate+train cycle is wasted. The required keys mirror what
    ``train_dpo._load_pairs`` reads off each line.
    """
    lines = _read_nonempty_lines(anchor_path)
    if not lines:
        raise SelfImproveError(f"anchor pairs file is empty: {anchor_path}")
    try:
        first = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise SelfImproveError(f"anchor pairs file {anchor_path} is not JSONL: {exc}") from exc
    if not isinstance(first, dict):
        raise SelfImproveError(f"anchor pairs file {anchor_path}: each line must be a JSON object, got {type(first).__name__}")
    missing = [key for key in ANCHOR_REQUIRED_KEYS if key not in first]
    if missing:
        raise SelfImproveError(f"anchor pairs file {anchor_path} missing export-dpo keys {missing}")


def diversity_stats(records: Sequence[Record]) -> DiversityStats:
    """Compute move diversity over the ``PerMoveRecord``s in a generation run."""
    moves = [r for r in records if isinstance(r, PerMoveRecord)]
    if not moves:
        return DiversityStats(n_moves=0, distinct_move_ratio=0.0, mean_top1_entropy=0.0)
    played = [m.model_move for m in moves]
    distinct_ratio = len(set(played)) / len(played)
    entropies = [_topk_entropy([e.logprob for e in m.model_top_k]) for m in moves if m.model_top_k]
    mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    return DiversityStats(n_moves=len(moves), distinct_move_ratio=distinct_ratio, mean_top1_entropy=mean_entropy)


def diversity_collapsed(current: DiversityStats, baseline: DiversityStats, config: CollapseConfig) -> tuple[bool, str]:
    """Return ``(collapsed, reason)`` comparing a generation run to the first one.

    Collapse fires on an absolute distinct-move floor, an absolute entropy
    floor, or a relative entropy drop versus the baseline generation — any
    one is enough to stop the loop before degeneracy compounds.

    A generation that recorded zero moves is a *failed/empty run*, not mode
    collapse, and is reported as such so the caller does not misdiagnose it.
    """
    if current.n_moves == 0:
        return False, "no moves recorded (empty generation, not collapse)"
    if current.distinct_move_ratio < config.min_distinct_move_ratio:
        return True, f"distinct_move_ratio {current.distinct_move_ratio:.3f} below floor {config.min_distinct_move_ratio:.3f}"
    if current.mean_top1_entropy < config.entropy_floor:
        return True, f"mean_top1_entropy {current.mean_top1_entropy:.3f} below floor {config.entropy_floor:.3f}"
    if baseline.mean_top1_entropy > 0.0:
        drop = 1.0 - current.mean_top1_entropy / baseline.mean_top1_entropy
        if drop > config.entropy_drop_frac:
            return True, f"mean_top1_entropy dropped {drop:.0%} vs baseline (> {config.entropy_drop_frac:.0%})"
    return False, "diversity within bounds"


def select_best_checkpoint(dpo_output_dir: Path) -> Path:
    """Pick the behaviorally-best checkpoint, not merely the final step.

    ``train_dpo`` keeps every eval-step checkpoint precisely because the last
    one is often *worse* on vs-stockfish — its documented "v2c lesson":
    ``rewards/accuracies`` plateaus while ``eval_loss`` keeps falling, so late
    checkpoints overfit. We read ``trainer_state.json`` and return the
    existing checkpoint whose step maximizes ``eval_rewards/accuracies``
    (tie-break: earlier step). When the run was too short to log any eval
    metric, we fall back to the final step.
    """
    candidates = list(dpo_output_dir.glob(CHECKPOINT_GLOB))
    if not candidates:
        raise SelfImproveError(f"no checkpoints found under {dpo_output_dir}/{CHECKPOINT_GLOB}")
    latest = max(candidates, key=_checkpoint_step)
    accuracy_by_step = _eval_accuracy_by_step(latest / TRAINER_STATE_FILENAME)
    by_step = {_checkpoint_step(c): c for c in candidates}
    scored = [step for step in by_step if step in accuracy_by_step]
    if not scored:
        return latest
    best = max(scored, key=lambda step: (accuracy_by_step[step], -step))
    return by_step[best]


def _eval_accuracy_by_step(trainer_state_path: Path) -> dict[int, float]:
    """Map training step -> ``eval_rewards/accuracies`` from a trainer_state.json log_history.

    A missing, unreadable, or half-written (corrupt-JSON) trainer_state.json
    yields ``{}`` so the caller falls back to the final checkpoint rather than
    aborting the whole loop on a transient file.
    """
    try:
        state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    history = state.get("log_history", [])
    return {entry["step"]: entry[EVAL_ACCURACY_KEY] for entry in history if EVAL_ACCURACY_KEY in entry and "step" in entry}


def draw_fraction(winrate_overall: dict[str, Any]) -> float:
    """Share of *scored* (finished) games that were draws, from a win_rate.overall block."""
    scored = winrate_overall["games"] - winrate_overall["unfinished"]
    return winrate_overall["draws"] / scored if scored else 0.0


def _checkpoint_step(path: Path) -> int:
    """Parse the integer training step out of a ``checkpoint-<step>`` directory name."""
    suffix = path.name.removeprefix(CHECKPOINT_PREFIX)
    if not suffix.isdigit():
        raise SelfImproveError(f"unparseable checkpoint name: {path.name}")
    return int(suffix)


def _topk_entropy(logprobs: list[float]) -> float:
    """Renormalized Shannon entropy (nats) of a top-K log-probability list.

    The captured top-K is a truncated distribution that need not sum to one,
    so probabilities are renormalized over the K entries before computing
    entropy. A single-entry list has entropy 0 (a fully peaked policy).
    """
    if len(logprobs) < 2:
        return 0.0
    hi = max(logprobs)
    weights = [math.exp(lp - hi) for lp in logprobs]
    total = sum(weights)
    if total <= 0.0:
        return 0.0
    probs = [w / total for w in weights]
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def _read_nonempty_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as fp:
        return [line.strip() for line in fp if line.strip()]


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for line in lines:
            fp.write(line + "\n")


def decision_record(
    *,
    iteration: int,
    candidate: RatchetMetrics,
    champion: RatchetMetrics,
    decision: RatchetDecision,
    diversity: DiversityStats,
    candidate_winrate: dict[str, Any],
    champion_winrate: dict[str, Any],
) -> dict[str, Any]:
    """Build the JSON-serializable provenance blob written per iteration.

    The raw ``win_rate.overall`` blocks (games/wins/losses/draws/unfinished)
    are recorded for both sides so draw-stalling — which inflates score_rate
    without real improvement — is auditable after the fact.
    """
    return {
        "iteration": iteration,
        "kept": decision.keep,
        "reason": decision.reason,
        "candidate": metrics_to_dict(candidate),
        "champion": metrics_to_dict(champion),
        "candidate_win_rate": candidate_winrate,
        "champion_win_rate": champion_winrate,
        "generation_diversity": {
            "n_moves": diversity.n_moves,
            "distinct_move_ratio": diversity.distinct_move_ratio,
            "mean_top1_entropy": diversity.mean_top1_entropy,
        },
    }


def metrics_to_dict(metrics: RatchetMetrics) -> dict[str, float | None]:
    """Serialize the three ratchet signals to a JSON-friendly dict (shared by decision.json and session.json)."""
    return {"score_rate": metrics.score_rate, "legal_rate": metrics.legal_rate, "cpl_mean": metrics.cpl_mean}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a pretty-printed JSON file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
