"""Pure-function metric aggregation over eval records.

Metrics are pure functions over the records produced by an evaluator.
Adding a new metric never requires re-running an eval — the records
written by the evaluator are the contract.

See `docs/eval_harness.md` §Metrics and §Stratification.

Within-run breakdowns (e.g. `by_phase`, `by_model_side`) live alongside
the overall aggregate as nested dicts keyed by the stratifying dimension.
Keys are pinned to the enums declared in `records.py` so the on-disk
`metrics.json` schema is stable across runs and trivially diffable.
Cross-run dimensions like `stockfish_skill` are emitted as top-level
scalars here — comparing across them is the diff layer's job, not this
module's.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from halluci_mate.eval.records import (
    Evaluator,
    PerGameRecord,
    PerLegalRateRecord,
    PerMoveRecord,
    PerPerplexityRecord,
    Phase,
    Side,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from halluci_mate.eval.records import Record


@dataclass(frozen=True)
class WinRateBucket:
    """Game-outcome tally for one stratum.

    `win_rate` and `score_rate` use *scored* games (`games - unfinished`)
    as the denominator: an unfinished game (max-plies hit, illegal-move
    abort) is not a half-point and not a loss, so it should not silently
    pull the rate down. This is the harness convention — chess tournament
    forfeits are typically counted as losses; we do not, because the
    forfeit reason here is usually "the model couldn't finish" rather
    than "the model resigned the point". `scored == 0` collapses both
    rates to 0.0 — the `games` / `unfinished` counts make that case
    unambiguous.
    """

    games: int
    wins: int
    losses: int
    draws: int
    unfinished: int
    win_rate: float
    score_rate: float


@dataclass(frozen=True)
class WinRateStats:
    """Overall + by-model_side breakdown of game outcomes."""

    overall: WinRateBucket
    by_model_side: dict[str, WinRateBucket]


@dataclass(frozen=True)
class LegalRateBucket:
    """Legal-rate tally for one stratum (n=0 ⇒ rate=0.0)."""

    n: int
    legal: int
    rate: float


@dataclass(frozen=True)
class LegalRateStats:
    """Overall + by-phase + by-model_side breakdown of legal-move rate."""

    overall: LegalRateBucket
    by_phase: dict[str, LegalRateBucket]
    by_model_side: dict[str, LegalRateBucket]


def compute_win_rate(games: list[PerGameRecord]) -> WinRateStats:
    """Aggregate per-game outcome records into win/draw/loss tallies."""
    overall = _winrate_bucket(games)
    by_side = {side.value: _winrate_bucket([g for g in games if g.model_side == side]) for side in Side}
    return WinRateStats(overall=overall, by_model_side=by_side)


def compute_legal_rate(moves: list[PerMoveRecord]) -> LegalRateStats:
    """Aggregate per-move records into legal-rate tallies, overall and stratified.

    Source field is `raw_sample_legal` — when `mask_used` is false the raw
    sample equals the played move and the bit is trivially true; the
    metric only carries signal when `mask_used` is true (i.e. constrained
    decoding).
    """
    return LegalRateStats(
        overall=_legal_rate_bucket(moves),
        by_phase=_legal_rate_by(moves, lambda m: m.phase, Phase),
        by_model_side=_legal_rate_by(moves, lambda m: m.model_side, Side),
    )


def _legal_rate_tally(records: Iterable[Record]) -> LegalRateBucket:
    """Cross-record-type legal-rate counting for the LEGAL_RATE evaluator.

    Both `PerMoveRecord` and `PerLegalRateRecord` carry a legality bit;
    `_compute_legal_rate_aggregate` consumes either (or both) through this
    helper so its on-disk payload shape stays aligned with `vs_stockfish`.
    """
    n = 0
    legal = 0
    for r in records:
        if isinstance(r, PerMoveRecord):
            n += 1
            legal += r.raw_sample_legal
        elif isinstance(r, PerLegalRateRecord):
            n += 1
            legal += r.legal
    return LegalRateBucket(n=n, legal=legal, rate=legal / n if n else 0.0)


def compute_all(records: Iterable[Record], config: dict[str, Any]) -> dict[str, Any]:
    """Return the full aggregate dict, dispatching on the evaluator name.

    `config` is the on-disk `config.json` payload — the evaluator name
    lives there, not on every record. Records are consumed eagerly so the
    caller may pass a generator.
    """
    records_list = list(records)
    evaluator = config.get("evaluator")
    # When a second evaluator lands, swap this if/elif for an
    # `Evaluator -> compute fn` dispatch table.
    if evaluator == Evaluator.VS_STOCKFISH.value:
        return _compute_vs_stockfish(records_list, config)
    if evaluator == Evaluator.LEGAL_RATE.value:
        return _compute_legal_rate_aggregate(records_list)
    if evaluator == Evaluator.PERPLEXITY.value:
        return _compute_perplexity(records_list)
    raise ValueError(f"compute_all: unsupported evaluator {evaluator!r}")


def _compute_legal_rate_aggregate(records: list[Record]) -> dict[str, Any]:
    # Match the `vs_stockfish` payload shape: nest under `overall` so the on-disk
    # path is always `legal_rate.overall.{n,legal,rate}`, regardless of evaluator.
    return {
        "evaluator": Evaluator.LEGAL_RATE.value,
        "legal_rate": {
            "overall": asdict(_legal_rate_tally(records)),
        },
    }


def _compute_perplexity(records: list[Record]) -> dict[str, Any]:
    perp_records = [r for r in records if isinstance(r, PerPerplexityRecord)]
    all_logprobs = [lp for r in perp_records for lp in r.token_logprobs]
    num_tokens = len(all_logprobs)
    # Empty input collapses every aggregate to 0.0 (not exp(0) = 1.0) so a
    # zero-record run is unambiguous in `metrics.json`.
    mean_nll = -sum(all_logprobs) / num_tokens if num_tokens else 0.0
    return {
        "evaluator": Evaluator.PERPLEXITY.value,
        "num_sequences": len(perp_records),
        "num_tokens": num_tokens,
        "mean_nll": mean_nll,
        "bits_per_token": mean_nll / math.log(2) if num_tokens else 0.0,
        "perplexity": math.exp(mean_nll) if num_tokens else 0.0,
    }


def _compute_vs_stockfish(records: list[Record], config: dict[str, Any]) -> dict[str, Any]:
    moves = [r for r in records if isinstance(r, PerMoveRecord)]
    games = [r for r in records if isinstance(r, PerGameRecord)]
    return {
        "evaluator": Evaluator.VS_STOCKFISH.value,
        "stockfish_skill": config.get("stockfish_skill"),
        "win_rate": asdict(compute_win_rate(games)),
        "legal_rate": asdict(compute_legal_rate(moves)),
    }


def _winrate_bucket(games: list[PerGameRecord]) -> WinRateBucket:
    wins = losses = draws = unfinished = 0
    for game in games:
        match game.result:
            case "*":
                unfinished += 1
            case "1/2-1/2":
                draws += 1
            case "1-0":
                wins += 1 if game.model_side == Side.WHITE else 0
                losses += 1 if game.model_side == Side.BLACK else 0
            case "0-1":
                wins += 1 if game.model_side == Side.BLACK else 0
                losses += 1 if game.model_side == Side.WHITE else 0
    scored = len(games) - unfinished
    win_rate = wins / scored if scored else 0.0
    score_rate = (wins + 0.5 * draws) / scored if scored else 0.0
    return WinRateBucket(
        games=len(games),
        wins=wins,
        losses=losses,
        draws=draws,
        unfinished=unfinished,
        win_rate=win_rate,
        score_rate=score_rate,
    )


def _legal_rate_bucket(moves: list[PerMoveRecord]) -> LegalRateBucket:
    legal = sum(m.raw_sample_legal for m in moves)
    rate = legal / len(moves) if moves else 0.0
    return LegalRateBucket(n=len(moves), legal=legal, rate=rate)


def _legal_rate_by[E: StrEnum](
    moves: list[PerMoveRecord],
    key: Callable[[PerMoveRecord], E],
    enum: type[E],
) -> dict[str, LegalRateBucket]:
    return {member.value: _legal_rate_bucket([m for m in moves if key(m) == member]) for member in enum}
