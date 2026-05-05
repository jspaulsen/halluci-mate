"""Pure-function metric aggregation over eval records.

Metrics are pure functions over the records produced by an evaluator.
Adding a new metric never requires re-running an eval — the records
written by the evaluator are the contract.

See `docs/eval_harness.md` §Metrics and §Stratification.

Stratified breakdowns live alongside the overall aggregate as nested
dicts (not separate functions), keyed by the stratifying dimension. Keys
are pinned to the enums declared in `records.py` so the on-disk
`metrics.json` schema is stable across runs and trivially diffable.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
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
    from collections.abc import Iterable

    from halluci_mate.eval.records import Record

# Keys read from the `config.json` payload written by `vs_stockfish`.
# Centralized here so the metrics module's contract with the evaluator's
# config dict is explicit and grep-able.
_CONFIG_EVALUATOR_KEY = "evaluator"
_CONFIG_STOCKFISH_SKILL_KEY = "stockfish_skill"


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


def compute_win_rate(records: Iterable[Record]) -> WinRateStats:
    """Aggregate per-game outcome records into win/draw/loss tallies.

    Filters to `PerGameRecord`; non-game records (per-move, etc.) are
    ignored so callers can pass an unfiltered records stream.
    """
    games = [r for r in records if isinstance(r, PerGameRecord)]
    overall = _winrate_bucket(games)
    by_side = {side.value: _winrate_bucket([g for g in games if g.model_side == side]) for side in Side}
    return WinRateStats(overall=overall, by_model_side=by_side)


def compute_legal_rate(records: Iterable[Record]) -> float:
    """Fraction of records whose unconstrained top-1 sample was legal.

    Accepts both per-move records (`PerMoveRecord.raw_sample_legal`) and
    per-legal-rate records (`PerLegalRateRecord.legal`) so the same metric
    works across `vs_stockfish` and `legal_rate` runs. Other record types
    are ignored, so callers can pass an unfiltered records stream. For
    `PerMoveRecord`s, the source bit only carries signal when `mask_used`
    is true (otherwise raw == played and the bit is trivially true).
    Returns 0.0 if there are no relevant records.
    """
    return _legal_rate_tally(records).rate


def _legal_rate_tally(records: Iterable[Record]) -> LegalRateBucket:
    """Single source of truth for legal-rate counting across record types.

    Both `PerMoveRecord` and `PerLegalRateRecord` carry a legality bit; this
    helper consumes either (or both) and returns the bucket so callers that
    need only the rate (`compute_legal_rate`) and callers that need the full
    `n`/`legal`/`rate` shape (`_compute_legal_rate_aggregate`) share one pass.
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
    evaluator = config.get(_CONFIG_EVALUATOR_KEY)
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
    return {
        "evaluator": Evaluator.VS_STOCKFISH.value,
        "stockfish_skill": config.get(_CONFIG_STOCKFISH_SKILL_KEY),
        "win_rate": asdict(compute_win_rate(records)),
        "legal_rate": {
            "overall": asdict(_legal_rate_bucket(moves)),
            "by_phase": {phase.value: asdict(_legal_rate_bucket([m for m in moves if m.phase == phase])) for phase in Phase},
            "by_model_side": {side.value: asdict(_legal_rate_bucket([m for m in moves if m.model_side == side])) for side in Side},
        },
    }


def _winrate_bucket(games: list[PerGameRecord]) -> WinRateBucket:
    wins = losses = draws = unfinished = 0
    for game in games:
        match _classify_game(game):
            case "win":
                wins += 1
            case "loss":
                losses += 1
            case "draw":
                draws += 1
            case _:
                unfinished += 1
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


def _classify_game(game: PerGameRecord) -> str:
    """Map a `PerGameRecord` to one of `win`/`loss`/`draw`/`unfinished`."""
    if game.result == "*":
        return "unfinished"
    if game.result == "1/2-1/2":
        return "draw"
    halluci_won = (game.result == "1-0" and game.model_side == Side.WHITE) or (game.result == "0-1" and game.model_side == Side.BLACK)
    return "win" if halluci_won else "loss"


def _legal_rate_bucket(moves: list[PerMoveRecord]) -> LegalRateBucket:
    legal = sum(m.raw_sample_legal for m in moves)
    rate = legal / len(moves) if moves else 0.0
    return LegalRateBucket(n=len(moves), legal=legal, rate=rate)
