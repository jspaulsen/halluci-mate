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
import statistics
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


@dataclass(frozen=True)
class CplBucket:
    """Centipawn-loss summary for one stratum (n=0 ⇒ all stats 0.0)."""

    n: int
    mean: float
    median: float
    p95: float


@dataclass(frozen=True)
class CplStats:
    """Overall + by-phase breakdown of centipawn loss."""

    overall: CplBucket
    by_phase: dict[str, CplBucket]


@dataclass(frozen=True)
class BlunderBucket:
    """Blunder tally for one stratum (n=0 ⇒ rate=0.0)."""

    n: int
    blunders: int
    rate: float


@dataclass(frozen=True)
class BlunderExcludingRepetitionStats:
    """Blunder rate after dropping moves stuck in a repeating position.

    Positions whose first four FEN fields (pieces / side-to-move /
    castling / en-passant — the threefold-repetition key) recur within a
    single game are filtered out before counting. The original blunder
    metric over-counts king-shuffle / forced-draw sequences where
    Stockfish's "best move" is "stop repeating, you'll lose," even when
    repetition is what saves the half-point.
    """

    overall: BlunderBucket
    by_phase: dict[str, BlunderBucket]


@dataclass(frozen=True)
class BlunderPositionContextStats:
    """Blunders split by the model-perspective eval *before* the move.

    `consequential` is blunders made when the model still had something
    to lose: model-relative eval before the move >=
    `-LOST_POSITION_THRESHOLD_CP`. `in_lost_position` is blunders made
    when the model was already that-far-down — they hardly change the
    outcome and dilute the headline blunder rate.
    """

    consequential: BlunderBucket
    in_lost_position: BlunderBucket


@dataclass(frozen=True)
class BlunderStats:
    """Overall + by-phase breakdown of blunder rate, plus context splits."""

    overall: BlunderBucket
    by_phase: dict[str, BlunderBucket]
    excluding_repetition: BlunderExcludingRepetitionStats
    by_position_context: BlunderPositionContextStats


# Model-perspective eval-before threshold (centipawns) for the
# "already lost" classification. A blunder from a position worse than
# this counts as `in_lost_position`; anything else is `consequential`.
# 300 cp ≈ a clean piece down — a position Stockfish at full strength
# converts from reliably, so further mistakes there are mostly noise.
LOST_POSITION_THRESHOLD_CP = 300


def compute_win_rate(games: list[PerGameRecord]) -> WinRateStats:
    """Aggregate per-game outcome records into win/draw/loss tallies."""
    return WinRateStats(
        overall=_winrate_bucket(games),
        by_model_side=_group_buckets(games, lambda g: g.model_side, Side, _winrate_bucket),
    )


def compute_legal_rate(moves: list[PerMoveRecord]) -> LegalRateStats:
    """Aggregate per-move records into legal-rate tallies, overall and stratified.

    Source field is `raw_sample_legal` — when `mask_used` is false the raw
    sample equals the played move and the bit is trivially true; the
    metric only carries signal when `mask_used` is true (i.e. constrained
    decoding).
    """
    return LegalRateStats(
        overall=_legal_rate_bucket(moves),
        by_phase=_group_buckets(moves, lambda m: m.phase, Phase, _legal_rate_bucket),
        by_model_side=_group_buckets(moves, lambda m: m.model_side, Side, _legal_rate_bucket),
    )


def compute_centipawn_loss(moves: list[PerMoveRecord]) -> CplStats:
    """Aggregate `centipawn_loss` over records that carry it.

    Records where `centipawn_loss is None` (i.e. `--sf-analyze` was off or the
    model played an illegal move) are skipped. With no usable records the
    overall bucket and every phase bucket collapse to zeros — matching the
    `LegalRateStats` convention so an `--sf-analyze`-off vs. -on diff stays
    structurally compatible.
    """
    return CplStats(
        overall=_cpl_bucket(moves),
        by_phase=_group_buckets(moves, lambda m: m.phase, Phase, _cpl_bucket),
    )


def compute_blunder_rate(moves: list[PerMoveRecord]) -> BlunderStats:
    """Aggregate `is_blunder` flags over records that carry them.

    Records where `is_blunder is None` are skipped — same gating as
    `compute_centipawn_loss`. See `BlunderExcludingRepetitionStats` and
    `BlunderPositionContextStats` for the rationale behind the two
    additional breakdowns.
    """
    non_repetition = _drop_repetition_moves(moves)
    consequential, in_lost = _partition_by_position_context(moves)
    return BlunderStats(
        overall=_blunder_bucket(moves),
        by_phase=_group_buckets(moves, lambda m: m.phase, Phase, _blunder_bucket),
        excluding_repetition=BlunderExcludingRepetitionStats(
            overall=_blunder_bucket(non_repetition),
            by_phase=_group_buckets(non_repetition, lambda m: m.phase, Phase, _blunder_bucket),
        ),
        by_position_context=BlunderPositionContextStats(
            consequential=_blunder_bucket(consequential),
            in_lost_position=_blunder_bucket(in_lost),
        ),
    )


def _cpl_bucket(moves: list[PerMoveRecord]) -> CplBucket:
    values = [m.centipawn_loss for m in moves if m.centipawn_loss is not None]
    if not values:
        return CplBucket(n=0, mean=0.0, median=0.0, p95=0.0)
    mean = sum(values) / len(values)
    median = float(statistics.median(values))
    # `statistics.quantiles` requires at least 2 data points; on a single
    # sample the 95th percentile is unambiguously that value.
    p95 = float(values[0]) if len(values) < 2 else float(statistics.quantiles(values, n=100)[94])
    return CplBucket(n=len(values), mean=mean, median=median, p95=p95)


def _blunder_bucket(moves: list[PerMoveRecord]) -> BlunderBucket:
    flagged = [m.is_blunder for m in moves if m.is_blunder is not None]
    blunders = sum(flagged)
    rate = blunders / len(flagged) if flagged else 0.0
    return BlunderBucket(n=len(flagged), blunders=blunders, rate=rate)


def _position_key(fen: str) -> str:
    """Return the threefold-repetition key for a FEN.

    The chess threefold-repetition rule compares positions on
    pieces / side-to-move / castling rights / en-passant target — the
    first four space-separated FEN fields. Half-move and full-move
    counters (fields 5-6) advance every move and would defeat the
    comparison.
    """
    return " ".join(fen.split()[:4])


def _drop_repetition_moves(moves: list[PerMoveRecord]) -> list[PerMoveRecord]:
    """Drop moves whose position-key recurs within the same game.

    A position seen more than once in the same `game_id` is treated as
    part of a repetition sequence; every visit (not just the second and
    later) is filtered. Inter-game collisions are ignored — repetition
    is a within-game concept.
    """
    counts: dict[tuple[str, str], int] = {}
    for move in moves:
        counts[(move.game_id, _position_key(move.fen_before))] = counts.get((move.game_id, _position_key(move.fen_before)), 0) + 1
    return [m for m in moves if counts[(m.game_id, _position_key(m.fen_before))] == 1]


def _partition_by_position_context(moves: list[PerMoveRecord]) -> tuple[list[PerMoveRecord], list[PerMoveRecord]]:
    """Split moves by model-perspective eval-before relative to the lost-position threshold.

    Returns `(consequential, in_lost_position)`. Moves missing
    `sf_eval_before_cp` (e.g. `--sf-analyze` off, illegal model move) are
    dropped from both buckets so they cannot inflate either rate; the
    `overall` bucket on `BlunderStats` is what carries the full
    population.
    """
    consequential: list[PerMoveRecord] = []
    in_lost: list[PerMoveRecord] = []
    for move in moves:
        if move.sf_eval_before_cp is None:
            continue
        sign = 1 if move.model_side == Side.WHITE else -1
        model_eval_before = sign * move.sf_eval_before_cp
        if model_eval_before >= -LOST_POSITION_THRESHOLD_CP:
            consequential.append(move)
        else:
            in_lost.append(move)
    return consequential, in_lost


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
    result: dict[str, Any] = {
        "evaluator": Evaluator.VS_STOCKFISH.value,
        "stockfish_skill": config.get("stockfish_skill"),
        "win_rate": asdict(compute_win_rate(games)),
        "legal_rate": asdict(compute_legal_rate(moves)),
    }
    # Only emit CPL / blunder blocks when the run actually populated them;
    # otherwise an `--sf-analyze`-off run would emit a misleading all-zeros
    # block and pollute schema diffs. A single populated record is enough to
    # opt in — the metric functions themselves skip per-record `None`s.
    if any(m.centipawn_loss is not None for m in moves):
        result["centipawn_loss"] = asdict(compute_centipawn_loss(moves))
        result["blunder_rate"] = asdict(compute_blunder_rate(moves))
    return result


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


def _group_buckets[R, B, E: StrEnum](
    records: list[R],
    key: Callable[[R], E],
    enum: type[E],
    bucket: Callable[[list[R]], B],
) -> dict[str, B]:
    return {member.value: bucket([r for r in records if key(r) == member]) for member in enum}
