"""Read-only leaderboard over a set of vs-stockfish search-sweep runs.

Scans an evals directory for runs whose checkpoint-tag starts with a given
prefix and prints one row per run: the search knobs (`search_k`,
`search_margin`) from `config.json`, the headline metrics (`score_rate`, CPL
median/p95, blunder/legal/tactical rates) via the shared `eval.compare`
loaders, and the override-rate derived from records (fraction of model
decisions where the played move differs from the policy argmax). Rows are
sorted by `score_rate`, best first. Writes nothing.

    uv run python scripts/sweep_summary.py --tag-prefix v2d-grid
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from halluci_mate.eval.compare import RunEntry, discover_runs, headline_metrics, load_or_compute_metrics
from halluci_mate.eval.records import Evaluator, PerMoveRecord, Record
from halluci_mate.eval.runs import RunReader

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_EVALS_DIR = Path("evals")


@dataclass(frozen=True)
class CellSummary:
    """One sweep cell: its search knobs plus headline + override metrics."""

    run_id: str
    search_k: int
    search_margin: float
    games: int
    score_rate: float
    wins: int
    draws: int
    losses: int
    cpl_median: float | None
    cpl_p95: float | None
    blunder_rate: float | None
    legal_rate: float | None
    tactical_oversight: float | None
    override_rate: float | None


def override_rate(records: Iterable[Record]) -> float | None:
    """Fraction of model decisions where the played move differs from the policy argmax.

    The policy argmax is ``model_top_k[0].move``; the played move is
    ``model_move`` (search's choice when search is on). Decisions with an empty
    ``model_top_k`` are skipped. Returns ``None`` when there are no eligible
    decisions.
    """
    eligible = 0
    overrides = 0
    for record in records:
        if not isinstance(record, PerMoveRecord) or not record.model_top_k:
            continue
        eligible += 1
        if record.model_move != record.model_top_k[0].move:
            overrides += 1
    return overrides / eligible if eligible else None


def summarize_run(entry: RunEntry) -> CellSummary | None:
    reader = RunReader(entry.run_dir)
    config = reader.read_config()
    if "search_k" not in config:
        return None  # not a search run — exclude it from the search sweep
    headline = headline_metrics(Evaluator.VS_STOCKFISH, load_or_compute_metrics(entry))
    return CellSummary(
        run_id=entry.run_id,
        search_k=int(config["search_k"]),
        search_margin=float(config["search_margin"]),
        games=int(config["games"]),
        score_rate=float(headline.get("score_rate", 0.0)),
        wins=int(headline.get("wins", 0)),
        draws=int(headline.get("draws", 0)),
        losses=int(headline.get("losses", 0)),
        cpl_median=headline.get("cpl_median"),
        cpl_p95=headline.get("cpl_p95"),
        blunder_rate=headline.get("blunder_rate"),
        legal_rate=headline.get("legal_rate"),
        tactical_oversight=headline.get("tactical_oversight"),
        override_rate=override_rate(reader.read_records()),
    )


def collect_summaries(evals_dir: Path, tag_prefix: str) -> list[CellSummary]:
    summaries = [
        summary
        for entry in discover_runs(evals_dir)
        if entry.evaluator is Evaluator.VS_STOCKFISH and _run_tag(entry.run_id).startswith(tag_prefix)
        if (summary := summarize_run(entry)) is not None
    ]
    summaries.sort(key=lambda cell: cell.score_rate, reverse=True)
    return summaries


def _run_tag(run_id: str) -> str:
    """Extract the tag segment from a `<timestamp>_<tag>_<evaluator>` run-id."""
    # `parts[1]` is the tag: `make_run_id` (halluci_mate.eval.runs) builds the id
    # as `<timestamp>_<tag>_<evaluator>` and forbids `_` in the tag.
    parts = run_id.split("_")
    return parts[1] if len(parts) >= 3 else ""


def _fmt(value: float | None, *, places: int = 1) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}"


def format_leaderboard(summaries: list[CellSummary]) -> str:
    header = f"{'k':>2} {'τ':>4} {'games':>5} {'score':>6} {'W/D/L':>9} {'cpl_med':>7} {'cpl_p95':>7} {'blndr%':>7} {'ovrd%':>6} {'legal%':>7} {'tact%':>6}"
    lines = [header, "-" * len(header)]
    for cell in summaries:
        wdl = f"{cell.wins}/{cell.draws}/{cell.losses}"
        lines.append(
            f"{cell.search_k:>2} {cell.search_margin:>4.1f} {cell.games:>5} {cell.score_rate:>6.3f} {wdl:>9} "
            f"{_fmt(cell.cpl_median, places=0):>7} {_fmt(cell.cpl_p95, places=0):>7} {_pct(cell.blunder_rate):>7} "
            f"{_pct(cell.override_rate):>6} {_pct(cell.legal_rate):>7} {_pct(cell.tactical_oversight):>6}"
        )
    return "\n".join(lines)


def main(
    tag_prefix: Annotated[str, typer.Option(help="Only summarize runs whose checkpoint-tag starts with this prefix (e.g. 'v2d-grid').")],
    evals_dir: Annotated[Path, typer.Option(help=f"Parent directory of run outputs (default: {DEFAULT_EVALS_DIR}).")] = DEFAULT_EVALS_DIR,
) -> None:
    summaries = collect_summaries(evals_dir, tag_prefix)
    if not summaries:
        typer.echo(f"No vs-stockfish runs under {evals_dir} with tag prefix {tag_prefix!r}.")
        raise typer.Exit(code=1)
    typer.echo(format_leaderboard(summaries))


if __name__ == "__main__":
    typer.run(main)
