"""Cross-run comparison helpers for the eval harness.

Streamlit UI lives in ``scripts/compare.py`` and stays a thin shell over the
pure functions in this module: scan ``evals/``, group by checkpoint identity,
enumerate runs per (model, evaluator) pair (newest first), and load (or
recompute) metrics. Keeping the data layer here means the comparison is
testable without running Streamlit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from halluci_mate.eval.metrics import compute_all
from halluci_mate.eval.records import Evaluator
from halluci_mate.eval.runs import CONFIG_FILENAME, METRICS_FILENAME, RUN_ID_TIMESTAMP_FORMAT, RunReader

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

# Evaluators surfaced in the comparison UI. Puzzles is intentionally omitted —
# no aggregate metric pipeline is wired up for it yet.
COMPARED_EVALUATORS: tuple[Evaluator, ...] = (Evaluator.VS_STOCKFISH, Evaluator.LEGAL_RATE, Evaluator.PERPLEXITY)


@dataclass(frozen=True)
class RunEntry:
    """One discovered eval run on disk."""

    run_id: str
    run_dir: Path
    checkpoint: str
    evaluator: Evaluator
    timestamp: datetime


def discover_runs(evals_dir: Path) -> list[RunEntry]:
    """Return every readable run under ``evals_dir``.

    A run is "readable" iff its directory contains a ``config.json`` carrying
    both an ``evaluator`` and a ``checkpoint`` field, and the run-id's leading
    timestamp parses. Anything else is silently skipped — the eval directory
    accumulates partial / aborted runs, and the comparison view should not
    blow up on them.
    """
    if not evals_dir.is_dir():
        return []
    entries: list[RunEntry] = []
    for child in sorted(evals_dir.iterdir()):
        entry = _try_load_entry(child)
        if entry is not None:
            entries.append(entry)
    return entries


def list_checkpoints(runs: Iterable[RunEntry]) -> list[str]:
    """Return the distinct checkpoint identities present in ``runs``, sorted."""
    return sorted({r.checkpoint for r in runs})


def runs_for(runs: Iterable[RunEntry], checkpoint: str, evaluator: Evaluator) -> list[RunEntry]:
    """Return every run for the given pair, newest first."""
    matches = [r for r in runs if r.checkpoint == checkpoint and r.evaluator is evaluator]
    return sorted(matches, key=lambda r: r.timestamp, reverse=True)


def latest_run_for(runs: Iterable[RunEntry], checkpoint: str, evaluator: Evaluator) -> RunEntry | None:
    """Return the most recent run (by run-id timestamp) for the given pair, or None."""
    ordered = runs_for(runs, checkpoint, evaluator)
    return ordered[0] if ordered else None


def load_or_compute_metrics(entry: RunEntry) -> dict[str, Any]:
    """Return metrics for ``entry`` — read ``metrics.json`` if present, else recompute.

    Older runs predate the auto-write of ``metrics.json`` from ``compute_all``;
    rather than force the user to ``eval.py report`` each one before comparing,
    we recompute on the fly from ``records.jsonl``. Result is not written back
    to disk — that would be a side effect from a read-only view.
    """
    reader = RunReader(entry.run_dir)
    metrics_path = entry.run_dir / METRICS_FILENAME
    if metrics_path.exists():
        return reader.read_metrics()
    config = reader.read_config()
    return compute_all(reader.read_records(), config)


def _try_load_entry(run_dir: Path) -> RunEntry | None:
    if not run_dir.is_dir():
        return None
    config_path = run_dir / CONFIG_FILENAME
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    evaluator_value = config.get("evaluator")
    checkpoint = config.get("checkpoint")
    if not isinstance(evaluator_value, str) or not isinstance(checkpoint, str):
        return None
    try:
        evaluator = Evaluator(evaluator_value)
    except ValueError:
        return None
    timestamp = _parse_run_timestamp(run_dir.name)
    if timestamp is None:
        return None
    return RunEntry(
        run_id=run_dir.name,
        run_dir=run_dir,
        checkpoint=checkpoint,
        evaluator=evaluator,
        timestamp=timestamp,
    )


def _parse_run_timestamp(run_id: str) -> datetime | None:
    head, _, _ = run_id.partition("_")
    try:
        return datetime.strptime(head, RUN_ID_TIMESTAMP_FORMAT)
    except ValueError:
        return None
