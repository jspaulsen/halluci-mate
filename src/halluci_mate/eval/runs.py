"""Run-directory layout for the eval harness.

A run is one invocation of one evaluator against one checkpoint. Its outputs
all live under `evals/<run-id>/`:

* `config.json`     — evaluator name + full args + checkpoint path + git sha
* `records.jsonl`   — per-event records, append-only
* `metrics.json`    — computed aggregates (re-derivable from records)
* `games.pgn`       — optional, for game-based evaluators

See `docs/eval_harness.md` §Run artifact layout.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from halluci_mate.eval.records import Record, record_from_dict, record_to_dict

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

CONFIG_FILENAME = "config.json"
RECORDS_FILENAME = "records.jsonl"
METRICS_FILENAME = "metrics.json"
GAMES_PGN_FILENAME = "games.pgn"

# Colon is illegal in Windows paths and noisy on Unix shells; the spec uses
# `T<HH-MM-SS>` for the time component to keep run-ids tool-friendly.
RUN_ID_TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%S"


def make_run_id(checkpoint_tag: str, evaluator: str, *, now: datetime | None = None) -> str:
    """Return a run id of the form `<timestamp>_<checkpoint-tag>_<evaluator>`.

    `now` defaults to the current UTC time and is parameterizable for tests.
    """
    timestamp_source = now if now is not None else datetime.now(UTC)
    timestamp = timestamp_source.strftime(RUN_ID_TIMESTAMP_FORMAT)
    return f"{timestamp}_{checkpoint_tag}_{evaluator}"


class RunWriter:
    """Writer for a single eval run directory.

    `append_record` opens the JSONL file per call; this is simpler and
    crash-safe at the cost of per-write file open overhead. Acceptable for
    skeleton scale (hundreds-to-low-thousands of records per run).
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_config(self, config: dict[str, Any]) -> None:
        (self.run_dir / CONFIG_FILENAME).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    def append_record(self, record: Record) -> None:
        line = json.dumps(record_to_dict(record), separators=(",", ":")) + "\n"
        with (self.run_dir / RECORDS_FILENAME).open("a", encoding="utf-8") as fp:
            fp.write(line)

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        (self.run_dir / METRICS_FILENAME).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    def write_pgn(self, pgn: str) -> None:
        (self.run_dir / GAMES_PGN_FILENAME).write_text(pgn, encoding="utf-8")


class RunReader:
    """Reader for a single eval run directory."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def read_config(self) -> dict[str, Any]:
        return json.loads((self.run_dir / CONFIG_FILENAME).read_text(encoding="utf-8"))

    def read_metrics(self) -> dict[str, Any]:
        return json.loads((self.run_dir / METRICS_FILENAME).read_text(encoding="utf-8"))

    def read_records(self) -> Iterator[Record]:
        path = self.run_dir / RECORDS_FILENAME
        with path.open(encoding="utf-8") as fp:
            for raw in fp:
                line = raw.strip()
                if not line:
                    continue
                yield record_from_dict(json.loads(line))

    def read_pgn(self) -> str:
        return (self.run_dir / GAMES_PGN_FILENAME).read_text(encoding="utf-8")
