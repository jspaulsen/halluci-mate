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
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Self

from halluci_mate.eval.records import Evaluator, Record, record_from_dict, record_to_dict

if TYPE_CHECKING:
    from types import TracebackType

CONFIG_FILENAME = "config.json"
RECORDS_FILENAME = "records.jsonl"
METRICS_FILENAME = "metrics.json"
GAMES_PGN_FILENAME = "games.pgn"

# Colon is illegal in Windows paths and noisy on Unix shells; the spec uses
# `T<HH-MM-SS>` for the time component to keep run-ids tool-friendly.
RUN_ID_TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%S"

CHECKPOINT_PREFIX = "checkpoint-"


def make_run_id(checkpoint_tag: str, evaluator: Evaluator, *, now: datetime | None = None) -> str:
    """Return a run id of the form `<timestamp>_<checkpoint-tag>_<evaluator>`.

    `now` defaults to the current UTC time and is parameterizable for tests.
    `checkpoint_tag` may not contain `_` because the run-id format joins on
    `_` and underscores in the tag would make the id un-round-trippable.
    `evaluator` is converted via `Evaluator.run_id_tag` (hyphen-cased) so the
    enum's underscore form is the single source of truth.
    """
    if "_" in checkpoint_tag:
        raise ValueError(f"checkpoint_tag must not contain '_'; got {checkpoint_tag!r}")
    timestamp = (now or datetime.now(UTC)).strftime(RUN_ID_TIMESTAMP_FORMAT)
    return f"{timestamp}_{checkpoint_tag}_{evaluator.run_id_tag}"


def derive_checkpoint_tag(checkpoint: str) -> str:
    """Build a default ``checkpoint_tag`` for the run-id.

    Hugging Face repo ids (``org/name`` shape — exactly one ``/``, no leading
    ``.`` / ``/``, no ``\\``) reduce to ``name``. Local paths shaped like
    ``<run>/checkpoint-<step>`` collapse to ``<run>-ckpt<step>`` so the
    run-id stays compact. Anything else uses the basename.

    The HF heuristic is checked before the on-disk path lookup so a repo id
    like ``org/name`` is not misclassified as a local directory if a
    same-named directory happens to exist relative to ``cwd``.
    """
    if _looks_like_hf_repo_id(checkpoint):
        return checkpoint.split("/")[-1]
    path = Path(checkpoint)
    if path.is_dir() and path.name.startswith(CHECKPOINT_PREFIX) and path.parent.name:
        return f"{path.parent.name}-ckpt{path.name.removeprefix(CHECKPOINT_PREFIX)}"
    return path.name


def resolve_checkpoint_tag(checkpoint: str, override: str | None) -> str:
    """Pick the tag for a run-id, enforcing the no-``_`` invariant.

    A user-supplied ``override`` containing ``_`` is rejected loudly to keep
    the contract with ``make_run_id`` legible (the help text promises
    ``--checkpoint-tag`` may not contain ``_``). Auto-derived tags coming
    from ``derive_checkpoint_tag`` may legitimately contain ``_`` (e.g. a
    run dir named ``my_run``); for those we silently rewrite ``_`` to ``-``
    rather than failing on a path the user did not type.
    """
    if override is not None:
        if "_" in override:
            raise ValueError(f"checkpoint_tag override must not contain '_'; got {override!r}")
        return override
    return derive_checkpoint_tag(checkpoint).replace("_", "-")


def _looks_like_hf_repo_id(checkpoint: str) -> bool:
    """Return True for inputs shaped like ``org/name`` (HF repo id).

    Exactly one ``/``, no leading ``.`` or ``/``, no ``\\``. Anything else
    (including absolute paths, relative paths with multiple segments, and
    Windows-style paths) is treated as a local path candidate.
    """
    if not checkpoint or checkpoint[0] in (".", "/") or "\\" in checkpoint:
        return False
    return checkpoint.count("/") == 1


class RunWriter:
    """Writer for a single eval run directory.

    Use as a context manager — `append_record` writes through a held file
    handle for the run's lifetime, avoiding per-record open overhead on
    networked filesystems where `runs-v1/` typically lives. Each append is
    flushed so a crash mid-run still leaves a parseable JSONL prefix.

    `write_config` / `write_metrics` / `write_pgn` write whole files and do
    not require the context manager.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._records_fp: IO[str] | None = None

    def __enter__(self) -> Self:
        self._records_fp = (self.run_dir / RECORDS_FILENAME).open("a", encoding="utf-8")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._records_fp is not None:
            self._records_fp.close()
            self._records_fp = None

    def write_config(self, config: dict[str, Any]) -> None:
        (self.run_dir / CONFIG_FILENAME).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    def append_record(self, record: Record) -> None:
        if self._records_fp is None:
            raise RuntimeError("RunWriter must be used as a context manager before calling append_record")
        line = json.dumps(record_to_dict(record), separators=(",", ":")) + "\n"
        self._records_fp.write(line)
        self._records_fp.flush()

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

    def read_records(self) -> list[Record]:
        """Read all records eagerly. Records files are small enough to fit in memory."""
        path = self.run_dir / RECORDS_FILENAME
        records: list[Record] = []
        with path.open(encoding="utf-8") as fp:
            for raw in fp:
                line = raw.strip()
                if not line:
                    continue
                records.append(record_from_dict(json.loads(line)))
        return records

    def read_pgn(self) -> str:
        return (self.run_dir / GAMES_PGN_FILENAME).read_text(encoding="utf-8")
