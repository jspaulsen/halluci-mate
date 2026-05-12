"""Post-hoc DPO preference-pair exporter.

Turns an existing ``vs_stockfish`` run directory into a JSONL of
``{prompt, chosen, rejected}`` preference pairs suitable for DPO training.
No replay: every pair is derived from records already on disk.

Two flavors, per ``docs/eval_harness.md`` §DPO seed data:

* **Legality** — works on any ``vs_stockfish`` run. Selects per-move
  records where the unconstrained raw sample was illegal
  (``mask_used and not raw_sample_legal``). The legal post-mask move is
  the chosen completion; the illegal raw sample is rejected.
* **Quality** — requires the run was collected with ``--sf-analyze`` so
  ``sf_best_move`` and ``centipawn_loss`` are populated. Selects records
  where ``centipawn_loss`` exceeds ``threshold``. Stockfish's best move
  is chosen; the model's move is rejected.

The output schema lives in this module by design — DPO trainer format is
a downstream concern and may evolve without changing the eval record
schema (see §DPO seed data, last paragraph).
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from halluci_mate.eval.records import PerMoveRecord
from halluci_mate.eval.runs import RunReader

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

# Default centipawn-loss threshold for quality pairs. Matches the default
# ``--blunder-threshold-cp`` in ``vs-stockfish`` so quality DPO pairs and
# blunder-rate metrics share one calibration point unless the user overrides.
DEFAULT_QUALITY_THRESHOLD_CP = 200

# Key under which ``vs_stockfish`` persists the ``--sf-analyze`` flag into
# ``config.json`` (see ``VsStockfishConfig.analyze``). Centralized here so a
# rename in the evaluator surfaces as a single test failure.
CONFIG_ANALYZE_KEY = "analyze"


class DpoFlavor(StrEnum):
    LEGALITY = "legality"
    QUALITY = "quality"
    BOTH = "both"


class DpoExportError(ValueError):
    """Raised when an export request is incompatible with the run's records."""


class DpoPair(BaseModel):
    """One DPO preference pair, serialized as JSONL.

    ``prompt`` is the FEN before the move; ``chosen`` / ``rejected`` are UCI
    moves. The DPO trainer is responsible for templating the prompt and
    moves into whatever string form the model expects — the exporter does
    not commit to a tokenization here.
    """

    model_config = ConfigDict(frozen=True)

    prompt: str
    chosen: str
    rejected: str


def build_legality_pairs(records: Iterable[PerMoveRecord]) -> Iterator[DpoPair]:
    """Yield one pair per record where masking rescued an illegal raw sample.

    ``mask_used`` is required so the chosen move (``model_move``) is the
    *constrained* legal move. Unmasked records where the raw sample was
    illegal terminate the game with ``illegal-move`` and have
    ``model_move == raw_sample_move`` — both sides would be illegal and the
    pair would be useless for legality training.
    """
    for record in records:
        if record.mask_used and not record.raw_sample_legal:
            yield DpoPair(prompt=record.fen_before, chosen=record.model_move, rejected=record.raw_sample_move)


def build_quality_pairs(records: Iterable[PerMoveRecord], threshold: int) -> Iterator[DpoPair]:
    """Yield one pair per record where the model lost more than ``threshold`` centipawns.

    Skips records where Stockfish's best move equals the model's move: a
    nonzero CPL on an identical move means the post-move analysis diverged
    from the pre-move PV (search noise), and a pair that prefers a move
    over itself would be a no-op for DPO.
    """
    for record in records:
        if record.sf_best_move is None or record.centipawn_loss is None:
            continue
        if record.centipawn_loss <= threshold:
            continue
        if record.sf_best_move == record.model_move:
            continue
        yield DpoPair(prompt=record.fen_before, chosen=record.sf_best_move, rejected=record.model_move)


def export_dpo(
    *,
    run_dir: Path,
    output: Path,
    flavor: DpoFlavor,
    threshold: int = DEFAULT_QUALITY_THRESHOLD_CP,
    dedup_by_fen: bool = False,
) -> int:
    """Read ``run_dir``'s records, build pairs, write JSONL to ``output``.

    Returns the number of pairs written. Raises ``DpoExportError`` when the
    quality flavor is requested against a run that was not collected with
    ``--sf-analyze`` — without CPL data the quality filter can only match
    zero records, and silently emitting an empty file would hide the
    misuse.
    """
    reader = RunReader(run_dir)
    config = reader.read_config()
    per_move_records = [r for r in reader.read_records() if isinstance(r, PerMoveRecord)]

    analyze = config.get(CONFIG_ANALYZE_KEY)
    if flavor in (DpoFlavor.QUALITY, DpoFlavor.BOTH) and not analyze:
        raise DpoExportError(
            f"flavor={flavor.value} requires a run collected with --sf-analyze; {run_dir}/config.json has {CONFIG_ANALYZE_KEY}={analyze!r}",
        )

    pairs: list[DpoPair] = []
    if flavor in (DpoFlavor.LEGALITY, DpoFlavor.BOTH):
        pairs.extend(build_legality_pairs(per_move_records))
    if flavor in (DpoFlavor.QUALITY, DpoFlavor.BOTH):
        pairs.extend(build_quality_pairs(per_move_records, threshold))

    if dedup_by_fen:
        pairs = _dedup_by_fen(pairs)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fp:
        for pair in pairs:
            fp.write(json.dumps(pair.model_dump(), separators=(",", ":")) + "\n")
    return len(pairs)


def _dedup_by_fen(pairs: list[DpoPair]) -> list[DpoPair]:
    """Keep the first pair per ``prompt`` (FEN).

    Order is preserved so that callers can reason about which record won a
    collision — the earlier-appearing record (by ``event_id``) wins. With
    ``--flavor both``, legality pairs are appended before quality pairs, so
    a legality pair takes precedence over a quality pair on the same FEN.
    """
    seen: set[str] = set()
    out: list[DpoPair] = []
    for pair in pairs:
        if pair.prompt in seen:
            continue
        seen.add(pair.prompt)
        out.append(pair)
    return out
