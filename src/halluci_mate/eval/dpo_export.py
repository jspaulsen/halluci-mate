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

from halluci_mate.eval.metrics import LOST_POSITION_THRESHOLD_CP, drop_repetition_moves, is_consequential
from halluci_mate.eval.records import PerMoveRecord, Side
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

    Fields:

    * ``prompt`` — FEN before the move. Kept for human inspection and dedup;
      the model itself is conditioned on ``moves_uci`` + ``model_side``, not
      on the FEN string.
    * ``moves_uci`` — UCI move history from the game's start position up to
      (but not including) the move being labeled. Combined with
      ``model_side`` this is the exact prompt context the model saw when it
      made the rejected move — required because the chess tokenizer has no
      FEN vocabulary (see ``Game.tokenize``).
    * ``model_side`` — ``"white"`` or ``"black"``; determines the leading
      ``<WHITE>``/``<BLACK>`` perspective token at training time.
    * ``chosen`` / ``rejected`` — UCI moves. The DPO trainer is responsible
      for templating these into the model's token form; the exporter does
      not commit to a tokenization here.
    """

    model_config = ConfigDict(frozen=True)

    prompt: str
    moves_uci: list[str]
    model_side: str
    chosen: str
    rejected: str


def _build_history_index(records: Iterable[PerMoveRecord]) -> dict[int, list[str]]:
    """Map ``event_id`` → UCI move history up to (excluding) that record's ply.

    Per-move records are only emitted on the model's plies, so the opponent's
    intervening move appears as the *next* record's ``prior_opponent_move``
    (see ``PerMoveRecord``). We walk each game in ply order, append the
    opponent's reply (if any) then the model's move after snapshotting, and
    the snapshot taken before either append is exactly the prompt context
    the model saw on that record.
    """
    by_game: dict[str, list[PerMoveRecord]] = {}
    for record in records:
        by_game.setdefault(record.game_id, []).append(record)

    index: dict[int, list[str]] = {}
    for game_records in by_game.values():
        game_records.sort(key=lambda r: r.ply)
        history: list[str] = []
        for record in game_records:
            if record.prior_opponent_move is not None:
                history.append(record.prior_opponent_move)
            index[record.event_id] = list(history)
            history.append(record.model_move)
    return index


def _side_value(side: Side | str) -> str:
    """Normalize a ``Side`` enum or its string form to its ``str`` value."""
    return side.value if isinstance(side, Side) else side


def build_legality_pairs(records: Iterable[PerMoveRecord]) -> Iterator[DpoPair]:
    """Yield one pair per record where masking rescued an illegal raw sample.

    ``mask_used`` is required so the chosen move (``model_move``) is the
    *constrained* legal move. Unmasked records where the raw sample was
    illegal terminate the game with ``illegal-move`` and have
    ``model_move == raw_sample_move`` — both sides would be illegal and the
    pair would be useless for legality training.
    """
    records = list(records)
    history_index = _build_history_index(records)
    for record in records:
        if record.mask_used and not record.raw_sample_legal:
            yield DpoPair(
                prompt=record.fen_before,
                moves_uci=history_index[record.event_id],
                model_side=_side_value(record.model_side),
                chosen=record.model_move,
                rejected=record.raw_sample_move,
            )


def build_quality_pairs(
    records: Iterable[PerMoveRecord],
    threshold: int,
    *,
    require_consequential: bool = False,
    exclude_repetition: bool = False,
    lost_position_threshold_cp: int = LOST_POSITION_THRESHOLD_CP,
) -> Iterator[DpoPair]:
    """Yield one pair per record where the model lost more than ``threshold`` centipawns.

    Skips records where Stockfish's best move equals the model's move: a
    nonzero CPL on an identical move means the post-move analysis diverged
    from the pre-move PV (search noise), and a pair that prefers a move
    over itself would be a no-op for DPO.

    ``require_consequential`` keeps only moves played from a not-yet-lost
    position (model-perspective eval-before >= ``-lost_position_threshold_cp``).
    Blunders in already-lost endgames teach the model to chase Stockfish's
    swindle-line, which usually does not generalize — see
    ``BlunderPositionContextStats`` for the rationale.

    ``exclude_repetition`` drops moves whose threefold-repetition key
    recurs within the same game. King-shuffle / forced-draw sequences are
    flagged as blunders by Stockfish even when repetition is precisely
    what saves the half-point; pairing against the "correct" non-repeating
    move teaches the model to abandon the only drawing line.

    ``exclude_repetition`` must materialize the input so the per-game
    counts can be tallied before iteration.
    """
    records = list(records)
    history_index = _build_history_index(records)
    if exclude_repetition:
        records = drop_repetition_moves(records)
    for record in records:
        if record.sf_best_move is None or record.centipawn_loss is None:
            continue
        if record.centipawn_loss <= threshold:
            continue
        if record.sf_best_move == record.model_move:
            continue
        if require_consequential and not is_consequential(record, lost_position_threshold_cp):
            continue
        yield DpoPair(
            prompt=record.fen_before,
            moves_uci=history_index[record.event_id],
            model_side=_side_value(record.model_side),
            chosen=record.sf_best_move,
            rejected=record.model_move,
        )


def export_dpo(
    *,
    run_dir: Path,
    output: Path,
    flavor: DpoFlavor,
    threshold: int = DEFAULT_QUALITY_THRESHOLD_CP,
    dedup_by_fen: bool = False,
    require_consequential: bool = False,
    exclude_repetition: bool = False,
) -> int:
    """Read ``run_dir``'s records, build pairs, write JSONL to ``output``.

    Returns the number of pairs written. Raises ``DpoExportError`` when the
    quality flavor is requested against a run that was not collected with
    ``--sf-analyze`` — without CPL data the quality filter can only match
    zero records, and silently emitting an empty file would hide the
    misuse.

    ``require_consequential`` and ``exclude_repetition`` are quality-only
    filters; legality pairs (which select on legality of a raw sample, not
    move quality) ignore them. See ``build_quality_pairs`` for semantics.
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
        pairs.extend(
            build_quality_pairs(
                per_move_records,
                threshold,
                require_consequential=require_consequential,
                exclude_repetition=exclude_repetition,
            )
        )

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
