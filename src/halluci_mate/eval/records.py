"""Record models for the eval harness.

Schema mirrors `docs/eval_harness.md` §Record schema. Every record carries the
common header (run_id, event_id, evaluator, checkpoint); the rest is shaped
per the workload — per-move (vs_stockfish, optionally puzzles), per-puzzle
(puzzles), per-legal-rate (legal_rate), or per-perplexity (perplexity).

`legal_rate` and `perplexity` are deliberately separate record types rather
than a shared "per-position" shape: they have disjoint payloads (move + legality
vs. per-token logprobs) and distinct metric pipelines, so unifying them only
forces filler values into one column or the other.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Tag, TypeAdapter

# PGN-style game outcome strings; `*` = unfinished (max-plies, illegal-move, etc.).
GameResult = Literal["1-0", "0-1", "1/2-1/2", "*"]

# Why a vs_stockfish game ended. Lives here (not on the evaluator) so
# `PerGameRecord` can pin the field without a circular import.
Termination = Literal["natural", "max-plies", "stockfish-resigned", "illegal-move"]


class Evaluator(StrEnum):
    VS_STOCKFISH = "vs_stockfish"
    PUZZLES = "puzzles"
    LEGAL_RATE = "legal_rate"
    PERPLEXITY = "perplexity"

    @property
    def run_id_tag(self) -> str:
        """Hyphen-cased form of the evaluator name for use in run-ids.

        Run-ids join components on `_`, so the evaluator slug must use `-`
        instead of `_` to keep the id round-trippable.
        """
        return self.value.replace("_", "-")


class Phase(StrEnum):
    OPENING = "opening"
    MIDDLE = "middle"
    ENDGAME = "endgame"


class Side(StrEnum):
    WHITE = "white"
    BLACK = "black"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class TopKEntry(_Frozen):
    """One entry in `model_top_k`: a candidate move and its log-probability."""

    move: str
    logprob: float


class RecordHeader(_Frozen):
    """Common header fields present on every eval record."""

    run_id: str
    event_id: int
    evaluator: Evaluator
    checkpoint: str


class PerMoveRecord(RecordHeader):
    """A single model decision in a game (vs_stockfish; optionally puzzles).

    Records are emitted only on plies where ``side_to_move == model_side``
    (i.e., one record per model decision, not one per ply). The opponent's
    reply to ``model_move`` shows up as ``prior_opponent_move`` on the
    *next* record for the same ``game_id`` — game reconstruction walks
    records in ``ply`` order and interleaves ``prior_opponent_move`` (when
    present) with ``model_move``.

    Sampling contract (relied on by metrics and DPO export):

    * ``model_move`` is what was actually played — post-mask when
      ``mask_used`` is true, the unconstrained top-1 otherwise.
    * ``raw_sample_move`` is always the unconstrained top-1. When
      ``mask_used`` is false it equals ``model_move``. When ``mask_used``
      is true and ``raw_sample_legal`` is false, this is the rejected
      illegal move that the Phase 2 (legality) DPO exporter pairs
      against ``model_move`` — storing it here, rather than relying on
      ``model_top_k[0]``, decouples DPO export from the configured top-k
      size and from whether top-k is captured pre- or post-mask.
    * ``model_top_k`` is the distribution actually sampled from:
      post-mask when ``mask_used`` is true, unconstrained otherwise.
      Do not use it to recover the raw-sample move — use
      ``raw_sample_move``.
    * ``prior_opponent_move`` is the move at ``ply - 1`` that produced
      ``fen_before``. ``None`` only when this is the game's first ply
      (``ply == 0``) and the model plays White.
    """

    game_id: str
    ply: int
    phase: Phase
    side_to_move: Side
    model_side: Side
    fen_before: str
    legal_moves: list[str]
    model_move: str
    model_top_k: list[TopKEntry]
    mask_used: bool
    raw_sample_move: str
    raw_sample_legal: bool
    prior_opponent_move: str | None
    sf_best_move: str | None
    sf_eval_before_cp: int | None
    sf_eval_after_cp: int | None
    centipawn_loss: int | None
    is_blunder: bool | None


class PerGameRecord(RecordHeader):
    """Terminal summary for one game in a `vs_stockfish` run.

    Per-move records describe individual model decisions; aggregating
    win/draw/loss requires terminal information that does not appear on any
    single move (the game can end on the opponent's reply, on max-plies, or
    on an illegal model move). Capturing that summary as a record keeps the
    metrics module a pure function over `records.jsonl` — no PGN parsing or
    config reading needed for win-rate aggregation.

    Emitted once per game by the evaluator, after the last per-move record
    for that game.

    The ``result`` field is the discriminator tag for this record class —
    see ``_discriminate_record``. Do not add a ``result`` field to any
    other record class without also updating the discriminator, or new
    records will silently validate as ``PerGameRecord`` and lose their
    payload.
    """

    game_id: str
    model_side: Side
    result: GameResult
    termination: Termination
    ply_count: int


class PerPuzzleRecord(RecordHeader):
    """A single puzzle attempt."""

    puzzle_id: str
    rating: int
    themes: list[str]
    fen: str
    solution: list[str]
    model_attempt: list[str]
    solved: bool


class PerLegalRateRecord(RecordHeader):
    """One unconstrained-prediction-on-a-position event for `legal_rate`.

    Captures whether the model's top-1 *unconstrained* sample at `fen` is
    legal. Used as the source for Phase 2 progress measurement and for
    legality DPO data when no full game is available.
    """

    position_id: str
    fen: str
    model_move: str
    legal: bool


class PerPerplexityRecord(RecordHeader):
    """One scored continuation for `perplexity`.

    `token_logprobs` holds the per-token log-probability of the actual
    continuation token sequence given the prefix ending at `fen`. The
    aggregate metric is the negative-log-likelihood mean over tokens.
    No `model_move`/`legal` here: perplexity does not sample from the model.
    """

    position_id: str
    fen: str
    token_logprobs: list[float]


def _discriminate_record(value: Any) -> str | None:
    """Return the variant tag for a record dict, or None.

    Pydantic only invokes this discriminator with dicts (during validation);
    serialization dispatches on the model class itself. Discriminator is the
    presence of a type-specific key, not the `evaluator` string — the schema
    allows puzzles to optionally emit per-move records, so the evaluator name
    is not a strict type tag. `PerMoveRecord` and `PerGameRecord` both carry
    `game_id` (a per-move record summarizes one decision *within* a game; a
    per-game record summarizes the outcome of the game itself), so they are
    distinguished by the field unique to each: `ply` vs. `result`. The
    chosen distinguishing keys are pairwise disjoint across record classes
    — see `tests/eval/records_test.py::test_record_discriminator_keys_are_disjoint`.
    """
    if not isinstance(value, dict):
        return None
    if "ply" in value:
        return "per_move"
    if "result" in value:
        return "per_game"
    if "puzzle_id" in value:
        return "per_puzzle"
    if "token_logprobs" in value:
        return "per_perplexity"
    if "legal" in value:
        return "per_legal_rate"
    return None


Record = Annotated[
    Annotated[PerMoveRecord, Tag("per_move")]
    | Annotated[PerGameRecord, Tag("per_game")]
    | Annotated[PerPuzzleRecord, Tag("per_puzzle")]
    | Annotated[PerLegalRateRecord, Tag("per_legal_rate")]
    | Annotated[PerPerplexityRecord, Tag("per_perplexity")],
    Discriminator(_discriminate_record),
]

_RECORD_ADAPTER: TypeAdapter[Record] = TypeAdapter(Record)


def record_to_dict(record: Record) -> dict[str, Any]:
    """Serialize a record to a JSON-ready dict."""
    return _RECORD_ADAPTER.dump_python(record, mode="json")


def record_from_dict(data: dict[str, Any]) -> Record:
    """Reconstruct a record from a dict, dispatching on shape."""
    return _RECORD_ADAPTER.validate_python(data)
