"""``perplexity`` evaluator: token-level cross-entropy on held-out sequences.

Cheapest metric in the harness; intended to run regularly. The evaluator does
**not** sample from the model — it only scores known token sequences. One
``PerPerplexityRecord`` is emitted per input row, carrying the per-token
log-probabilities of the actual continuation.

Input format (jsonl, one row per sequence):

```json
{"id": "g1", "perspective": "white", "moves": ["e2e4", "e7e5", ...], "is_draw": false}
```

* ``id`` → ``position_id`` on the record.
* ``perspective`` is ``"white"`` or ``"black"`` and selects the leading
  ``<WHITE>`` / ``<BLACK>`` token, mirroring ``game_to_sequences``.
* ``moves`` is a list of UCI move strings.
* ``is_draw`` (optional, default false): when true a ``<DRAW>`` token is
  appended before scoring, matching the training-data shape.

We score the move tokens (and the optional ``<DRAW>``) but not a trailing
``<EOS>``: ``<EOS>`` is near-deterministic given the rest of the sequence and
folding it in would deflate the NLL.

Every record's ``fen`` is the standard starting position because the v1 input
schema only carries full games (the move list always starts from move 0). The
schema column exists because future variants may score continuations from
non-start prefixes; until then the value is constant on every record.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Protocol

import chess
import torch

from halluci_mate.chess_tokenizer import (
    BLACK_TOKEN_ID,
    DRAW_TOKEN_ID,
    UNK_TOKEN_ID,
    WHITE_TOKEN_ID,
    ChessTokenizer,
)
from halluci_mate.eval.records import Evaluator, PerPerplexityRecord
from halluci_mate.eval.runs import RunWriter

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import Any


_PERSPECTIVE_TOKEN_IDS: dict[str, int] = {
    "white": WHITE_TOKEN_ID,
    "black": BLACK_TOKEN_ID,
}


class _ModelOutput(Protocol):
    """Minimum surface of a HuggingFace model output we read.

    The full ``CausalLMOutputWithPast`` shape is irrelevant — we only need
    ``logits`` for scoring, so the protocol pins exactly that and lets test
    stubs return a duck-typed object.
    """

    @property
    def logits(self) -> torch.Tensor: ...


class _ModelCallable(Protocol):
    """Callable that runs a forward pass over a batch of input ids.

    Real implementation: a HuggingFace ``PreTrainedModel`` invoked as
    ``model(input_ids=...)``. Stubbed in tests with a ``__call__`` that
    returns a duck-typed ``_ModelOutput``.
    """

    def __call__(self, *, input_ids: torch.Tensor) -> _ModelOutput: ...


class PerplexityScorer(Protocol):
    """Structural type for the engine surface used by the perplexity evaluator.

    Wider than ``Predictor`` (which just calls ``predict_with_metadata``):
    perplexity needs the raw model + tokenizer + device because it scores
    arbitrary token sequences rather than asking for one move. Defined as a
    Protocol so tests can stub the three attributes without standing up a
    real ``ChessInferenceEngine``.
    """

    @property
    def model(self) -> _ModelCallable: ...
    @property
    def tokenizer(self) -> ChessTokenizer: ...
    @property
    def device(self) -> torch.device: ...


@dataclass(frozen=True)
class PerplexityConfig:
    """Knobs for ``run_perplexity``.

    ``max_sequences`` short-circuits the input iterator after N rows; ``None``
    consumes the whole file.
    """

    data_path: Path
    max_sequences: int | None = None

    def __post_init__(self) -> None:
        if self.max_sequences is not None and self.max_sequences < 1:
            raise ValueError(f"max_sequences must be >= 1; got {self.max_sequences}")


def run_perplexity(
    *,
    engine: PerplexityScorer,
    config: PerplexityConfig,
    run_dir: Path,
    run_id: str,
    checkpoint: str,
    extra_config: dict[str, object] | None = None,
) -> int:
    """Score every sequence under ``config.data_path`` and write a run dir.

    Returns the number of records emitted. Caller manages ``engine``'s
    lifecycle. ``extra_config`` is merged into ``config.json`` (same pattern
    as the other evaluators) so the CLI can persist effective sampling
    parameters without duplicating them on ``PerplexityConfig``.
    """
    writer = RunWriter(run_dir)
    writer.write_config(_build_config_payload(config=config, run_id=run_id, checkpoint=checkpoint, extra=extra_config))

    event_id = 0
    with writer:
        for row in _iter_sequences(config.data_path, max_sequences=config.max_sequences):
            token_ids = _tokenize_sequence(row, tokenizer=engine.tokenizer)
            token_logprobs = _score_sequence(token_ids, model=engine.model, device=engine.device)
            writer.append_record(
                PerPerplexityRecord(
                    run_id=run_id,
                    event_id=event_id,
                    evaluator=Evaluator.PERPLEXITY,
                    checkpoint=checkpoint,
                    position_id=row["id"],
                    fen=chess.STARTING_FEN,
                    token_logprobs=token_logprobs,
                )
            )
            event_id += 1
    return event_id


def _build_config_payload(*, config: PerplexityConfig, run_id: str, checkpoint: str, extra: dict[str, object] | None) -> dict[str, object]:
    payload: dict[str, object] = {
        "evaluator": Evaluator.PERPLEXITY.value,
        "run_id": run_id,
        "checkpoint": checkpoint,
    }
    payload.update(asdict(config))
    payload["data_path"] = str(payload["data_path"])
    if extra:
        payload.update(extra)
    return payload


def _iter_sequences(path: Path, *, max_sequences: int | None) -> Iterator[dict[str, Any]]:
    """Yield validated rows from ``path`` (jsonl), capped at ``max_sequences``."""
    yielded = 0
    with path.open(encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"perplexity input row must be a JSON object; got {type(row).__name__}: {row!r}")
            _validate_row(row)
            yield row
            yielded += 1
            if max_sequences is not None and yielded >= max_sequences:
                return


def _validate_row(row: dict[str, Any]) -> None:
    for required in ("id", "perspective", "moves"):
        if required not in row:
            raise ValueError(f"perplexity input row missing required field {required!r}: {row!r}")
    if row["perspective"] not in _PERSPECTIVE_TOKEN_IDS:
        raise ValueError(f"perplexity input row has unknown perspective {row['perspective']!r}; expected 'white' or 'black'")
    if not isinstance(row["moves"], list) or not row["moves"]:
        raise ValueError(f"perplexity input row needs a non-empty 'moves' list: {row!r}")


def _tokenize_sequence(row: dict[str, Any], *, tokenizer: ChessTokenizer) -> list[int]:
    token_ids = [_PERSPECTIVE_TOKEN_IDS[row["perspective"]]]
    for move in row["moves"]:
        move_id = tokenizer.move_to_id(move)
        # ``move_to_id`` falls back to ``<UNK>`` for unknown UCI strings; that
        # would silently degrade the score rather than reject bad input. Fail
        # loudly here with the offending row id so a malformed dataset is
        # immediately diagnosable.
        if move_id == UNK_TOKEN_ID:
            raise ValueError(f"perplexity row {row['id']!r} contains move {move!r} not in tokenizer vocabulary")
        token_ids.append(move_id)
    if row.get("is_draw", False):
        token_ids.append(DRAW_TOKEN_ID)
    return token_ids


def _score_sequence(token_ids: list[int], *, model: _ModelCallable, device: torch.device) -> list[float]:
    """Run a single forward pass on ``token_ids`` and return per-target logprobs.

    Targets are positions ``1..len(token_ids)-1`` of the input — the
    perspective token at position 0 has no preceding context and is excluded
    from the perplexity sum. Returns a list of length ``len(token_ids) - 1``.
    """
    if len(token_ids) < 2:
        return []
    input_ids = torch.tensor([token_ids], device=device)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits[0]  # [seq_len, vocab]
    log_probs = torch.log_softmax(logits[:-1], dim=-1)
    targets = input_ids[0, 1:].unsqueeze(-1)
    gathered = log_probs.gather(-1, targets).squeeze(-1)
    return gathered.tolist()
