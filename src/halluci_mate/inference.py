"""Chess LLM inference: load a trained checkpoint and generate the next move."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import chess
import torch
from transformers import AutoModelForCausalLM

from halluci_mate.chess_tokenizer import ChessTokenizer
from halluci_mate.eval.records import TopKEntry
from halluci_mate.game import KVCacheState

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedModel

    from halluci_mate.game import Game


class IllegalMoveError(ValueError):
    """Model emitted a token that is not a legal move in the current position."""


class GameOverError(ValueError):
    """Caller invoked ``predict`` on a position with no legal moves (checkmate, stalemate)."""


@dataclass(frozen=True)
class MovePrediction:
    """Per-position inference output, including the data the eval harness needs.

    ``played_move`` is ``None`` when constrained decoding is disabled and the
    sampled token does not parse as a legal move; in that case the caller can
    still inspect ``model_move_uci`` and the rest of the metadata.

    ``raw_sample_move_uci`` and ``raw_sample_legal`` always describe the
    unconstrained top-1 of the raw logits, regardless of ``mask_used`` —
    these are what Phase 2 legality DPO pairs against ``model_move``.

    ``model_top_k`` comes from the *sampled-from* distribution: post-mask
    when ``mask_used`` is true, raw otherwise. Empty list when the caller
    passes ``record_top_k=0``.
    """

    played_move: chess.Move | None
    model_move_uci: str
    raw_sample_move_uci: str
    raw_sample_legal: bool
    model_top_k: list[TopKEntry]
    mask_used: bool


class Predictor(Protocol):
    """Structural type for objects that can produce a ``MovePrediction``.

    Lets the eval harness (and tests) accept any object exposing
    ``predict_with_metadata`` rather than the concrete ``ChessInferenceEngine``
    — the only method the harness actually calls.
    """

    def predict_with_metadata(
        self,
        game: Game,
        *,
        constrained: bool | None = None,
        record_top_k: int = 5,
    ) -> MovePrediction: ...


class ChessInferenceEngine:
    """Generate moves from a trained chess LLM checkpoint.

    Conditions on ``Game.perspective`` — the color the caller wants to win as —
    which is independent of whose turn it is. ``Game.tokenize`` prepends the
    ``<WHITE>`` or ``<BLACK>`` token accordingly. Supports constrained decoding
    (mask to legal UCI move tokens) or unconstrained sampling (the raw model).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: ChessTokenizer,
        device: torch.device,
        *,
        constrained: bool = True,
        temperature: float = 0.0,
        top_k: int = 0,
    ) -> None:
        self.model = model
        self.tokenizer: ChessTokenizer = tokenizer
        self.device: torch.device = device
        self.constrained: bool = constrained
        self.temperature: float = temperature
        self.top_k: int = top_k

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path | str,
        *,
        constrained: bool = True,
        temperature: float = 0.0,
        top_k: int = 0,
        device: str | torch.device | None = None,
    ) -> ChessInferenceEngine:
        """Load an engine from a Hugging Face Trainer checkpoint directory."""
        resolved_device = torch.device(device) if device else _default_device()
        tokenizer = ChessTokenizer()

        model = AutoModelForCausalLM.from_pretrained(str(checkpoint), device_map=str(resolved_device))
        model.eval()

        return cls(
            model=model,
            tokenizer=tokenizer,
            device=resolved_device,
            constrained=constrained,
            temperature=temperature,
            top_k=top_k,
        )

    @torch.inference_mode()
    def predict(
        self,
        game: Game,
        constrained: bool | None = None,
    ) -> chess.Move:
        """Sample the next move; raise ``IllegalMoveError`` if the sample is not legal."""
        prediction = self.predict_with_metadata(game, constrained=constrained, record_top_k=0)
        if prediction.played_move is not None:
            return prediction.played_move
        # Reparse the rejected token so the error message distinguishes
        # "not a valid UCI" (unknown vocab) from "valid UCI but illegal here".
        token = prediction.model_move_uci
        if _parse_uci(token) is None:
            raise IllegalMoveError(f"Model produced token {token!r} which is not a valid UCI move")
        raise IllegalMoveError(f"Model produced token {token!r} which is not a legal move in position {game.board.fen()}")

    @torch.inference_mode()
    def predict_with_metadata(
        self,
        game: Game,
        *,
        constrained: bool | None = None,
        record_top_k: int = 5,
    ) -> MovePrediction:
        """Sample the next move and return both played and unconstrained-top-1 metadata.

        Set ``record_top_k`` to 0 to skip top-K extraction (used by the
        legacy ``predict`` wrapper).
        """
        use_constrained = self.constrained if constrained is None else constrained

        raw_logits = self._forward_logits(game)

        raw_top_id = int(torch.argmax(raw_logits).item())
        raw_top_token = self.tokenizer.convert_ids_to_tokens(raw_top_id)
        raw_sample_legal = _parse_legal(raw_top_token, game.board) is not None

        if use_constrained:
            legal_moves = list(game.board.legal_moves)
            if not legal_moves:
                raise GameOverError(f"No legal moves available in position: {game.board.fen()}")
            legal_token_ids = [self.tokenizer.move_to_id(move.uci()) for move in legal_moves]
            mask = torch.full_like(raw_logits, float("-inf"))
            mask[torch.tensor(legal_token_ids, device=raw_logits.device)] = 0.0
            play_logits = raw_logits + mask
        else:
            play_logits = raw_logits

        played_token_id = _sample(play_logits, temperature=self.temperature, top_k=self.top_k)
        played_token = self.tokenizer.convert_ids_to_tokens(int(played_token_id))
        played_move = _parse_legal(played_token, game.board)

        model_top_k = _extract_top_k(play_logits, self.tokenizer, record_top_k) if record_top_k > 0 else []

        return MovePrediction(
            played_move=played_move,
            model_move_uci=played_token,
            raw_sample_move_uci=raw_top_token,
            raw_sample_legal=raw_sample_legal,
            model_top_k=model_top_k,
            mask_used=use_constrained,
        )

    def _forward_logits(self, game: Game) -> torch.Tensor:
        """Run a forward pass for ``game`` and return the next-token logits.

        Updates ``game.cache`` after the forward, before any sampling, so a
        downstream sampling failure (illegal move on retry, etc.) doesn't
        leave the cache in a stale state — the cache reflects the tokens
        actually forwarded.
        """
        tokens = game.tokenize(self.tokenizer)
        cached = game.cache
        cached_len = len(cached.tokens) if cached is not None else 0
        # Fast path requires strictly-new tokens and a matching prefix. Prefix
        # equality catches pop+push-different-moves rewrites that would feed
        # stale KVs. Equal length falls through to the slow path.
        # TODO(perf): on an equal-length cache hit (predict called twice with no
        # new move, e.g. post-IllegalMoveError retry or alternative sampling) we
        # could cache last-token logits and skip the forward entirely. Minor
        # perf; not a hot path today.
        if cached is not None and cached_len < len(tokens) and tuple(tokens[:cached_len]) == cached.tokens:
            new_ids = torch.tensor([tokens[cached_len:]], device=self.device)
            cache_position = torch.arange(cached_len, len(tokens), device=self.device)
            outputs = self.model(
                input_ids=new_ids,
                past_key_values=cached.cache,
                use_cache=True,
                cache_position=cache_position,
            )
        else:
            all_ids = torch.tensor([tokens], device=self.device)
            outputs = self.model(input_ids=all_ids, use_cache=True)

        game.cache = KVCacheState(cache=outputs.past_key_values, tokens=tuple(tokens))
        return outputs.logits[0, -1, :]


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sample(logits: torch.Tensor, *, temperature: float, top_k: int) -> torch.Tensor:
    """Sample a token id from 1D ``logits`` with optional temperature and top-k."""
    if temperature <= 0.0:
        return torch.argmax(logits)

    scaled = logits / temperature
    if top_k > 0:
        k = min(top_k, scaled.size(-1))
        top_vals, top_idx = torch.topk(scaled, k=k)
        chosen = torch.multinomial(torch.softmax(top_vals, dim=-1), num_samples=1)
        return top_idx[chosen].squeeze()
    return torch.multinomial(torch.softmax(scaled, dim=-1), num_samples=1).squeeze()


def _parse_uci(token: str) -> chess.Move | None:
    """Return the parsed move if ``token`` is valid UCI, else ``None``."""
    try:
        return chess.Move.from_uci(token)
    except chess.InvalidMoveError:
        return None


def _parse_legal(token: str, board: chess.Board) -> chess.Move | None:
    """Return the parsed move if ``token`` is legal in ``board``, else ``None``.

    Used for the *played* move — illegal here is recoverable (the eval harness
    wants to capture it); raising would force every caller to wrap in try/except.
    """
    move = _parse_uci(token)
    if move is None or move not in board.legal_moves:
        return None
    return move


def _extract_top_k(logits: torch.Tensor, tokenizer: ChessTokenizer, k: int) -> list[TopKEntry]:
    """Top-K of ``log_softmax(logits)`` mapped to ``TopKEntry`` records."""
    log_probs = torch.log_softmax(logits, dim=-1)
    capped_k = min(k, log_probs.size(-1))
    top_vals, top_idx = torch.topk(log_probs, k=capped_k)
    return [TopKEntry(move=tokenizer.convert_ids_to_tokens(int(idx)), logprob=float(val)) for val, idx in zip(top_vals.tolist(), top_idx.tolist(), strict=True)]
