"""Chess LLM inference: load a trained checkpoint and generate the next move."""

from __future__ import annotations

from typing import TYPE_CHECKING

import chess
import torch
from transformers import AutoModelForCausalLM

from halluci_mate.chess_tokenizer import ChessTokenizer

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedModel

    from halluci_mate.game import Game


class IllegalMoveError(ValueError):
    """Model emitted a token that is not a legal move in the current position."""


class GameOverError(ValueError):
    """Caller invoked ``predict`` on a position with no legal moves (checkmate, stalemate)."""


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
        # TODO(eval-harness): docs/eval_harness.md requires exposing
        # `raw_sample_legal` (did the unconstrained top-1 land on a legal move?)
        # and top-k candidates alongside the played move — needed for Phase 2
        # legality-DPO seed pairs. Extend the return shape when the eval harness
        # lands; leaving `-> Move` alone for now since nothing else consumes it.
        use_constrained = self.constrained if constrained is None else constrained

        tokens = game.tokenize(self.tokenizer)
        # TODO(perf): re-encodes the full history every call (O(n²) across a game).
        # Qwen3 supports KV caching via past_key_values + use_cache=True, but we'd
        # need to maintain a cache on the engine/Game and feed only the tokens added
        # since the last call. Hot path for the eval harness once DPO scale kicks in.
        outputs = self.model(torch.tensor([tokens], device=self.device))
        logits = outputs.logits[0, -1, :]

        if use_constrained:
            legal_moves = list(game.board.legal_moves)

            if not legal_moves:
                raise GameOverError(f"No legal moves available in position: {game.board.fen()}")

            legal_token_ids = [self.tokenizer.move_to_id(move.uci()) for move in legal_moves]

            mask = torch.full_like(logits, float("-inf"))
            mask[torch.tensor(legal_token_ids, device=logits.device)] = 0.0
            logits = logits + mask

        next_token_id = _sample(logits, temperature=self.temperature, top_k=self.top_k)
        next_token = self.tokenizer.convert_ids_to_tokens(int(next_token_id))

        try:
            move = chess.Move.from_uci(next_token)
        except chess.InvalidMoveError:
            raise IllegalMoveError(f"Model produced token {next_token!r} which is not a valid UCI move") from None

        if not game.is_legal_move(move):
            raise IllegalMoveError(f"Model produced token {next_token!r} which is not a legal move in position {game.board.fen()}")

        return move


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
