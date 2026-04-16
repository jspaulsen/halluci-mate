"""Chess LLM inference: load a trained checkpoint and generate the next move."""

from __future__ import annotations

from typing import TYPE_CHECKING

import chess
import torch
from transformers import AutoModelForCausalLM

from halluci_mate.chess_tokenizer import BLACK_TOKEN_ID, WHITE_TOKEN_ID, ChessTokenizer

if TYPE_CHECKING:
    from pathlib import Path


class IllegalMoveError(ValueError):
    """Model emitted a token that is not a legal move in the current position."""


class ChessInferenceEngine:
    """Generate moves from a trained chess LLM checkpoint.

    Conditions on the side-to-move winning (i.e. prepends ``<WHITE>`` or
    ``<BLACK>`` depending on ``board.turn``). Supports constrained decoding
    (mask to legal UCI move tokens) or unconstrained sampling (the raw model).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: ChessTokenizer,
        device: torch.device,
        *,
        constrained: bool = True,
        temperature: float = 0.0,
        top_k: int = 0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.constrained = constrained
        self.temperature = temperature
        self.top_k = top_k

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path | str,
        *,
        constrained: bool = True,
        temperature: float = 0.0,
        top_k: int = 0,
        device: str | None = None,
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

    def generate_move(
        self,
        board: chess.Board,
        *,
        constrained: bool | None = None,
    ) -> chess.Move:
        """Generate the next move for ``board.turn``.

        Move history is taken from ``board.move_stack``. If ``board`` was
        constructed from a mid-game FEN with no pushed moves, the model sees
        only the perspective token as context.

        Args:
            board: current position; ``board.turn`` is the side we generate for
                (the model conditions on that side winning).
            constrained: override the engine's default constrained flag for
                this call. The UCI engine uses this to fall back to constrained
                mode when unconstrained sampling emits an illegal token.

        Raises:
            IllegalMoveError: unconstrained mode produced a token that does not
                parse to a legal UCI move in the current position.
            ValueError: no legal moves available (checkmate or stalemate).
        """
        use_constrained = self.constrained if constrained is None else constrained
        input_ids = self._build_input_ids(board)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
        logits = outputs.logits[0, -1, :]

        if use_constrained:
            legal_ids = _legal_move_token_ids(board, self.tokenizer)
            if not legal_ids:
                raise ValueError(f"No legal moves available in position: {board.fen()}")
            mask = torch.full_like(logits, float("-inf"))
            mask[torch.tensor(legal_ids, device=logits.device)] = 0.0
            logits = logits + mask

        token_id = _sample(logits, temperature=self.temperature, top_k=self.top_k)
        token = self.tokenizer.convert_ids_to_tokens(int(token_id))
        move = _token_to_legal_move(token, board)
        if move is None:
            raise IllegalMoveError(f"Model produced token {token!r} which is not a legal move in position {board.fen()}")
        return move

    def _build_input_ids(self, board: chess.Board) -> torch.Tensor:
        """Build the prompt: ``<perspective> move1 move2 ...`` as token ids.

        Moves come from ``board.move_stack`` — the single source of truth for
        the game history; callers cannot pass a history that disagrees with
        the board state.
        """
        perspective_id = WHITE_TOKEN_ID if board.turn == chess.WHITE else BLACK_TOKEN_ID
        vocab = self.tokenizer.get_vocab()
        history_ids = [vocab.get(move.uci(), self.tokenizer.unk_token_id) for move in board.move_stack]
        return torch.tensor([[perspective_id, *history_ids]], dtype=torch.long, device=self.device)


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _legal_move_token_ids(board: chess.Board, tokenizer: ChessTokenizer) -> list[int]:
    """Return the vocab ids of every legal UCI move in ``board``."""
    vocab = tokenizer.get_vocab()
    return [tok_id for move in board.legal_moves if (tok_id := vocab.get(move.uci())) is not None]


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


def _token_to_legal_move(token: str, board: chess.Board) -> chess.Move | None:
    """Parse ``token`` as UCI and return the move iff it is legal in ``board``."""
    try:
        move = chess.Move.from_uci(token)
    except (chess.InvalidMoveError, ValueError):
        return None
    return move if move in board.legal_moves else None
