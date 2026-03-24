"""
Custom HuggingFace-compatible tokenizer for UCI chess moves.

Vocabulary:
- Special tokens: <PAD>, <UNK>, <EOS>, <WHITE>, <BLACK>, <DRAW> (indices 0-5)
- Geometric UCI moves: ~1,792 tokens (all valid from-to square combinations)
- Total: ~1,798 tokens
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import chess
from transformers import PreTrainedTokenizer

if TYPE_CHECKING:
    from collections.abc import Sequence

# Special token definitions
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
EOS_TOKEN = "<EOS>"
WHITE_TOKEN = "<WHITE>"
BLACK_TOKEN = "<BLACK>"
DRAW_TOKEN = "<DRAW>"

# Special token indices
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
EOS_TOKEN_ID = 2
WHITE_TOKEN_ID = 3
BLACK_TOKEN_ID = 4
DRAW_TOKEN_ID = 5


def _generate_move_vocabulary() -> list[str]:
    """Generate all valid UCI move tokens.

    Returns sorted list of geometric UCI move strings (Queen + Knight patterns).
    """
    moves: set[str] = set()

    for src in chess.SQUARES:
        for dst in chess.SQUARES:
            if src == dst:
                continue

            src_rank = chess.square_rank(src)
            dst_rank = chess.square_rank(dst)
            src_file = chess.square_file(src)
            dst_file = chess.square_file(dst)

            dist_rank = abs(src_rank - dst_rank)
            dist_file = abs(src_file - dst_file)

            is_diagonal = dist_rank == dist_file
            is_straight = dist_rank == 0 or dist_file == 0
            is_knight = (dist_rank == 2 and dist_file == 1) or (dist_rank == 1 and dist_file == 2)

            if is_diagonal or is_straight or is_knight:
                move_uci = chess.Move(src, dst).uci()
                moves.add(move_uci)

    return sorted(moves)


class ChessTokenizer(PreTrainedTokenizer):
    """HuggingFace-compatible tokenizer for UCI chess notation.

    Sequence format:
    - Won game: <WHITE> e2e4 e7e5 g1f3 ... <EOS> (white won)
    - Won game: <BLACK> e2e4 e7e5 g1f3 ... <EOS> (black won)
    - Drawn game: <WHITE> e2e4 d7d5 ... <DRAW> <EOS>
    - Drawn game: <BLACK> e2e4 d7d5 ... <DRAW> <EOS>

    Moves are always in standard order (white first, alternating).
    The leading token indicates perspective ("play as this color").
    For wins, that color won. For draws, both perspectives are trained.
    """

    # Defaults to these values
    # model_input_names: ClassVar[list[str]] = ["input_ids", "attention_mask"]

    def __init__(self, **kwargs) -> None:
        # Build vocabulary: special tokens first, then moves
        self._special_tokens = [PAD_TOKEN, UNK_TOKEN, EOS_TOKEN, WHITE_TOKEN, BLACK_TOKEN, DRAW_TOKEN]
        self._move_tokens = _generate_move_vocabulary()
        self._vocab = self._special_tokens + self._move_tokens

        # Build lookup dictionaries
        self._token_to_id: dict[str, int] = {token: idx for idx, token in enumerate(self._vocab)}
        self._id_to_token: dict[int, str] = {idx: token for idx, token in enumerate(self._vocab)}

        # Initialize parent class with special tokens
        super().__init__(
            pad_token=PAD_TOKEN,
            unk_token=UNK_TOKEN,
            eos_token=EOS_TOKEN,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self._vocab)

    def get_vocab(self) -> dict[str, int]:
        """Return the vocabulary as a dictionary of token to index."""
        return self._token_to_id.copy()

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        """Split text into tokens on whitespace."""
        return text.split()

    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token string to its vocabulary index."""
        return self._token_to_id.get(token, UNK_TOKEN_ID)

    def _convert_id_to_token(self, index: int) -> str:
        """Convert a vocabulary index to its token string."""
        return self._id_to_token.get(index, UNK_TOKEN)

    def convert_tokens_to_string(self, tokens: Sequence[str]) -> str:
        """Convert a sequence of tokens to a single string."""
        return " ".join(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        """Save the vocabulary to a JSON file for checkpointing."""
        save_path = Path(save_directory)
        if not save_path.is_dir():
            save_path = save_path.parent

        vocab_file = save_path / f"{filename_prefix + '-' if filename_prefix else ''}vocab.json"
        with open(vocab_file, "w") as f:
            json.dump(self._token_to_id, f, indent=2)

        return (str(vocab_file),)
