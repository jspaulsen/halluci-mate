from dataclasses import dataclass, field
from enum import IntEnum

import chess

from halluci_mate.chess_tokenizer import BLACK_TOKEN_ID, WHITE_TOKEN_ID, ChessTokenizer


class Perspective(IntEnum):
    WHITE = WHITE_TOKEN_ID
    BLACK = BLACK_TOKEN_ID


@dataclass(frozen=True)
class KVCacheState:
    """Opaque per-game KV cache snapshot. ``cache`` is a transformers cache object
    (e.g. ``DynamicCache``); typed as ``object`` so ``game.py`` stays free of
    transformers imports. ``tokens`` is the full token prefix the cache was
    built from — the engine compares it against the current token list to
    detect history rewrites before taking the fast path.

    Memory: ``cache`` holds device tensors proportional to ``len(tokens)``.
    Long-lived ``Game`` instances (e.g. reused across games in an eval loop)
    should call :meth:`Game.reset_cache` on game end to release the tensors."""

    cache: object
    tokens: tuple[int, ...]


@dataclass
class Game:
    board: chess.Board
    perspective: Perspective
    cache: KVCacheState | None = field(default=None, repr=False, compare=False)

    def tokenize(self, tokenizer: ChessTokenizer) -> list[int]:
        tokens: list[int] = [self.perspective]

        for move in self.board.move_stack:
            move_token_id = tokenizer.move_to_id(move.uci())
            tokens.append(move_token_id)

        return tokens

    def is_legal_move(self, move: chess.Move) -> bool:
        return move in self.board.legal_moves

    def push_move(self, move: chess.Move) -> None:
        self.board.push(move)

    def reset_cache(self) -> None:
        """Escape hatch for same-length prefix swaps the engine's prefix check
        cannot detect (e.g. ``board.set_fen`` to a position whose token count
        matches the cached length). Also the right call on game end to release
        cache tensors held by long-lived ``Game`` instances."""
        self.cache = None
