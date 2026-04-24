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
    transformers imports. ``token_count`` is the length of the token prefix the
    cache was built from."""

    cache: object
    token_count: int


@dataclass
class Game:
    board: chess.Board
    perspective: Perspective
    _kv_cache: KVCacheState | None = field(default=None, repr=False, compare=False)

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
        """Drop the KV cache. Call after any board mutation that is NOT a plain
        append of one move via ``push_move`` — ``board.set_fen``, ``board.reset``,
        ``board.pop``, or any direct ``board.push`` that rewrites history.
        Forgetting to call this produces silently-wrong logits on the next
        ``predict``, since the fast path only checks token-count monotonicity."""
        self._kv_cache = None
