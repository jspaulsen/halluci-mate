from dataclasses import dataclass
from enum import Enum
from typing import Self

import chess

from halluci_mate.chess_tokenizer import BLACK_TOKEN_ID, WHITE_TOKEN_ID, ChessTokenizer


class Perspective(Enum):
    WHITE = WHITE_TOKEN_ID
    BLACK = BLACK_TOKEN_ID


@dataclass
class Game:
    board: chess.Board
    perspective: Perspective

    def tokenize(self, tokenizer: ChessTokenizer) -> list[int]:
        tokens = [self.perspective.value]

        for move in self.board.move_stack:
            move_token_id = tokenizer.move_to_id(move.uci())
            tokens.append(move_token_id)

        return tokens

    def is_legal_move(self, move: chess.Move) -> bool:
        return move in self.board.legal_moves

    def push_move(self, move: chess.Move) -> Self:
        self.board.push(move)

        return self
