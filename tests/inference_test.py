"""Unit tests for halluci_mate.inference."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING

import chess
import pytest
import torch

from halluci_mate.chess_tokenizer import BLACK_TOKEN_ID, EOS_TOKEN_ID, WHITE_TOKEN_ID, ChessTokenizer
from halluci_mate.game import Game, Perspective
from halluci_mate.inference import ChessInferenceEngine, GameOverError, IllegalMoveError

if TYPE_CHECKING:
    from collections.abc import Sequence

CPU = torch.device("cpu")
# Fool's mate: the shortest possible checkmate sequence. Reaches a no-legal-moves position.
FOOLS_MATE = ("f2f3", "e7e5", "g2g4", "d8h4")


class FakeModel:
    """Stand-in for AutoModelForCausalLM that returns controllable logits."""

    def __init__(
        self,
        vocab_size: int,
        forced_token_id: int | None = None,
        boost_ids: Sequence[int] | None = None,
    ) -> None:
        self._vocab_size = vocab_size
        self._forced = forced_token_id
        self._boost = list(boost_ids) if boost_ids else []
        self.last_input_ids: torch.Tensor | None = None

    def __call__(self, input_ids: torch.Tensor, **_: object) -> types.SimpleNamespace:
        self.last_input_ids = input_ids
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, self._vocab_size)
        if self._forced is not None:
            logits[:, -1, self._forced] = 100.0
        for tok_id in self._boost:
            logits[:, -1, tok_id] = 10.0
        return types.SimpleNamespace(logits=logits)


def _engine(
    model: FakeModel,
    *,
    constrained: bool = True,
    temperature: float = 0.0,
    top_k: int = 0,
) -> ChessInferenceEngine:
    tokenizer = ChessTokenizer()
    return ChessInferenceEngine(
        model=model,  # type: ignore[arg-type]
        tokenizer=tokenizer,
        device=CPU,
        constrained=constrained,
        temperature=temperature,
        top_k=top_k,
    )


def _board_after(moves: Sequence[str]) -> chess.Board:
    board = chess.Board()
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
    return board


def _game(board: chess.Board, perspective: Perspective = Perspective.WHITE) -> Game:
    return Game(board=board, perspective=perspective)


class TestPredictConstrained:
    def test_startpos_returns_legal_move(self) -> None:
        tokenizer = ChessTokenizer()
        engine = _engine(FakeModel(tokenizer.vocab_size))
        game = _game(chess.Board())
        move = engine.predict(game)
        assert move in game.board.legal_moves

    def test_illegal_token_boost_ignored_when_constrained(self) -> None:
        tokenizer = ChessTokenizer()
        # Force a token that is geometrically valid UCI but not legal from startpos.
        illegal_uci = "e2e5"
        illegal_id = tokenizer.get_vocab()[illegal_uci]
        engine = _engine(FakeModel(tokenizer.vocab_size, forced_token_id=illegal_id))
        move = engine.predict(_game(chess.Board()))
        assert move.uci() != illegal_uci
        assert move in chess.Board().legal_moves

    def test_raises_when_no_legal_moves(self) -> None:
        tokenizer = ChessTokenizer()
        engine = _engine(FakeModel(tokenizer.vocab_size))
        game = _game(_board_after(FOOLS_MATE), perspective=Perspective.BLACK)
        with pytest.raises(GameOverError, match="No legal moves"):
            engine.predict(game)


class TestPredictUnconstrained:
    def test_raises_on_illegal_model_output(self) -> None:
        tokenizer = ChessTokenizer()
        engine = _engine(
            FakeModel(tokenizer.vocab_size, forced_token_id=EOS_TOKEN_ID),
            constrained=False,
        )
        with pytest.raises(IllegalMoveError):
            engine.predict(_game(chess.Board()))

    def test_returns_move_when_model_picks_legal(self) -> None:
        tokenizer = ChessTokenizer()
        legal_id = tokenizer.get_vocab()["e2e4"]
        engine = _engine(
            FakeModel(tokenizer.vocab_size, forced_token_id=legal_id),
            constrained=False,
        )
        move = engine.predict(_game(chess.Board()))
        assert move == chess.Move.from_uci("e2e4")

    def test_constrained_override_forces_legality(self) -> None:
        tokenizer = ChessTokenizer()
        engine = _engine(
            FakeModel(tokenizer.vocab_size, forced_token_id=EOS_TOKEN_ID),
            constrained=False,
        )
        move = engine.predict(_game(chess.Board()), constrained=True)
        assert move in chess.Board().legal_moves


class TestConditioning:
    def test_prepends_white_perspective_token(self) -> None:
        tokenizer = ChessTokenizer()
        model = FakeModel(tokenizer.vocab_size)
        engine = _engine(model)
        engine.predict(_game(chess.Board(), perspective=Perspective.WHITE))
        assert model.last_input_ids is not None
        assert int(model.last_input_ids[0, 0]) == WHITE_TOKEN_ID

    def test_prepends_black_perspective_token(self) -> None:
        tokenizer = ChessTokenizer()
        model = FakeModel(tokenizer.vocab_size)
        engine = _engine(model)
        board = _board_after(["e2e4"])
        engine.predict(_game(board, perspective=Perspective.BLACK))
        assert model.last_input_ids is not None
        assert int(model.last_input_ids[0, 0]) == BLACK_TOKEN_ID
        assert int(model.last_input_ids[0, 1]) == tokenizer.get_vocab()["e2e4"]

    def test_history_tokens_come_from_board_move_stack(self) -> None:
        tokenizer = ChessTokenizer()
        model = FakeModel(tokenizer.vocab_size)
        engine = _engine(model)
        played = ["e2e4", "e7e5", "g1f3"]
        board = _board_after(played)
        engine.predict(_game(board, perspective=Perspective.WHITE))
        vocab = tokenizer.get_vocab()
        assert model.last_input_ids is not None
        # [perspective, e2e4, e7e5, g1f3]
        assert [int(x) for x in model.last_input_ids[0, 1:]] == [vocab[uci] for uci in played]
