"""Unit tests for halluci_mate.uci_engine."""

from __future__ import annotations

import io
import types

import chess
import torch

from halluci_mate.chess_tokenizer import ChessTokenizer
from halluci_mate.inference import ChessInferenceEngine
from halluci_mate.uci_engine import UciEngine

CPU = torch.device("cpu")


class FakeModel:
    """Uniform-logits stand-in; constrained decoding will produce a legal move."""

    def __init__(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size

    def __call__(self, input_ids: torch.Tensor, **_: object) -> types.SimpleNamespace:
        batch, seq = input_ids.shape
        return types.SimpleNamespace(logits=torch.zeros(batch, seq, self._vocab_size))


def _make_engine() -> ChessInferenceEngine:
    tokenizer = ChessTokenizer()
    return ChessInferenceEngine(
        model=FakeModel(tokenizer.vocab_size),
        tokenizer=tokenizer,
        device=CPU,
        constrained=True,
    )


def _make_uci(stdin_text: str = "") -> tuple[UciEngine, io.StringIO, io.StringIO]:
    stdin = io.StringIO(stdin_text)
    stdout = io.StringIO()
    stderr = io.StringIO()
    return UciEngine(engine=_make_engine(), stdin=stdin, stdout=stdout, stderr=stderr), stdout, stderr


class TestBasicCommands:
    def test_uci_identifies_engine(self) -> None:
        uci, stdout, _ = _make_uci()
        uci.handle("uci")
        out = stdout.getvalue()
        assert "id name halluci-mate" in out
        assert "uciok" in out

    def test_isready_responds_readyok(self) -> None:
        uci, stdout, _ = _make_uci()
        uci.handle("isready")
        assert stdout.getvalue().strip() == "readyok"

    def test_quit_returns_true(self) -> None:
        uci, _, _ = _make_uci()
        assert uci.handle("quit") is True

    def test_unknown_command_ignored(self) -> None:
        uci, stdout, _ = _make_uci()
        assert uci.handle("debug on") is False
        assert stdout.getvalue() == ""


class TestPosition:
    def test_ucinewgame_resets_state(self) -> None:
        uci, _, _ = _make_uci()
        uci.handle("position startpos moves e2e4 e7e5")
        uci.handle("ucinewgame")
        assert uci.board == chess.Board()

    def test_position_startpos(self) -> None:
        uci, _, _ = _make_uci()
        uci.handle("position startpos")
        assert uci.board == chess.Board()

    def test_position_startpos_with_moves(self) -> None:
        uci, _, _ = _make_uci()
        uci.handle("position startpos moves e2e4 e7e5 g1f3")
        expected = chess.Board()
        for uci_str in ("e2e4", "e7e5", "g1f3"):
            expected.push(chess.Move.from_uci(uci_str))
        assert uci.board == expected
        assert [m.uci() for m in uci.board.move_stack] == ["e2e4", "e7e5", "g1f3"]

    def test_position_fen(self) -> None:
        uci, _, _ = _make_uci()
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        uci.handle(f"position fen {fen}")
        assert uci.board == chess.Board(fen)

    def test_position_fen_with_moves(self) -> None:
        uci, _, _ = _make_uci()
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        uci.handle(f"position fen {fen} moves e7e5")
        expected = chess.Board(fen)
        expected.push(chess.Move.from_uci("e7e5"))
        assert uci.board == expected


class TestMalformedInput:
    def test_malformed_move_logged_and_skipped(self) -> None:
        uci, _, stderr = _make_uci()
        uci.handle("position startpos moves e2e4 garbage e7e5")
        # Good moves before and after "garbage" should still be applied.
        expected = chess.Board()
        for uci_str in ("e2e4", "e7e5"):
            expected.push(chess.Move.from_uci(uci_str))
        assert uci.board == expected
        assert "garbage" in stderr.getvalue()


class TestGo:
    def test_go_emits_legal_bestmove(self) -> None:
        uci, stdout, _ = _make_uci()
        uci.handle("position startpos")
        uci.handle("go")
        output = stdout.getvalue().strip()
        assert output.startswith("bestmove ")
        move_uci = output.split()[1]
        assert chess.Move.from_uci(move_uci) in chess.Board().legal_moves


class TestRunLoop:
    def test_run_consumes_stream_until_quit(self) -> None:
        script = "uci\nisready\nposition startpos\ngo\nquit\n"
        uci, stdout, _ = _make_uci(script)
        uci.run()
        out = stdout.getvalue()
        assert "uciok" in out
        assert "readyok" in out
        assert "bestmove " in out
