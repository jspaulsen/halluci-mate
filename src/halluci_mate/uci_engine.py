"""Minimal UCI protocol loop for the halluci-mate inference engine.

Supports only the commands needed for basic play against a GUI or lichess-bot:
uci, isready, ucinewgame, position, go, quit. Time controls are ignored — ``go``
always returns immediately with the model's chosen move.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TextIO

import chess

from halluci_mate.inference import IllegalMoveError

if TYPE_CHECKING:
    from halluci_mate.inference import ChessInferenceEngine

ENGINE_NAME = "halluci-mate"
ENGINE_AUTHOR = "halluci-mate contributors"

# A FEN string always has exactly 6 whitespace-separated fields.
_FEN_FIELD_COUNT = 6


@dataclass
class UciEngine:
    """Drives a ``ChessInferenceEngine`` over the UCI protocol on text streams."""

    engine: ChessInferenceEngine
    stdin: TextIO = field(default_factory=lambda: sys.stdin)
    stdout: TextIO = field(default_factory=lambda: sys.stdout)
    stderr: TextIO = field(default_factory=lambda: sys.stderr)
    _board: chess.Board = field(default_factory=chess.Board)

    @property
    def board(self) -> chess.Board:
        """The current board state. Read-only view for tests and callers."""
        return self._board

    def run(self) -> None:
        """Read UCI commands from ``stdin`` until ``quit``."""
        for raw_line in self.stdin:
            line = raw_line.strip()
            if not line:
                continue
            if self.handle(line):
                return

    def handle(self, line: str) -> bool:
        """Process one UCI command. Returns True if the engine should quit."""
        parts = line.split()
        cmd, args = parts[0], parts[1:]

        if cmd == "uci":
            self._write(f"id name {ENGINE_NAME}")
            self._write(f"id author {ENGINE_AUTHOR}")
            self._write("uciok")
        elif cmd == "isready":
            self._write("readyok")
        elif cmd == "ucinewgame":
            self._board = chess.Board()
        elif cmd == "position":
            self._set_position(args)
        elif cmd == "go":
            self._go()
        elif cmd == "quit":
            return True
        # Unknown commands are ignored per UCI convention.
        return False

    def _write(self, text: str) -> None:
        self.stdout.write(f"{text}\n")
        self.stdout.flush()

    def _set_position(self, args: list[str]) -> None:
        if not args:
            return

        if args[0] == "startpos":
            self._board = chess.Board()
            rest = args[1:]
        elif args[0] == "fen":
            fen_tokens = args[1 : 1 + _FEN_FIELD_COUNT]
            self._board = chess.Board(" ".join(fen_tokens))
            rest = args[1 + _FEN_FIELD_COUNT :]
        else:
            return

        if rest and rest[0] == "moves":
            for uci in rest[1:]:
                self._apply_move(uci)

    def _apply_move(self, uci: str) -> None:
        # Moves arrive over stdin from an external GUI — a system boundary.
        # Log and skip anything malformed rather than crashing the engine.
        try:
            move = chess.Move.from_uci(uci)
            self._board.push(move)
        except (chess.InvalidMoveError, chess.IllegalMoveError, AssertionError) as exc:
            self.stderr.write(f"info string ignoring malformed move {uci!r}: {exc}\n")
            self.stderr.flush()

    def _go(self) -> None:
        try:
            move = self.engine.generate_move(self._board)
        except IllegalMoveError as exc:
            self.stderr.write(f"info string unconstrained sampling produced illegal move ({exc}); falling back to constrained\n")
            self.stderr.flush()
            move = self.engine.generate_move(self._board, constrained=True)
        self._write(f"bestmove {move.uci()}")
