"""Parse PGN movetext into UCI move lists using python-chess."""

from __future__ import annotations

import io

import chess.pgn

# Promotion moves produce 5-char UCI (e.g. e7e8q). The tokenizer only has 4-char
# geometric moves, so we strip the promotion suffix until the tokenizer is extended.
_UCI_MOVE_LENGTH = 4

_RESULT_TO_OUTCOME: dict[str, str] = {
    "1-0": "white",
    "0-1": "black",
    "1/2-1/2": "draw",
}


def parse_movetext(movetext: str) -> list[str]:
    """Parse PGN movetext into a list of UCI move strings.

    Handles clock annotations, move numbers, comments, and result markers
    via chess.pgn.read_game().

    Args:
        movetext: Raw PGN movetext string from Lichess dataset.

    Returns:
        List of UCI move strings, e.g. ["e2e4", "e7e5", "g1f3"].

    Raises:
        ValueError: If movetext is empty or unparseable.
    """
    if not movetext or not movetext.strip():
        raise ValueError("Empty movetext")

    pgn_string = f'[Event "?"]\n[Result "*"]\n\n{movetext}'
    game = chess.pgn.read_game(io.StringIO(pgn_string))

    if game is None:
        raise ValueError(f"Failed to parse movetext: {movetext[:100]}")

    moves = [move.uci()[:_UCI_MOVE_LENGTH] for move in game.mainline_moves()]

    if not moves:
        raise ValueError(f"No moves found in movetext: {movetext[:100]}")

    return moves


def parse_result(result: str) -> str:
    """Convert PGN result string to outcome label.

    Args:
        result: PGN result string ("1-0", "0-1", "1/2-1/2").

    Returns:
        One of "white", "black", or "draw".

    Raises:
        ValueError: If result is not a recognized PGN result.
    """
    outcome = _RESULT_TO_OUTCOME.get(result)
    if outcome is None:
        raise ValueError(f"Unrecognized result: {result!r}")
    return outcome
