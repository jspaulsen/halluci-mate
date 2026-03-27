"""Transform parsed chess games into training sequence strings.

Implements the format from docs/data_format.md:
- Won games: 1 sequence from winner's perspective
- Drawn games: 2 sequences (both perspectives)
"""

from __future__ import annotations

from halluci_mate.chess_tokenizer import BLACK_TOKEN, DRAW_TOKEN, EOS_TOKEN, WHITE_TOKEN

_PERSPECTIVE_TOKENS: dict[str, str] = {
    "white": WHITE_TOKEN,
    "black": BLACK_TOKEN,
}


def game_to_sequences(moves: list[str], outcome: str) -> list[str]:
    """Transform a parsed game into training sequence string(s).

    Args:
        moves: UCI move strings in standard order (white first, alternating).
        outcome: Game outcome - "white", "black", or "draw".

    Returns:
        List of training sequence strings ready for tokenization.

    Raises:
        ValueError: If outcome is not "white", "black", or "draw".
    """
    if outcome == "white":
        return [_format_sequence(moves, "white")]
    if outcome == "black":
        return [_format_sequence(moves, "black")]
    if outcome == "draw":
        return [
            _format_sequence(moves, "white", is_draw=True),
            _format_sequence(moves, "black", is_draw=True),
        ]
    raise ValueError(f"Invalid outcome: {outcome!r}")


def _format_sequence(moves: list[str], perspective: str, is_draw: bool = False) -> str:
    """Format moves into a single training sequence string.

    Args:
        moves: UCI move strings in chronological order.
        perspective: "white" or "black".
        is_draw: Whether to append DRAW token before EOS.

    Returns:
        Space-separated sequence string.
    """
    perspective_token = _PERSPECTIVE_TOKENS[perspective]
    tokens: list[str] = [perspective_token, *moves]
    if is_draw:
        tokens.append(DRAW_TOKEN)
    tokens.append(EOS_TOKEN)
    return " ".join(tokens)
