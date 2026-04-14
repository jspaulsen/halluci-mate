"""Tests for PGN movetext parsing and result conversion."""

from __future__ import annotations

import pytest

from halluci_mate.pgn_to_uci import parse_movetext, parse_result

# --- parse_movetext ---


SIMPLE_MOVETEXT = "1. e4 e5 2. Nf3 Nc6"
SIMPLE_EXPECTED = ["e2e4", "e7e5", "g1f3", "b8c6"]

CLOCK_MOVETEXT = "1. e4 { [%clk 0:03:00] } 1... d5 { [%clk 0:03:00] } 2. exd5 { [%clk 0:02:58] }"
CLOCK_EXPECTED = ["e2e4", "d7d5", "e4d5"]

WITH_RESULT_MOVETEXT = "1. e4 e5 2. Nf3 Nc6 1-0"
WITH_RESULT_EXPECTED = ["e2e4", "e7e5", "g1f3", "b8c6"]


def test_simple_movetext() -> None:
    assert parse_movetext(SIMPLE_MOVETEXT) == SIMPLE_EXPECTED


def test_clock_annotations_stripped() -> None:
    assert parse_movetext(CLOCK_MOVETEXT) == CLOCK_EXPECTED


def test_result_marker_stripped() -> None:
    assert parse_movetext(WITH_RESULT_MOVETEXT) == WITH_RESULT_EXPECTED


def test_kingside_castling() -> None:
    movetext = "1. e4 e5 2. Nf3 Nf6 3. Bc4 Bc5 4. O-O O-O"
    moves = parse_movetext(movetext)
    # White kingside castle: e1g1, Black kingside castle: e8g8
    assert moves[6] == "e1g1"
    assert moves[7] == "e8g8"


def test_queenside_castling() -> None:
    movetext = "1. d4 d5 2. Nc3 Nc6 3. Be3 Be6 4. Qd3 Qd6 5. O-O-O O-O-O"
    moves = parse_movetext(movetext)
    assert moves[8] == "e1c1"
    assert moves[9] == "e8c8"


def test_promotion_preserves_suffix() -> None:
    # Construct a position where a pawn promotes
    movetext = "1. h4 g5 2. hxg5 f5 3. gxf6 Nh6 4. fxe7 Nf5 5. exd8=Q"
    moves = parse_movetext(movetext)
    # Promotion UCI is 5 chars with piece suffix (q/r/b/n) preserved
    assert moves[-1] == "e7d8q"


def test_promotion_underpromotion_preserves_suffix() -> None:
    movetext = "1. h4 g5 2. hxg5 f5 3. gxf6 Nh6 4. fxe7 Nf5 5. exd8=N"
    moves = parse_movetext(movetext)
    assert moves[-1] == "e7d8n"


def test_ambiguous_knight_move() -> None:
    # Two knights can reach d2: Nbd2 vs Nfd2
    movetext = "1. Nf3 e5 2. Nc3 d5 3. Ncd5"
    moves = parse_movetext(movetext)
    # c3 knight moves to d5
    assert moves[4] == "c3d5"


def test_empty_movetext_raises() -> None:
    with pytest.raises(ValueError, match="Empty movetext"):
        parse_movetext("")


def test_whitespace_only_movetext_raises() -> None:
    with pytest.raises(ValueError, match="Empty movetext"):
        parse_movetext("   ")


def test_malformed_movetext_raises() -> None:
    with pytest.raises(ValueError):
        parse_movetext("not a chess game at all xyz")


# --- parse_result ---


def test_white_wins() -> None:
    assert parse_result("1-0") == "white"


def test_black_wins() -> None:
    assert parse_result("0-1") == "black"


def test_draw() -> None:
    assert parse_result("1/2-1/2") == "draw"


def test_incomplete_result_raises() -> None:
    with pytest.raises(ValueError, match="Unrecognized result"):
        parse_result("*")


def test_invalid_result_raises() -> None:
    with pytest.raises(ValueError, match="Unrecognized result"):
        parse_result("invalid")
