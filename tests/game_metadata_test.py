"""Tests for game metadata classification."""

from __future__ import annotations

import pytest

from halluci_mate.game_metadata import classify_elo_bucket, classify_opening_family, classify_termination_type, classify_time_control, is_blitz

# --- ELO bucket ---

ELO_CASES = [
    (800, 1000, "<1200"),
    (1199, 1199, "<1200"),
    (1200, 1200, "1200-1600"),
    (1500, 1700, "1600-2000"),
    (1600, 1600, "1600-2000"),
    (2000, 2000, "2000+"),
    (2400, 2600, "2000+"),
]


@pytest.mark.parametrize(("white", "black", "expected"), ELO_CASES)
def test_classify_elo_bucket(white: int, black: int, expected: str) -> None:
    assert classify_elo_bucket(white, black) == expected


# --- Opening family ---

OPENING_CASES = [
    ("e2e4", "e4"),
    ("d2d4", "d4"),
    ("c2c4", "c4"),
    ("g1f3", "Nf3"),
    ("b2b3", "other"),
    ("f2f4", "other"),
    ("a2a3", "other"),
]


@pytest.mark.parametrize(("first_move", "expected"), OPENING_CASES)
def test_classify_opening_family(first_move: str, expected: str) -> None:
    assert classify_opening_family(first_move) == expected


# --- Time control ---

TIME_CASES = [
    ("60+0", "bullet"),       # 60s effective
    ("120+0", "bullet"),      # 120s effective
    ("180+0", "blitz"),       # 180s effective (boundary)
    ("300+0", "blitz"),       # 300s effective
    ("300+3", "blitz"),       # 300 + 120 = 420s
    ("300+5", "rapid"),       # 300 + 200 = 500s
    ("600+0", "rapid"),       # 600s effective
    ("600+10", "rapid"),      # 600 + 400 = 1000s
    ("1500+0", "classical"),  # 1500s (boundary)
    ("1800+30", "classical"), # 1800 + 1200 = 3000s
    ("-", "classical"),       # correspondence
    ("invalid", "classical"), # unparseable
]


@pytest.mark.parametrize(("tc", "expected"), TIME_CASES)
def test_classify_time_control(tc: str, expected: str) -> None:
    assert classify_time_control(tc) == expected


# --- Termination type ---

TERMINATION_CASES = [
    ("1-0", "decisive"),
    ("0-1", "decisive"),
    ("1/2-1/2", "draw"),
]


@pytest.mark.parametrize(("result", "expected"), TERMINATION_CASES)
def test_classify_termination_type(result: str, expected: str) -> None:
    assert classify_termination_type(result) == expected


# --- is_blitz ---

BLITZ_CASES = [
    ("180+0", True),
    ("300+0", True),
    ("300+3", True),
    ("60+0", False),     # bullet
    ("600+0", False),    # rapid
    ("1500+0", False),   # classical
    ("-", False),        # correspondence
]


@pytest.mark.parametrize(("tc", "expected"), BLITZ_CASES)
def test_is_blitz(tc: str, expected: bool) -> None:
    assert is_blitz(tc) == expected
