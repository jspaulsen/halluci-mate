"""Tests for game metadata classification."""

from __future__ import annotations

import pytest

from halluci_mate.game_metadata import classify_elo_bucket, classify_opening_family, classify_termination_type

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


# --- Termination type ---

TERMINATION_CASES = [
    ("1-0", "decisive"),
    ("0-1", "decisive"),
    ("1/2-1/2", "draw"),
]


@pytest.mark.parametrize(("result", "expected"), TERMINATION_CASES)
def test_classify_termination_type(result: str, expected: str) -> None:
    assert classify_termination_type(result) == expected
