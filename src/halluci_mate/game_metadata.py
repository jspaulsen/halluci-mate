"""Classify game metadata for stratified eval set construction."""

from __future__ import annotations

_OPENING_FAMILY_MAP: dict[str, str] = {
    "e2e4": "e4",
    "d2d4": "d4",
    "c2c4": "c4",
    "g1f3": "Nf3",
}

# Upper boundary (exclusive) → bucket label
_ELO_BOUNDARIES: list[tuple[int, str]] = [
    (1200, "<1200"),
    (1600, "1200-1600"),
    (2000, "1600-2000"),
]
_ELO_TOP_BUCKET = "2000+"



def classify_elo_bucket(white_elo: int, black_elo: int) -> str:
    """Classify average ELO into a rating bucket."""
    avg = (white_elo + black_elo) // 2
    for boundary, label in _ELO_BOUNDARIES:
        if avg < boundary:
            return label
    return _ELO_TOP_BUCKET


def classify_opening_family(first_move: str) -> str:
    """Classify opening family from the first UCI move."""
    return _OPENING_FAMILY_MAP.get(first_move, "other")


def classify_termination_type(result: str) -> str:
    """Classify result as decisive or draw."""
    if result == "1/2-1/2":
        return "draw"
    return "decisive"
