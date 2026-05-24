"""Turn a filtered Lichess stream into self-play opening seeds.

Reuses the high-Elo filter and PGN parsing from ``data_preparation`` /
``pgn_to_uci``; keeps only Rapid + Classical Normal-termination games long
enough to supply ``seed_plies`` opening moves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from halluci_mate.data_preparation import passes_highelo_filter
from halluci_mate.pgn_to_uci import parse_movetext

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

# Lichess ``Event`` substrings for the long time controls we seed from.
_RAPID_CLASSICAL = ("rapid", "classical")

SeedOpening = tuple[str, list[str]]


def is_rapid_or_classical(event: str) -> bool:
    """True if the Lichess ``Event`` names a rapid or classical game (not blitz/bullet)."""
    lowered = event.lower()
    return any(tc in lowered for tc in _RAPID_CLASSICAL)


def iter_seed_openings(stream: Iterable[dict], seed_plies: int, min_elo: int, max_rating_diff: int, max_elo_gap: int) -> Iterator[SeedOpening]:
    """Yield ``(seed_source, first seed_plies UCI moves)`` for qualifying games."""
    for sample in stream:
        if sample.get("Termination") != "Normal" or not is_rapid_or_classical(sample.get("Event", "")):
            continue
        if not passes_highelo_filter(sample, min_elo, max_rating_diff, max_elo_gap):
            continue
        movetext = sample.get("movetext")
        if movetext is None:
            continue
        try:
            moves = parse_movetext(movetext)
        except ValueError:
            continue
        if len(moves) < seed_plies:
            continue
        yield sample.get("Site", "unknown"), moves[:seed_plies]
