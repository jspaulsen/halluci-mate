from __future__ import annotations

from halluci_mate.distill.seeds import is_rapid_or_classical, iter_seed_openings

_OK_GAME = {
    "Event": "Rated Rapid game",
    "Termination": "Normal",
    "Site": "https://lichess.org/abc",
    "WhiteElo": "2400",
    "BlackElo": "2410",
    "WhiteRatingDiff": "5",
    "BlackRatingDiff": "-5",
    "movetext": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 *",
}


def test_is_rapid_or_classical_excludes_blitz() -> None:
    assert is_rapid_or_classical("Rated Rapid game")
    assert is_rapid_or_classical("Rated Classical game")
    assert not is_rapid_or_classical("Rated Blitz game")
    assert not is_rapid_or_classical("Rated Bullet game")


def test_iter_seed_openings_extracts_prefix() -> None:
    seeds = list(iter_seed_openings([_OK_GAME], seed_plies=4, min_elo=2000, max_rating_diff=30, max_elo_gap=200))
    assert len(seeds) == 1
    source, moves = seeds[0]
    assert source == "https://lichess.org/abc"
    assert moves == ["e2e4", "e7e5", "g1f3", "b8c6"]


def test_iter_seed_openings_skips_blitz_and_lowelo_and_short() -> None:
    blitz = {**_OK_GAME, "Event": "Rated Blitz game"}
    low_elo = {**_OK_GAME, "WhiteElo": "1500", "BlackElo": "1500"}
    too_short = {**_OK_GAME, "movetext": "1. e4 e5 *"}
    seeds = list(iter_seed_openings([blitz, low_elo, too_short], seed_plies=4, min_elo=2000, max_rating_diff=30, max_elo_gap=200))
    assert seeds == []


def test_iter_seed_openings_skips_rows_missing_required_fields() -> None:
    no_termination = {k: v for k, v in _OK_GAME.items() if k != "Termination"}
    no_movetext = {k: v for k, v in _OK_GAME.items() if k != "movetext"}
    seeds = list(iter_seed_openings([no_termination, no_movetext], seed_plies=4, min_elo=2000, max_rating_diff=30, max_elo_gap=200))
    assert seeds == []
