"""Tests for game-to-sequence transformation."""

from __future__ import annotations

import pytest

from halluci_mate.game_to_sequences import game_to_sequences

SAMPLE_MOVES = ["e2e4", "e7e5", "g1f3", "b8c6"]


def test_white_win_produces_one_sequence() -> None:
    seqs = game_to_sequences(SAMPLE_MOVES, "white")
    assert len(seqs) == 1


def test_white_win_starts_with_white_token() -> None:
    seq = game_to_sequences(SAMPLE_MOVES, "white")[0]
    assert seq.startswith("<WHITE>")


def test_white_win_ends_with_eos() -> None:
    seq = game_to_sequences(SAMPLE_MOVES, "white")[0]
    assert seq.endswith("<EOS>")


def test_white_win_has_no_draw_token() -> None:
    seq = game_to_sequences(SAMPLE_MOVES, "white")[0]
    assert "<DRAW>" not in seq


def test_black_win_starts_with_black_token() -> None:
    seq = game_to_sequences(SAMPLE_MOVES, "black")[0]
    assert seq.startswith("<BLACK>")


def test_black_win_produces_one_sequence() -> None:
    seqs = game_to_sequences(SAMPLE_MOVES, "black")
    assert len(seqs) == 1


def test_draw_produces_two_sequences() -> None:
    seqs = game_to_sequences(SAMPLE_MOVES, "draw")
    assert len(seqs) == 2


def test_draw_has_both_perspectives() -> None:
    seqs = game_to_sequences(SAMPLE_MOVES, "draw")
    assert seqs[0].startswith("<WHITE>")
    assert seqs[1].startswith("<BLACK>")


def test_draw_sequences_contain_draw_token() -> None:
    seqs = game_to_sequences(SAMPLE_MOVES, "draw")
    for seq in seqs:
        assert "<DRAW>" in seq
        assert seq.endswith("<DRAW> <EOS>")


def test_moves_in_correct_order() -> None:
    seq = game_to_sequences(SAMPLE_MOVES, "white")[0]
    tokens = seq.split()
    # Skip first token (WHITE) and last token (EOS)
    move_tokens = tokens[1:-1]
    assert move_tokens == SAMPLE_MOVES


def test_invalid_outcome_raises() -> None:
    with pytest.raises(ValueError, match="Invalid outcome"):
        game_to_sequences(SAMPLE_MOVES, "invalid")
