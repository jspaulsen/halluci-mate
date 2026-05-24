from __future__ import annotations

import random

import chess
import chess.engine
import pytest

from halluci_mate.distill.engine_selfplay import (
    SelfPlayConfig,
    _AdjudicationState,
    _best_white_cp,
    _pv_first_move,
    _softmax,
    _stm_cp,
    select_wobble_move,
)


def test_config_defaults() -> None:
    config = SelfPlayConfig()
    assert config.depth == 18
    assert config.multipv == 4
    assert config.wobble_cp == 30
    assert config.seed_plies == 12
    assert config.max_plies == 200


def test_config_rejects_multipv_below_one() -> None:
    with pytest.raises(ValueError, match="multipv"):
        SelfPlayConfig(multipv=0)


def test_config_rejects_nonpositive_depth() -> None:
    with pytest.raises(ValueError, match="depth"):
        SelfPlayConfig(depth=0)


def test_config_rejects_nonpositive_wobble_temp() -> None:
    with pytest.raises(ValueError, match="wobble_temp"):
        SelfPlayConfig(wobble_temp=0.0)


def _info(cp: int, pv_uci: str | None) -> chess.engine.InfoDict:
    info: chess.engine.InfoDict = {"score": chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE)}
    if pv_uci is not None:
        info["pv"] = [chess.Move.from_uci(pv_uci)]
    return info


def test_stm_cp_white_to_move_is_white_relative() -> None:
    assert _stm_cp(_info(80, "e2e4"), chess.WHITE) == 80


def test_stm_cp_black_to_move_flips_sign() -> None:
    assert _stm_cp(_info(80, "e2e4"), chess.BLACK) == -80


def test_best_white_cp_uses_first_info() -> None:
    assert _best_white_cp([_info(120, "e2e4"), _info(90, "d2d4")]) == 120


def test_best_white_cp_clamps_mate() -> None:
    info: chess.engine.InfoDict = {"score": chess.engine.PovScore(chess.engine.Mate(2), chess.WHITE)}
    assert _best_white_cp([info]) > 90_000


def test_pv_first_move_returns_none_without_pv() -> None:
    assert _pv_first_move({"score": chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE)}) is None


def test_softmax_sums_to_one_and_orders() -> None:
    weights = _softmax([0.0, 1.0, 2.0])
    assert sum(weights) == pytest.approx(1.0)
    assert weights[2] > weights[1] > weights[0]


def test_wobble_filters_out_of_band_moves() -> None:
    # Black to move: white-relative scores invert. Make e7e5/g8f6 best for Black.
    infos = [_info(-200, "e7e5"), _info(-205, "g8f6"), _info(50, "a7a6")]
    config = SelfPlayConfig(wobble_cp=30)
    rng = random.Random(0)
    chosen = {select_wobble_move(infos, chess.BLACK, config, rng).uci() for _ in range(50)}
    assert chosen <= {"e7e5", "g8f6"}  # a7a6 is >30cp worse, never chosen


def test_wobble_single_candidate_is_deterministic() -> None:
    infos = [_info(20, "e2e4"), _info(-400, "a2a3")]
    config = SelfPlayConfig(wobble_cp=30)
    assert select_wobble_move(infos, chess.WHITE, config, random.Random(0)).uci() == "e2e4"


def test_wobble_seeded_rng_is_reproducible() -> None:
    infos = [_info(20, "e2e4"), _info(10, "d2d4"), _info(5, "g1f3")]
    config = SelfPlayConfig(wobble_cp=30)
    a = [select_wobble_move(infos, chess.WHITE, config, random.Random(7)).uci() for _ in range(5)]
    b = [select_wobble_move(infos, chess.WHITE, config, random.Random(7)).uci() for _ in range(5)]
    assert a == b


def test_wobble_raises_when_no_pv() -> None:
    infos = [{"score": chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE)}]
    with pytest.raises(ValueError, match="no candidate moves"):
        select_wobble_move(infos, chess.WHITE, SelfPlayConfig(), random.Random(0))


def test_adjudicates_resign_after_consecutive_winning_plies() -> None:
    config = SelfPlayConfig(resign_cp=700, resign_plies=4)
    state = _AdjudicationState()
    verdicts = [state.update(900, ply, config) for ply in range(4)]
    assert verdicts[:3] == [None, None, None]
    assert verdicts[3] == ("adjudicated-win", "white")


def test_resign_run_resets_when_eval_drops_back() -> None:
    config = SelfPlayConfig(resign_cp=700, resign_plies=3)
    state = _AdjudicationState()
    assert state.update(900, 0, config) is None
    assert state.update(100, 1, config) is None  # back in band -> run resets
    assert state.update(900, 2, config) is None
    assert state.update(900, 3, config) is None
    assert state.update(900, 4, config) == ("adjudicated-win", "white")


def test_black_winning_eval_adjudicates_black() -> None:
    config = SelfPlayConfig(resign_cp=700, resign_plies=2)
    state = _AdjudicationState()
    assert state.update(-800, 10, config) is None
    assert state.update(-800, 11, config) == ("adjudicated-win", "black")


def test_adjudicates_draw_only_after_min_ply() -> None:
    config = SelfPlayConfig(draw_cp=15, draw_plies=2, draw_min_ply=60)
    state = _AdjudicationState()
    assert state.update(0, 58, config) is None  # before draw_min_ply, no counting
    assert state.update(0, 59, config) is None
    assert state.update(0, 60, config) is None
    assert state.update(0, 61, config) == ("adjudicated-draw", "draw")


def test_decisive_ply_resets_draw_run_without_adjudicating() -> None:
    # A non-adjudicating decisive spike (eval out of the draw band but below the
    # resign streak) must reset draw_run so it cannot carry over into a later
    # spurious draw verdict. Resign needs 10 plies here, so the spike at ply 61
    # does not return early -- execution reaches the draw branch and resets.
    config = SelfPlayConfig(draw_cp=15, draw_plies=2, draw_min_ply=60, resign_cp=700, resign_plies=10)
    state = _AdjudicationState()
    assert state.update(0, 60, config) is None  # draw_run -> 1
    assert state.update(800, 61, config) is None  # decisive but no verdict; draw_run reset to 0
    assert state.draw_run == 0
    assert state.update(0, 62, config) is None  # draw_run only back to 1, no premature draw
