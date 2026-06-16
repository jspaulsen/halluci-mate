"""Hermetic end-to-end tests for the ``scripts/self_improve.py`` orchestration glue.

The loop is exercised in-process with the heavy boundaries stubbed: every
``subprocess.run`` (vs-stockfish generation + held-out eval, train_dpo) and
``export_dpo`` is faked, so no GPU, Stockfish, or checkpoint is needed. The
point is to verify the control flow the unit tests do NOT cover — stdout run-dir
parsing, keep/revert bookkeeping, best-checkpoint selection, the converged /
plateau stops, and the decision.json / session.json provenance — since that glue
is otherwise unexercised until a real GPU run.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import scripts.self_improve as si
from halluci_mate.eval.records import Evaluator
from halluci_mate.eval.runs import RunWriter, make_run_id
from halluci_mate.self_improve import CollapseConfig, RatchetConfig
from tests.helpers.eval_records import make_per_move_record

GEN_MOVES = ["e2e4", "d2d4", "g1f3", "b1c3"]
CANDIDATE_CKPT_NAME = "checkpoint-10"


def _flags(cmd: list[str]) -> dict[str, str]:
    """Map ``--flag value`` pairs from an argv list, skipping bare flags like ``--sf-analyze``."""
    return {tok: cmd[i + 1] for i, tok in enumerate(cmd) if tok.startswith("--") and i + 1 < len(cmd) and not cmd[i + 1].startswith("--")}


def _eval_metrics(score_rate: float, legal_rate: float = 0.99, cpl: float = 50.0) -> dict[str, Any]:
    """A vs-stockfish metrics.json payload with controllable ratchet signals (10 finished games)."""
    games = 10
    wins = round(score_rate * games)
    return {
        "evaluator": "vs_stockfish",
        "win_rate": {"overall": {"games": games, "wins": wins, "losses": games - wins, "draws": 0, "unfinished": 0, "win_rate": score_rate, "score_rate": score_rate}},
        "legal_rate": {"overall": {"n": 100, "legal": round(legal_rate * 100), "rate": legal_rate}},
        "centipawn_loss": {"overall": {"n": 100, "mean": cpl, "median": cpl, "p95": cpl}},
    }


class _FakePopen:
    """Minimal subprocess.Popen stand-in: yields canned stdout lines, then exposes returncode."""

    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = iter(stdout.splitlines(keepends=True))
        self.returncode = returncode

    def __enter__(self) -> _FakePopen:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


def _fake_popen_factory(cand_score: float, champ_score: float, cand_legal: float = 0.99) -> Any:
    """Build a ``subprocess.Popen`` stub for the streamed vs-stockfish gen/eval calls."""

    def fake_popen(cmd: list[str], **_kwargs: Any) -> _FakePopen:
        if "vs-stockfish" not in cmd:
            raise AssertionError(f"unexpected Popen command: {cmd}")
        return _fake_vs_stockfish(_flags(cmd), cand_score, champ_score, cand_legal)

    return fake_popen


def _fake_run(cmd: list[str], **_kwargs: Any) -> SimpleNamespace:
    """``subprocess.run`` stub for the (live-streamed, uncaptured) train_dpo launch."""
    if not any("train_dpo.py" in token for token in cmd):
        raise AssertionError(f"unexpected run command: {cmd}")
    flags = _flags(cmd)
    ckpt = Path(flags["--output-directory"]) / "mlflow-run" / CANDIDATE_CKPT_NAME
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "trainer_state.json").write_text(json.dumps({"log_history": [{"step": 10, "eval_rewards/accuracies": 0.7}]}), encoding="utf-8")
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def _fake_vs_stockfish(flags: dict[str, str], cand_score: float, champ_score: float, cand_legal: float = 0.99) -> _FakePopen:
    tag = flags["--checkpoint-tag"]
    run_dir = Path(flags["--evals-dir"]) / make_run_id(tag, Evaluator.VS_STOCKFISH)
    if tag.endswith("gen"):
        with RunWriter(run_dir) as writer:
            for i, move in enumerate(GEN_MOVES):
                writer.append_record(make_per_move_record(i, model_move=move))
    else:
        is_candidate = tag.endswith("cand")
        score = cand_score if is_candidate else champ_score
        legal = cand_legal if is_candidate else 0.99
        RunWriter(run_dir).write_metrics(_eval_metrics(score, legal_rate=legal))
    return _FakePopen(f"Run id: x\nRun dir: {run_dir}\n")


def _fake_export_factory(n_pairs: int) -> Any:
    def fake_export(*, output: Path, **_kwargs: Any) -> int:
        if n_pairs > 0:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('{"prompt":"p","moves_uci":["e2e4"],"model_side":"white","chosen":"d2d4","rejected":"a2a3"}\n', encoding="utf-8")
        return n_pairs

    return fake_export


def _patch_heavy_steps(monkeypatch: Any, *, cand_score: float, champ_score: float, n_pairs: int, cand_legal: float = 0.99) -> None:
    """Stub the three heavy boundaries: vs-stockfish (Popen), train_dpo (run), export_dpo."""
    monkeypatch.setattr(si.subprocess, "Popen", _fake_popen_factory(cand_score, champ_score, cand_legal))
    monkeypatch.setattr(si.subprocess, "run", _fake_run)
    monkeypatch.setattr(si, "export_dpo", _fake_export_factory(n_pairs))


def _cfg(tmp_path: Path, **overrides: Any) -> si.LoopConfig:
    session_dir = tmp_path / "sess"
    defaults: dict[str, Any] = {
        "base_model": "base/model",
        "session_dir": session_dir,
        "evals_dir": session_dir / "evals",
        "use_accelerate": False,
        "stockfish_bin": "stockfish",
        "n_gen_games": 4,
        "datagen_skill": 3,
        "datagen_depth": 8,
        "gen_temperature": 0.7,
        "gen_top_k": 0,
        "record_top_k": 5,
        "cpl_threshold": 200,
        "anchor_fraction": 0.0,
        "anchor_pairs_path": None,
        "dpo_beta": 0.1,
        "dpo_lr": 1e-5,
        "dpo_epochs": 1,
        "dpo_save_steps": 10,
        "n_eval_games": 10,
        "heldout_skill": 6,
        "heldout_depth": 12,
        "ratchet": RatchetConfig(score_rate_epsilon=0.03, legal_rate_floor=0.95, legal_rate_max_regression=0.005),
        "collapse": CollapseConfig(min_distinct_move_ratio=0.05, entropy_floor=0.05, entropy_drop_frac=0.5),
        "iterations": 1,
        "plateau_patience": 2,
        "seed": 4042,
    }
    defaults.update(overrides)
    return si.LoopConfig(**defaults)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_loop_keeps_improving_candidate(tmp_path: Path, monkeypatch: Any) -> None:
    _patch_heavy_steps(monkeypatch, cand_score=0.65, champ_score=0.5, n_pairs=1)
    cfg = _cfg(tmp_path)

    si._loop(cfg)

    session = _read(cfg.session_dir / "session.json")
    assert session["final_champion"].endswith(CANDIDATE_CKPT_NAME)
    assert session["history"][0]["kept"] is True
    decision = _read(cfg.session_dir / "iter01" / "decision.json")
    assert decision["kept"] is True
    # Draw-stall auditing: both win_rate blocks are recorded.
    assert decision["candidate_win_rate"]["score_rate"] == 0.65
    assert decision["champion_win_rate"]["score_rate"] == 0.5


def test_loop_reverts_worse_candidate(tmp_path: Path, monkeypatch: Any) -> None:
    _patch_heavy_steps(monkeypatch, cand_score=0.40, champ_score=0.5, n_pairs=1)
    cfg = _cfg(tmp_path)

    si._loop(cfg)

    session = _read(cfg.session_dir / "session.json")
    assert session["final_champion"] == "base/model"  # reverted to base
    assert session["history"][0]["kept"] is False
    assert _read(cfg.session_dir / "iter01" / "decision.json")["kept"] is False


def test_loop_vetoes_candidate_below_legal_floor(tmp_path: Path, monkeypatch: Any) -> None:
    # Candidate scores higher but plays below the legal-rate floor -> veto -> revert despite the score gain.
    _patch_heavy_steps(monkeypatch, cand_score=0.80, champ_score=0.5, n_pairs=1, cand_legal=0.90)
    cfg = _cfg(tmp_path)

    si._loop(cfg)

    session = _read(cfg.session_dir / "session.json")
    assert session["final_champion"] == "base/model"
    decision = _read(cfg.session_dir / "iter01" / "decision.json")
    assert decision["kept"] is False
    assert "floor" in decision["reason"]


def test_loop_converges_when_no_quality_pairs(tmp_path: Path, monkeypatch: Any) -> None:
    _patch_heavy_steps(monkeypatch, cand_score=0.65, champ_score=0.5, n_pairs=0)  # no pairs -> soft convergence
    cfg = _cfg(tmp_path, iterations=3)

    si._loop(cfg)

    session = _read(cfg.session_dir / "session.json")
    assert session["final_champion"] == "base/model"
    assert "converged" in session["history"][-1]["stopped"]


def test_loop_stops_on_plateau(tmp_path: Path, monkeypatch: Any) -> None:
    _patch_heavy_steps(monkeypatch, cand_score=0.40, champ_score=0.5, n_pairs=1)
    cfg = _cfg(tmp_path, iterations=5, plateau_patience=2)

    si._loop(cfg)

    session = _read(cfg.session_dir / "session.json")
    # Two consecutive reverts trip plateau_patience; the loop stops before iteration 3.
    iterations_run = [h for h in session["history"] if "iteration" in h and "kept" in h]
    assert len(iterations_run) == 2
    assert "plateau" in session["history"][-1]["stopped"]


def test_loop_rejects_malformed_anchor_file(tmp_path: Path) -> None:
    # Anchoring on a non-export-dpo file must fail before any generation runs.
    bad_anchor = tmp_path / "anchor.jsonl"
    bad_anchor.write_text("not a dpo pair\n", encoding="utf-8")
    cfg = _cfg(tmp_path, anchor_pairs_path=bad_anchor, anchor_fraction=0.5)

    with pytest.raises(si.SelfImproveError):
        si._loop(cfg)


def test_warn_draw_spike_fires_when_candidate_draws_more(capsys: Any) -> None:
    champion = {"games": 10, "wins": 5, "losses": 5, "draws": 0, "unfinished": 0}  # 0.0 draw fraction
    candidate = {"games": 10, "wins": 2, "losses": 3, "draws": 5, "unfinished": 0}  # 0.5 draw fraction
    si._warn_draw_spike(candidate, champion)
    assert "draw rate spiked" in capsys.readouterr().out


def test_warn_draw_spike_silent_when_below_threshold(capsys: Any) -> None:
    champion = {"games": 10, "wins": 5, "losses": 4, "draws": 1, "unfinished": 0}  # 0.1 draw fraction
    candidate = {"games": 10, "wins": 4, "losses": 4, "draws": 2, "unfinished": 0}  # 0.2 draw fraction; gap 0.1 < DRAW_SPIKE_WARN
    si._warn_draw_spike(candidate, champion)
    assert "draw rate spiked" not in capsys.readouterr().out
