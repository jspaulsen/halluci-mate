"""Autonomous self-improvement loop for halluci-mate (Stage A).

Closes the verifier-anchored loop around the existing eval + DPO machinery:

    generate (vs-stockfish, sampled)  ->  select (export-dpo quality)
      ->  anchor (mix fixed high-Elo pairs)  ->  fine-tune (train_dpo)
      ->  evaluate (fixed held-out vs-stockfish)  ->  keep-or-revert

Stockfish is the ground-truth verifier, so this is an RLVR-style loop: the
held-out battery uses a *different, fixed* Stockfish skill/depth and greedy
decoding, isolated from the data-generation signal, so only gains that
generalize pass the ratchet. The champion only ever moves uphill on held-out
score; non-improving rounds revert and the next round regenerates from the
same champion.

Heavy steps shell out to the existing CLIs (``scripts/eval.py``,
``scripts/train_dpo.py``) so each runs in its own process and frees CUDA
memory between steps. The pure decision logic lives in
``halluci_mate.self_improve`` and is unit-tested without a GPU.

Example (smoke test, one iteration):

    uv run python scripts/self_improve.py \\
        --base-model jspaulsen/halluci-mate-v2b \\
        --session-name smoke --iterations 1 \\
        --n-gen-games 8 --n-eval-games 8 \\
        --datagen-skill 3 --heldout-skill 6 --no-accelerate
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from halluci_mate.eval.dpo_export import DpoFlavor, export_dpo
from halluci_mate.eval.runs import RunReader
from halluci_mate.self_improve import (
    CollapseConfig,
    DiversityStats,
    RatchetConfig,
    RatchetDecision,
    RatchetMetrics,
    SelfImproveError,
    decision_record,
    diversity_collapsed,
    diversity_stats,
    draw_fraction,
    extract_ratchet_metrics,
    metrics_to_dict,
    mix_anchor_pairs,
    ratchet_decision,
    select_best_checkpoint,
    validate_anchor_pairs,
    write_json,
)

# Warn (loudly, not a veto) when a candidate's draw fraction exceeds the
# champion's by more than this — the signature of draw-to-survive reward hacking.
DRAW_SPIKE_WARN = 0.15


class NoQualityPairs(SelfImproveError):
    """Raised when a generation round yields no quality pairs — treated as convergence, not failure."""


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR_PREFIX = "Run dir:"
# Last N chars of a failed subprocess' combined output to surface in the raised error.
OUTPUT_TAIL = 4000


@dataclass(frozen=True)
class LoopConfig:
    """All knobs for one self-improvement session, assembled from the CLI."""

    base_model: str
    session_dir: Path
    evals_dir: Path
    use_accelerate: bool
    stockfish_bin: str
    n_gen_games: int
    datagen_skill: int
    datagen_depth: int
    gen_temperature: float
    gen_top_k: int
    record_top_k: int
    cpl_threshold: int
    anchor_fraction: float
    anchor_pairs_path: Path | None
    dpo_beta: float
    dpo_lr: float
    dpo_epochs: int
    dpo_save_steps: int
    n_eval_games: int
    heldout_skill: int
    heldout_depth: int
    ratchet: RatchetConfig
    collapse: CollapseConfig
    iterations: int
    plateau_patience: int
    seed: int


@dataclass(frozen=True)
class IterationOutcome:
    """What one iteration produced, before the loop applies ratchet bookkeeping."""

    decision: RatchetDecision
    candidate_ckpt: str
    candidate_metrics: RatchetMetrics
    diversity: DiversityStats


def _run_capture(cmd: list[str]) -> str:
    """Run a command from the repo root, streaming its output live while capturing it; raise on failure.

    stderr is merged into stdout so the live log stays ordered and the failure
    tail is complete; the returned text is parsed for the announced run dir.
    These steps (vs-stockfish generation + held-out eval) run for minutes, so
    streaming keeps the loop observable rather than silent until completion.
    """
    captured: list[str] = []
    with subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        for line in proc.stdout or ():
            print(line, end="")
            captured.append(line)
    output = "".join(captured)
    if proc.returncode != 0:
        raise SelfImproveError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{output[-OUTPUT_TAIL:]}")
    return output


def _run_streamed(cmd: list[str]) -> None:
    """Run a command from the repo root with inherited stdio (live logs), raise on failure."""
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise SelfImproveError(f"command failed ({result.returncode}): {' '.join(cmd)}")


def _parse_run_dir(stdout: str) -> Path:
    """Extract the run directory ``vs-stockfish`` announced on stdout (``Run dir: ...``).

    KNOWN TECH DEBT: this is an implicit string contract with ``eval.py`` (its
    ``Run dir:`` print line) — the loop's most fragile coupling. The run id is in
    principle derivable (``make_run_id`` + ``--evals-dir``), so the durable fix is
    to have ``eval.py`` accept a pre-computed ``--run-id`` the orchestrator shares,
    or emit machine-readable output, removing the dependence on the log format.
    """
    for line in stdout.splitlines():
        if line.startswith(RUN_DIR_PREFIX):
            return Path(line.removeprefix(RUN_DIR_PREFIX).strip())
    raise SelfImproveError(f"could not find '{RUN_DIR_PREFIX}' in eval output")


def _vs_stockfish_cmd(cfg: LoopConfig, *, checkpoint: str, tag: str, games: int, skill: int, depth: int, temperature: float, top_k: int) -> list[str]:
    """Build the ``eval.py vs-stockfish --sf-analyze`` command shared by generation and held-out eval."""
    return [
        "uv", "run", "python", "scripts/eval.py", "vs-stockfish",
        "--checkpoint", checkpoint, "--checkpoint-tag", tag,
        "--evals-dir", str(cfg.evals_dir), "--stockfish", cfg.stockfish_bin,
        "--games", str(games), "--halluci-color", "alternate",
        "--stockfish-skill", str(skill), "--stockfish-depth", str(depth),
        "--temperature", str(temperature), "--top-k", str(top_k),
        "--record-top-k", str(cfg.record_top_k), "--sf-analyze",
        "--blunder-threshold-cp", str(cfg.cpl_threshold),
    ]  # fmt: skip


def _train_cmd(cfg: LoopConfig, *, pairs: Path, base: str, out_dir: Path) -> list[str]:
    """Build the ``train_dpo.py`` command, continuing DPO from the current champion."""
    launcher = ["uv", "run", "accelerate", "launch"] if cfg.use_accelerate else ["uv", "run", "python"]
    return [
        *launcher, "scripts/train_dpo.py",
        "--pairs-path", str(pairs), "--base-model", base, "--output-directory", str(out_dir),
        "--beta", str(cfg.dpo_beta), "--learning-rate", str(cfg.dpo_lr),
        "--epochs", str(cfg.dpo_epochs), "--save-steps", str(cfg.dpo_save_steps), "--seed", str(cfg.seed),
    ]  # fmt: skip


def _generate(cfg: LoopConfig, champion: str, iteration: int) -> Path:
    """Play sampled games from the champion vs data-gen Stockfish; return the run dir."""
    cmd = _vs_stockfish_cmd(
        cfg,
        checkpoint=champion,
        tag=f"iter{iteration:02d}gen",
        games=cfg.n_gen_games,
        skill=cfg.datagen_skill,
        depth=cfg.datagen_depth,
        temperature=cfg.gen_temperature,
        top_k=cfg.gen_top_k,
    )
    return _parse_run_dir(_run_capture(cmd))


def _select(cfg: LoopConfig, gen_run_dir: Path, iter_dir: Path) -> Path:
    """Export fresh on-policy quality pairs from this iteration's games (online DPO)."""
    online = iter_dir / "pairs_online.jsonl"
    n_pairs = export_dpo(
        run_dir=gen_run_dir,
        output=online,
        flavor=DpoFlavor.QUALITY,
        threshold=cfg.cpl_threshold,
        require_consequential=True,
        exclude_repetition=True,
    )
    if n_pairs == 0:
        raise NoQualityPairs(f"0 quality pairs from {gen_run_dir} (champion made no blunders above --cpl-threshold, or --n-gen-games too low)")
    print(f"[select] {n_pairs} online quality pairs -> {online}")
    return online


def _anchor(cfg: LoopConfig, online: Path, iter_dir: Path) -> Path:
    """Mix a fixed fraction of anchor pairs into the online pairs (collapse resistance)."""
    combined = iter_dir / "pairs_combined.jsonl"
    result = mix_anchor_pairs(online_path=online, anchor_path=cfg.anchor_pairs_path, fraction=cfg.anchor_fraction, seed=cfg.seed, out_path=combined)
    print(f"[anchor] online={result.online} anchor={result.anchor} total={result.total} -> {combined}")
    return combined


def _train(cfg: LoopConfig, pairs: Path, base: str, iter_dir: Path) -> Path:
    """Run DPO and return the behaviorally-best candidate checkpoint (by eval accuracy, not last step)."""
    out_dir = iter_dir / "dpo"
    _run_streamed(_train_cmd(cfg, pairs=pairs, base=base, out_dir=out_dir))
    candidate = select_best_checkpoint(out_dir)
    print(f"[train] candidate checkpoint: {candidate}")
    return candidate


def _held_out_eval(cfg: LoopConfig, checkpoint: str, tag: str) -> tuple[RatchetMetrics, dict[str, Any]]:
    """Evaluate a checkpoint on the fixed, isolated held-out battery (greedy).

    Returns the ratchet metrics and the raw ``win_rate.overall`` block (for
    draw-stall auditing). NOTE: Stockfish Skill Level < 20 randomizes move
    choice (unseeded), so this battery is NOT bit-reproducible; the loop
    co-evaluates the champion each round so both sides see a fresh draw rather
    than comparing against a stale cached value.
    """
    cmd = _vs_stockfish_cmd(
        cfg,
        checkpoint=checkpoint,
        tag=tag,
        games=cfg.n_eval_games,
        skill=cfg.heldout_skill,
        depth=cfg.heldout_depth,
        temperature=0.0,
        top_k=0,
    )
    metrics = RunReader(_parse_run_dir(_run_capture(cmd))).read_metrics()
    return extract_ratchet_metrics(metrics), metrics["win_rate"]["overall"]


def _run_iteration(cfg: LoopConfig, iteration: int, champion: str) -> IterationOutcome:
    """Run one full generate -> select -> anchor -> train -> co-evaluate -> decide pass."""
    iter_dir = cfg.session_dir / f"iter{iteration:02d}"
    gen_run_dir = _generate(cfg, champion, iteration)
    diversity = diversity_stats(RunReader(gen_run_dir).read_records())
    combined = _anchor(cfg, _select(cfg, gen_run_dir, iter_dir), iter_dir)
    candidate_ckpt = _train(cfg, combined, champion, iter_dir)
    candidate_metrics, candidate_wr = _held_out_eval(cfg, str(candidate_ckpt), f"iter{iteration:02d}cand")
    champion_metrics, champion_wr = _held_out_eval(cfg, champion, f"iter{iteration:02d}champ")
    _warn_draw_spike(candidate_wr, champion_wr)
    decision = ratchet_decision(candidate_metrics, champion_metrics, cfg.ratchet)
    write_json(
        iter_dir / "decision.json",
        decision_record(
            iteration=iteration,
            candidate=candidate_metrics,
            champion=champion_metrics,
            decision=decision,
            diversity=diversity,
            candidate_winrate=candidate_wr,
            champion_winrate=champion_wr,
        ),
    )
    return IterationOutcome(decision=decision, candidate_ckpt=str(candidate_ckpt), candidate_metrics=candidate_metrics, diversity=diversity)


def _warn_draw_spike(candidate_wr: dict[str, Any], champion_wr: dict[str, Any]) -> None:
    """Print a warning if the candidate draws far more than the champion (possible draw-stall hack)."""
    candidate_draws = draw_fraction(candidate_wr)
    champion_draws = draw_fraction(champion_wr)
    if candidate_draws > champion_draws + DRAW_SPIKE_WARN:
        print(f"[warn] draw rate spiked {champion_draws:.2f} -> {candidate_draws:.2f}; score_rate gain may be draw-stalling, not real improvement")


def _stop_reason(cfg: LoopConfig, diversity: DiversityStats, baseline: DiversityStats, consecutive_reverts: int) -> str:
    """Return a non-empty stop reason on diversity collapse or plateau, else ``""``."""
    collapsed, why = diversity_collapsed(diversity, baseline, cfg.collapse)
    if collapsed:
        return f"diversity collapse: {why}"
    if consecutive_reverts >= cfg.plateau_patience:
        return f"plateau: {consecutive_reverts} consecutive non-improving iterations"
    return ""


def _loop(cfg: LoopConfig) -> None:
    """Run up to ``cfg.iterations`` ratcheted iterations, co-evaluating the champion each round."""
    if cfg.anchor_fraction > 0:
        if cfg.anchor_pairs_path is None:
            raise SelfImproveError("anchor_fraction > 0 requires --anchor-pairs-path; pass an anchor file or set --anchor-fraction 0")
        validate_anchor_pairs(cfg.anchor_pairs_path)
    anchoring = f"on ({cfg.anchor_fraction:.0%} from {cfg.anchor_pairs_path})" if cfg.anchor_fraction > 0 else "OFF"
    print(f"[start] champion={cfg.base_model} held-out=skill{cfg.heldout_skill}/depth{cfg.heldout_depth} games={cfg.n_eval_games} anchoring={anchoring}")
    champion = cfg.base_model
    champion_metrics: RatchetMetrics | None = None
    baseline_div: DiversityStats | None = None
    consecutive_reverts = 0
    history: list[dict[str, object]] = []
    for iteration in range(1, cfg.iterations + 1):
        try:
            outcome = _run_iteration(cfg, iteration, champion)
        except NoQualityPairs as exc:
            print(f"[stop] converged: {exc}")
            history.append({"iteration": iteration, "stopped": f"converged: {exc}"})
            break
        baseline_div = outcome.diversity if baseline_div is None else baseline_div
        kept = outcome.decision.keep
        print(f"[ratchet] iter {iteration}: {'KEEP' if kept else 'REVERT'} — {outcome.decision.reason}")
        if kept:
            champion, champion_metrics, consecutive_reverts = outcome.candidate_ckpt, outcome.candidate_metrics, 0
        else:
            consecutive_reverts += 1
        history.append({"iteration": iteration, "kept": kept, "champion": champion, "reason": outcome.decision.reason})
        stop = _stop_reason(cfg, outcome.diversity, baseline_div, consecutive_reverts)
        if stop:
            print(f"[stop] {stop}")
            history.append({"stopped": stop})
            break

    # champion_metrics is None iff no candidate was ever kept (champion is still the base model).
    final = None if champion_metrics is None else metrics_to_dict(champion_metrics)
    write_json(cfg.session_dir / "session.json", {"base_model": cfg.base_model, "final_champion": champion, "final_metrics": final, "history": history})
    print(f"[done] final champion: {champion}{'' if champion != cfg.base_model else ' (unchanged — no iteration improved on the base model)'}")


def main(
    base_model: Annotated[str, typer.Option(help="HF repo id or local checkpoint to start the loop from.")] = "jspaulsen/halluci-mate-v2b",
    session_name: Annotated[str, typer.Option(help="Names the work dir under self-improve/<name>/.")] = "default",
    iterations: Annotated[int, typer.Option(help="Max ratcheted iterations (K).")] = 4,
    stockfish: Annotated[str, typer.Option(help="Path to the stockfish binary.")] = "stockfish",
    use_accelerate: Annotated[bool, typer.Option("--accelerate/--no-accelerate", help="Launch train_dpo via accelerate (multi-GPU) or plain python.")] = True,
    n_gen_games: Annotated[int, typer.Option(help="Games played per iteration to generate training data.")] = 50,
    datagen_skill: Annotated[int, typer.Option(help="Stockfish skill for data generation (distinct from held-out).")] = 3,
    datagen_depth: Annotated[int, typer.Option(help="Stockfish depth for data-gen analysis (raise for CPL signal).")] = 8,
    gen_temperature: Annotated[float, typer.Option(help="Sampling temperature during generation (>0 for on-policy variety).")] = 0.7,
    gen_top_k: Annotated[int, typer.Option(help="Top-k sampling cutoff during generation (0 = disabled).")] = 0,
    record_top_k: Annotated[int, typer.Option(help="K candidates captured per move (used for diversity/entropy).")] = 5,
    cpl_threshold: Annotated[int, typer.Option(help="Centipawn-loss threshold for a quality pair / blunder tag.")] = 200,
    anchor_fraction: Annotated[
        float, typer.Option(help="Fraction of combined pairs drawn from the anchor file (0 disables anchoring; any value > 0 requires --anchor-pairs-path).")
    ] = 0.0,
    anchor_pairs_path: Annotated[
        Path | None, typer.Option(help="Fixed high-Elo anchor pairs JSONL (collapse resistance). MUST be export-dpo schema: {moves_uci, model_side, chosen, rejected} per line.")
    ] = None,
    dpo_beta: Annotated[float, typer.Option(help="DPO KL strength.")] = 0.1,
    dpo_lr: Annotated[float, typer.Option(help="DPO learning rate.")] = 1e-5,
    dpo_epochs: Annotated[int, typer.Option(help="DPO epochs per iteration.")] = 2,
    dpo_save_steps: Annotated[
        int, typer.Option(help="train_dpo checkpoint + eval cadence; lower it for small smoke runs (<100 steps) so candidates still get eval-accuracy scores for selection.")
    ] = 100,
    n_eval_games: Annotated[
        int,
        typer.Option(
            help="Games per checkpoint in the held-out battery (champion+candidate co-evaluated each round). Raise for less ratchet noise; one game ~= 1/games of score_rate."
        ),
    ] = 50,
    heldout_skill: Annotated[
        int, typer.Option(help="Held-out Stockfish skill (fixed all session; isolated from data-gen). NOTE skill < 20 is unseeded-random, so the battery is not bit-reproducible.")
    ] = 6,
    heldout_depth: Annotated[int, typer.Option(help="Held-out Stockfish depth (fixed all session).")] = 12,
    score_rate_epsilon: Annotated[
        float, typer.Option(help="Min held-out score_rate gain to count as improvement. Keep >= ~1 game (1/n-eval-games); decisions below the binomial noise floor are unreliable.")
    ] = 0.03,
    legal_rate_floor: Annotated[
        float, typer.Option(help="Hard floor: a candidate below this legal_rate is vetoed. Keep well below the base model's raw legal rate (~0.987) or every candidate is vetoed.")
    ] = 0.95,
    legal_rate_max_regression: Annotated[float, typer.Option(help="Max tolerated legal_rate drop vs champion before veto.")] = 0.005,
    plateau_patience: Annotated[int, typer.Option(help="Stop after this many consecutive non-improving iterations.")] = 2,
    min_distinct_move_ratio: Annotated[float, typer.Option(help="Stop if generation distinct-move ratio falls below this.")] = 0.05,
    entropy_floor: Annotated[float, typer.Option(help="Stop if mean top-k entropy (nats) falls below this.")] = 0.05,
    entropy_drop_frac: Annotated[float, typer.Option(help="Stop if mean top-k entropy drops more than this fraction vs baseline.")] = 0.5,
    seed: Annotated[
        int,
        typer.Option(
            help="Seed for anchor sampling and DPO. Does NOT seed generation (vs-stockfish has no seed), so games vary each round — intentional fresh exploration after a revert."
        ),
    ] = 4042,
) -> None:
    session_dir = (REPO_ROOT / "self-improve" / session_name).resolve()
    cfg = LoopConfig(
        base_model=base_model,
        session_dir=session_dir,
        evals_dir=session_dir / "evals",
        use_accelerate=use_accelerate,
        stockfish_bin=stockfish,
        n_gen_games=n_gen_games,
        datagen_skill=datagen_skill,
        datagen_depth=datagen_depth,
        gen_temperature=gen_temperature,
        gen_top_k=gen_top_k,
        record_top_k=record_top_k,
        cpl_threshold=cpl_threshold,
        anchor_fraction=anchor_fraction,
        anchor_pairs_path=anchor_pairs_path,
        dpo_beta=dpo_beta,
        dpo_lr=dpo_lr,
        dpo_epochs=dpo_epochs,
        dpo_save_steps=dpo_save_steps,
        n_eval_games=n_eval_games,
        heldout_skill=heldout_skill,
        heldout_depth=heldout_depth,
        ratchet=RatchetConfig(score_rate_epsilon=score_rate_epsilon, legal_rate_floor=legal_rate_floor, legal_rate_max_regression=legal_rate_max_regression),
        collapse=CollapseConfig(min_distinct_move_ratio=min_distinct_move_ratio, entropy_floor=entropy_floor, entropy_drop_frac=entropy_drop_frac),
        iterations=iterations,
        plateau_patience=plateau_patience,
        seed=seed,
    )
    _loop(cfg)


if __name__ == "__main__":
    typer.run(main)
