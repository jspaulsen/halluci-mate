"""halluci-mate eval harness CLI.

Single entry point for running evaluators against a checkpoint and for
recomputing aggregates over an existing run directory. Wired-up subcommands:

* ``vs-stockfish`` — play N games against Stockfish, write the HAL-5 run
  directory, and aggregate ``metrics.json`` via ``compute_all``.
* ``report <run-id>`` — recompute ``metrics.json`` from an existing
  ``records.jsonl`` + ``config.json``. No re-run.

Additional evaluators land per the build order in ``docs/eval_harness.md``.

The ``--checkpoint`` argument accepts either a local checkpoint directory
(e.g. ``runs-v1/marvelous-deer-608/checkpoint-9660``) or a Hugging Face repo
id (e.g. ``jspaulsen/halluci-mate-v1a``); both are handled by
``AutoModelForCausalLM.from_pretrained`` under the hood.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import chess.engine

from halluci_mate.eval.evaluators.vs_stockfish import (
    STOCKFISH_SKILL_MAX,
    STOCKFISH_SKILL_MIN,
    GameOutcome,
    VsStockfishConfig,
    run_vs_stockfish,
)
from halluci_mate.eval.metrics import compute_all
from halluci_mate.eval.records import Evaluator
from halluci_mate.eval.runs import RunReader, RunWriter, make_run_id, resolve_checkpoint_tag
from halluci_mate.inference import ChessInferenceEngine

DEFAULT_EVALS_DIR = Path("evals")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point. ``argv`` is exposed for in-process smoke tests."""
    parser = argparse.ArgumentParser(description="halluci-mate eval harness CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    vs_parser = subparsers.add_parser(
        "vs-stockfish",
        help="Play N games against Stockfish and emit a HAL-5 run directory.",
    )
    _add_vs_stockfish_args(vs_parser)
    vs_parser.set_defaults(func=_run_vs_stockfish_cmd)

    report_parser = subparsers.add_parser(
        "report",
        help="Recompute metrics.json from an existing run directory's records.jsonl.",
    )
    _add_report_args(report_parser)
    report_parser.set_defaults(func=_run_report_cmd)

    args = parser.parse_args(argv)
    args.func(args)


def _add_vs_stockfish_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint directory or Hugging Face repo id.")
    parser.add_argument(
        "--checkpoint-tag",
        default=None,
        help="Tag used in run-id; defaults to a sanitized form of --checkpoint. User-supplied tags must not contain '_'.",
    )
    parser.add_argument("--evals-dir", type=Path, default=DEFAULT_EVALS_DIR, help=f"Parent directory for run outputs (default: {DEFAULT_EVALS_DIR}).")
    parser.add_argument("--stockfish", default="stockfish", help="Path to the stockfish binary (default: 'stockfish' on PATH).")
    parser.add_argument("--games", type=int, default=2, help="Number of games to play (default: 2).")
    parser.add_argument(
        "--halluci-color",
        choices=["white", "black", "alternate"],
        default="alternate",
        help="Which color halluci-mate plays. 'alternate' flips each game starting with white.",
    )
    parser.add_argument(
        "--stockfish-skill",
        type=int,
        default=0,
        help=f"Stockfish 'Skill Level' UCI option [{STOCKFISH_SKILL_MIN}-{STOCKFISH_SKILL_MAX}] (default: 0, weakest).",
    )
    parser.add_argument("--stockfish-depth", type=int, default=1, help="Stockfish search depth per move (default: 1).")
    parser.add_argument("--stockfish-movetime", type=float, default=None, help="Stockfish time per move in seconds. Overrides --stockfish-depth if set.")
    parser.add_argument("--max-plies", type=int, default=400, help="Abort a game after this many plies (default: 400).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for halluci-mate (0.0 = greedy).")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling cutoff (0 = disabled).")
    parser.add_argument("--unconstrained", action="store_true", help="Disable legal-move masking for halluci-mate.")
    parser.add_argument("--record-top-k", type=int, default=5, help="K for the top-K candidates captured per record (0 = skip).")
    parser.add_argument("--blunder-threshold-cp", type=int, default=200, help="Centipawn loss threshold for is_blunder.")
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available, else cpu).")


def _run_vs_stockfish_cmd(args: argparse.Namespace) -> None:
    config = VsStockfishConfig(
        games=args.games,
        halluci_color=args.halluci_color,
        stockfish_skill=args.stockfish_skill,
        stockfish_depth=args.stockfish_depth if args.stockfish_movetime is None else None,
        stockfish_movetime=args.stockfish_movetime,
        max_plies=args.max_plies,
        unconstrained=args.unconstrained,
        record_top_k=args.record_top_k,
        blunder_threshold_cp=args.blunder_threshold_cp,
    )

    tag = resolve_checkpoint_tag(args.checkpoint, args.checkpoint_tag)
    run_id = make_run_id(tag, Evaluator.VS_STOCKFISH)
    run_dir = args.evals_dir / run_id
    print(f"Run id: {run_id}")
    print(f"Run dir: {run_dir}")

    engine = ChessInferenceEngine.from_checkpoint(
        args.checkpoint,
        constrained=not args.unconstrained,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )

    # Source-of-truth for sampling params is the engine itself; persist its
    # effective values so the on-disk record can't desync from what was used.
    extra_config: dict[str, object] = {"temperature": engine.temperature, "top_k": engine.top_k}

    stockfish = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    try:
        outcomes = run_vs_stockfish(
            engine=engine,
            stockfish=stockfish,
            config=config,
            run_dir=run_dir,
            run_id=run_id,
            checkpoint=str(args.checkpoint),
            extra_config=extra_config,
        )
    finally:
        stockfish.quit()

    _aggregate_metrics(run_dir)
    _print_summary(outcomes, run_dir)


def _add_report_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run_id", help="Run id under --evals-dir; the run directory must already contain records.jsonl and config.json.")
    parser.add_argument("--evals-dir", type=Path, default=DEFAULT_EVALS_DIR, help=f"Parent directory holding the run (default: {DEFAULT_EVALS_DIR}).")


def _run_report_cmd(args: argparse.Namespace) -> None:
    run_dir = args.evals_dir / args.run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    metrics = _aggregate_metrics(run_dir)
    print(f"Wrote metrics.json for {args.run_id}")
    print(f"  evaluator: {metrics.get('evaluator')}")


def _aggregate_metrics(run_dir: Path) -> dict[str, object]:
    """Read records + config from ``run_dir``, compute aggregates, write ``metrics.json``.

    Shared by ``vs-stockfish`` (after a fresh run) and ``report`` (over an
    existing run). Reading records back from disk rather than threading them
    through the evaluator keeps the contract identical in both paths: the
    on-disk ``records.jsonl`` is the single input to ``compute_all``.
    """
    reader = RunReader(run_dir)
    config = reader.read_config()
    records = reader.read_records()
    metrics = compute_all(records, config)
    RunWriter(run_dir).write_metrics(metrics)
    return metrics


def _halluci_won(outcome: GameOutcome) -> bool:
    return (outcome.result == "1-0" and outcome.halluci_color == chess.WHITE) or (outcome.result == "0-1" and outcome.halluci_color == chess.BLACK)


def _print_summary(outcomes: list[GameOutcome], run_dir: Path) -> None:
    wins = sum(1 for o in outcomes if _halluci_won(o))
    draws = sum(1 for o in outcomes if o.result == "1/2-1/2")
    unfinished = sum(1 for o in outcomes if o.result == "*")
    losses = len(outcomes) - wins - draws - unfinished
    total_scored = wins + draws + losses
    score = wins + 0.5 * draws
    pct = (score / total_scored * 100.0) if total_scored else 0.0

    print("\n=== Summary ===")
    print(f"Games played: {len(outcomes)}  (unfinished: {unfinished})")
    print(f"halluci-mate: {wins}W / {draws}D / {losses}L  —  score {score:.1f}/{total_scored} ({pct:.1f}%)")
    print(f"Artifacts:    {run_dir}")

    for i, outcome in enumerate(outcomes, start=1):
        side = "White" if outcome.halluci_color == chess.WHITE else "Black"
        print(f"  Game {i}: halluci-mate as {side} — {outcome.result} ({outcome.termination}, {outcome.ply_count} plies)")


if __name__ == "__main__":
    main()
