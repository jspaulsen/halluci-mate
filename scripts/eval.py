"""halluci-mate eval harness CLI.

Single entry point for running evaluators against a checkpoint and for
recomputing aggregates over an existing run directory. Wired-up subcommands:

* ``vs-stockfish`` — play N games against Stockfish, write the HAL-5 run
  directory, and aggregate ``metrics.json`` via ``compute_all``.
* ``legal-rate`` — run the model unconstrained on a set of positions
  (FENs or PGN-sampled) and record top-1 legality.
* ``perplexity`` — token-level cross-entropy on held-out game sequences.
* ``report <run-id>`` — recompute ``metrics.json`` from an existing
  ``records.jsonl`` + ``config.json``. No re-run.

Additional evaluators land per the build order in ``docs/eval_harness.md``.

The ``--checkpoint`` argument accepts either a local checkpoint directory
(e.g. ``runs-v1/marvelous-deer-608/checkpoint-9660``) or a Hugging Face repo
id (e.g. ``jspaulsen/halluci-mate-v1a``); both are handled by
``AutoModelForCausalLM.from_pretrained`` under the hood.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, cast

import chess
import chess.engine
import click.exceptions
import typer

from halluci_mate.eval.evaluators.legal_rate import DEFAULT_SAMPLE_N, LegalRateConfig, run_legal_rate
from halluci_mate.eval.evaluators.perplexity import PerplexityConfig, run_perplexity
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

HalluciColor = Literal["white", "black", "alternate"]

app = typer.Typer(
    help="halluci-mate eval harness CLI.",
    add_completion=False,
    pretty_exceptions_enable=False,
    no_args_is_help=True,
)


@app.command("vs-stockfish", help="Play N games against Stockfish and emit a HAL-5 run directory.")
def vs_stockfish_cmd(
    checkpoint: Annotated[str, typer.Option(help="Local checkpoint directory or Hugging Face repo id.")],
    checkpoint_tag: Annotated[str | None, typer.Option(help="Tag used in run-id; defaults to a sanitized form of --checkpoint. User-supplied tags must not contain '_'.")] = None,
    evals_dir: Annotated[Path, typer.Option(help=f"Parent directory for run outputs (default: {DEFAULT_EVALS_DIR}).")] = DEFAULT_EVALS_DIR,
    device: Annotated[str | None, typer.Option(help="Torch device (default: cuda if available, else cpu).")] = None,
    stockfish: Annotated[str, typer.Option(help="Path to the stockfish binary (default: 'stockfish' on PATH).")] = "stockfish",
    games: Annotated[int, typer.Option(help="Number of games to play (default: 2).")] = 2,
    halluci_color: Annotated[HalluciColor, typer.Option(help="Which color halluci-mate plays. 'alternate' flips each game starting with white.")] = "alternate",
    stockfish_skill: Annotated[int, typer.Option(help=f"Stockfish 'Skill Level' UCI option [{STOCKFISH_SKILL_MIN}-{STOCKFISH_SKILL_MAX}] (default: 0, weakest).")] = 0,
    stockfish_depth: Annotated[int, typer.Option(help="Stockfish search depth per move (default: 1).")] = 1,
    stockfish_movetime: Annotated[float | None, typer.Option(help="Stockfish time per move in seconds. Overrides --stockfish-depth if set.")] = None,
    max_plies: Annotated[int, typer.Option(help="Abort a game after this many plies (default: 400).")] = 400,
    temperature: Annotated[float, typer.Option(help="Sampling temperature for halluci-mate (0.0 = greedy).")] = 0.0,
    top_k: Annotated[int, typer.Option(help="Top-k sampling cutoff (0 = disabled).")] = 0,
    unconstrained: Annotated[bool, typer.Option("--unconstrained", help="Disable legal-move masking for halluci-mate.")] = False,
    record_top_k: Annotated[int, typer.Option(help="K for the top-K candidates captured per record (0 = skip).")] = 5,
    blunder_threshold_cp: Annotated[int, typer.Option(help="Centipawn loss threshold for is_blunder.")] = 200,
) -> None:
    config = VsStockfishConfig(
        games=games,
        halluci_color=halluci_color,
        stockfish_skill=stockfish_skill,
        stockfish_depth=stockfish_depth if stockfish_movetime is None else None,
        stockfish_movetime=stockfish_movetime,
        max_plies=max_plies,
        unconstrained=unconstrained,
        record_top_k=record_top_k,
        blunder_threshold_cp=blunder_threshold_cp,
    )

    run_id, run_dir = _resolve_run_dir(checkpoint, checkpoint_tag, evals_dir, Evaluator.VS_STOCKFISH)

    engine = ChessInferenceEngine.from_checkpoint(
        checkpoint,
        constrained=not unconstrained,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )

    # Source-of-truth for sampling params is the engine itself; persist its
    # effective values so the on-disk record can't desync from what was used.
    extra_config: dict[str, object] = {"temperature": engine.temperature, "top_k": engine.top_k}

    sf_engine = chess.engine.SimpleEngine.popen_uci(stockfish)
    try:
        outcomes = run_vs_stockfish(
            engine=engine,
            stockfish=sf_engine,
            config=config,
            run_dir=run_dir,
            run_id=run_id,
            checkpoint=str(checkpoint),
            extra_config=extra_config,
        )
    finally:
        sf_engine.quit()

    _aggregate_metrics(run_dir)
    _print_summary(outcomes, run_dir)


@app.command("legal-rate", help="Run the model unconstrained on a set of positions and record top-1 legality.")
def legal_rate_cmd(
    checkpoint: Annotated[str, typer.Option(help="Local checkpoint directory or Hugging Face repo id.")],
    checkpoint_tag: Annotated[str | None, typer.Option(help="Tag used in run-id; defaults to a sanitized form of --checkpoint. User-supplied tags must not contain '_'.")] = None,
    evals_dir: Annotated[Path, typer.Option(help=f"Parent directory for run outputs (default: {DEFAULT_EVALS_DIR}).")] = DEFAULT_EVALS_DIR,
    device: Annotated[str | None, typer.Option(help="Torch device (default: cuda if available, else cpu).")] = None,
    positions: Annotated[Path | None, typer.Option(help="Path to a file with one FEN per line.")] = None,
    sample_from_games: Annotated[Path | None, typer.Option(help="Path to a PGN file; positions are reservoir-sampled across all (game, ply) pairs.")] = None,
    n: Annotated[int, typer.Option(help=f"Number of positions to sample from the PGN source (default: {DEFAULT_SAMPLE_N}). Ignored when --positions is set.")] = DEFAULT_SAMPLE_N,
    seed: Annotated[int, typer.Option(help="Seed for the PGN sampler (default: 0). Ignored when --positions is set.")] = 0,
) -> None:
    if (positions is None) == (sample_from_games is None):
        raise typer.BadParameter("exactly one of --positions / --sample-from-games is required.")

    config = LegalRateConfig(
        positions_path=positions,
        sample_from_games_path=sample_from_games,
        sample_n=n,
        seed=seed,
    )

    run_id, run_dir = _resolve_run_dir(checkpoint, checkpoint_tag, evals_dir, Evaluator.LEGAL_RATE)

    engine = ChessInferenceEngine.from_checkpoint(checkpoint, constrained=False, device=device)

    # No `extra_config` here: legal-rate uses argmax for the legality bit, so
    # there are no engine-side sampling knobs whose effective values we'd need
    # to persist into `config.json`. Mirrors the comment in vs-stockfish.
    n_records = run_legal_rate(
        engine=engine,
        config=config,
        run_dir=run_dir,
        run_id=run_id,
        checkpoint=str(checkpoint),
    )

    metrics = _aggregate_metrics(run_dir)
    # Trust the on-disk schema: `compute_all` for `LEGAL_RATE` always emits
    # `legal_rate.overall.rate`, pinned by `metrics_test.test_compute_all_legal_rate_*`.
    legal_rate_block = cast("dict[str, dict[str, float]]", metrics["legal_rate"])
    rate = legal_rate_block["overall"]["rate"]
    print("\n=== Summary ===")
    print(f"Positions scored: {n_records}")
    print(f"legal_rate:       {rate:.4f}")
    print(f"Artifacts:        {run_dir}")


@app.command("perplexity", help="Score held-out game sequences and emit per-position token logprobs.")
def perplexity_cmd(
    checkpoint: Annotated[str, typer.Option(help="Local checkpoint directory or Hugging Face repo id.")],
    data: Annotated[Path, typer.Option(help="Path to a jsonl file of held-out sequences.")],
    checkpoint_tag: Annotated[str | None, typer.Option(help="Tag used in run-id; defaults to a sanitized form of --checkpoint. User-supplied tags must not contain '_'.")] = None,
    evals_dir: Annotated[Path, typer.Option(help=f"Parent directory for run outputs (default: {DEFAULT_EVALS_DIR}).")] = DEFAULT_EVALS_DIR,
    device: Annotated[str | None, typer.Option(help="Torch device (default: cuda if available, else cpu).")] = None,
    max_sequences: Annotated[int | None, typer.Option(help="Stop after this many sequences (default: process all).")] = None,
) -> None:
    config = PerplexityConfig(data_path=data, max_sequences=max_sequences)

    run_id, run_dir = _resolve_run_dir(checkpoint, checkpoint_tag, evals_dir, Evaluator.PERPLEXITY)

    engine = ChessInferenceEngine.from_checkpoint(checkpoint, constrained=False, device=device)

    # No `extra_config` here: perplexity scores known token sequences and
    # never samples, so there are no engine-side sampling knobs to persist.
    # Mirrors the comment in vs-stockfish.
    n_records = run_perplexity(
        engine=engine,
        config=config,
        run_dir=run_dir,
        run_id=run_id,
        checkpoint=str(checkpoint),
    )

    metrics = _aggregate_metrics(run_dir)
    # Trust the on-disk schema: `compute_all` for `PERPLEXITY` always emits
    # these four keys, pinned by `metrics_test.test_compute_all_perplexity_*`
    # (including the empty-records case). `.get(..., default)` would hide a
    # schema regression by silently printing zero.
    num_tokens = cast("int", metrics["num_tokens"])
    mean_nll = cast("float", metrics["mean_nll"])
    bits_per_token = cast("float", metrics["bits_per_token"])
    print("\n=== Summary ===")
    print(f"Sequences scored: {n_records}")
    print(f"Tokens scored:    {num_tokens}")
    print(f"mean NLL:         {mean_nll:.4f}")
    print(f"bits/token:       {bits_per_token:.4f}")
    print(f"Artifacts:        {run_dir}")


@app.command("report", help="Recompute metrics.json from an existing run directory's records.jsonl.")
def report_cmd(
    run_id: Annotated[str, typer.Argument(help="Run id under --evals-dir; the run directory must already contain records.jsonl and config.json.")],
    evals_dir: Annotated[Path, typer.Option(help=f"Parent directory holding the run (default: {DEFAULT_EVALS_DIR}).")] = DEFAULT_EVALS_DIR,
) -> None:
    run_dir = evals_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"run path is not a directory: {run_dir}")
    metrics = _aggregate_metrics(run_dir)
    print(f"Wrote metrics.json for {run_id}")
    print(f"  evaluator: {metrics['evaluator']}")


def _resolve_run_dir(checkpoint: str, checkpoint_tag: str | None, evals_dir: Path, evaluator: Evaluator) -> tuple[str, Path]:
    """Build the ``(run_id, run_dir)`` pair shared by every fresh-run subcommand and announce it."""
    tag = resolve_checkpoint_tag(checkpoint, checkpoint_tag)
    run_id = make_run_id(tag, evaluator)
    run_dir = evals_dir / run_id
    print(f"Run id: {run_id}")
    print(f"Run dir: {run_dir}")
    return run_id, run_dir


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


def main(argv: list[str] | None = None) -> None:
    """CLI entry point. ``argv`` is exposed for in-process smoke tests.

    Uses click's ``standalone_mode=False`` so domain exceptions (e.g.
    ``FileNotFoundError`` from ``report``) propagate to callers, while
    click/typer's own usage and exit signals are still converted to
    ``SystemExit`` for argparse-parity (tests rely on both contracts).
    """
    try:
        app(args=argv, prog_name="eval", standalone_mode=False)
    except click.exceptions.UsageError as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from exc
    except click.exceptions.Exit as exc:
        raise SystemExit(exc.exit_code) from exc
    except click.exceptions.Abort as exc:
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
