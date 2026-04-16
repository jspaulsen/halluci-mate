"""CLI entry to launch the halluci-mate UCI engine."""

from __future__ import annotations

import argparse
from pathlib import Path

from halluci_mate.inference import ChessInferenceEngine
from halluci_mate.uci_engine import UciEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="halluci-mate UCI chess engine.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained checkpoint directory.",
    )
    parser.add_argument(
        "--unconstrained",
        action="store_true",
        help="Disable legal-move masking and sample freely. Illegal samples fall back to constrained mode.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy argmax, the default).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling cutoff (0 = disabled, the default).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: cuda if available, else cpu).",
    )
    args = parser.parse_args()

    engine = ChessInferenceEngine.from_checkpoint(
        args.checkpoint,
        constrained=not args.unconstrained,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )
    UciEngine(engine=engine).run()


if __name__ == "__main__":
    main()
