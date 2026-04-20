"""Play halluci-mate against Stockfish over one or more games and report results."""

# NOTE: This script is kinda a pile of shit; it's useful to reference but we shouldn't use it for patterns.
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from halluci_mate.game.game import Game, Perspective
from halluci_mate.inference import ChessInferenceEngine

STOCKFISH_SKILL_MIN = 0
STOCKFISH_SKILL_MAX = 20
DEFAULT_MAX_PLIES = 400


@dataclass
class GameResult:
    white: str
    black: str
    result: str
    halluci_color: chess.Color
    ply_count: int
    termination: str
    pgn: str


def _play_one_game(
    halluci: ChessInferenceEngine,
    stockfish: chess.engine.SimpleEngine,
    *,
    halluci_color: chess.Color,
    stockfish_limit: chess.engine.Limit,
    max_plies: int,
) -> GameResult:
    perspective = Perspective.WHITE if halluci_color == chess.WHITE else Perspective.BLACK
    game = Game(board=chess.Board(), perspective=perspective)

    pgn = chess.pgn.Game()
    pgn.headers["White"] = "halluci-mate" if halluci_color == chess.WHITE else "Stockfish"
    pgn.headers["Black"] = "Stockfish" if halluci_color == chess.WHITE else "halluci-mate"
    node: chess.pgn.GameNode = pgn

    termination = "natural"
    while not game.board.is_game_over(claim_draw=True):
        if game.board.ply() >= max_plies:
            termination = "max-plies"
            break

        if game.board.turn == halluci_color:
            move = halluci.predict(game)
        else:
            move = stockfish.play(game.board, stockfish_limit).move
            if move is None:
                termination = "stockfish-resigned"
                break

        game.push_move(move)
        node = node.add_variation(move)

    result = game.board.result(claim_draw=True) if termination == "natural" else "*"
    pgn.headers["Result"] = result

    return GameResult(
        white=pgn.headers["White"],
        black=pgn.headers["Black"],
        result=result,
        halluci_color=halluci_color,
        ply_count=game.board.ply(),
        termination=termination,
        pgn=str(pgn),
    )


def _score_for_halluci(result: GameResult) -> float:
    if result.result == "1-0":
        return 1.0 if result.halluci_color == chess.WHITE else 0.0
    if result.result == "0-1":
        return 1.0 if result.halluci_color == chess.BLACK else 0.0
    if result.result == "1/2-1/2":
        return 0.5
    return 0.0


def _summarize(results: list[GameResult]) -> None:
    wins = sum(1 for r in results if _score_for_halluci(r) == 1.0 and r.result != "*")
    draws = sum(1 for r in results if r.result == "1/2-1/2")
    losses = sum(1 for r in results if _score_for_halluci(r) == 0.0 and r.result != "*")
    unfinished = sum(1 for r in results if r.result == "*")
    total_scored = wins + draws + losses
    score = wins + 0.5 * draws
    pct = (score / total_scored * 100.0) if total_scored else 0.0

    print("\n=== Summary ===")
    print(f"Games played: {len(results)}  (unfinished: {unfinished})")
    print(f"halluci-mate: {wins}W / {draws}D / {losses}L  —  score {score:.1f}/{total_scored} ({pct:.1f}%)")

    print("\n=== PGNs ===")
    for i, r in enumerate(results, start=1):
        halluci_side = "White" if r.halluci_color == chess.WHITE else "Black"
        print(f"\nGame {i} — halluci-mate as {halluci_side} — {r.result} ({r.termination})")
        print(r.pgn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play halluci-mate against Stockfish.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained halluci-mate checkpoint directory.")
    parser.add_argument("--stockfish", type=str, default="stockfish", help="Path to the stockfish binary (default: 'stockfish' on PATH).")
    parser.add_argument("--games", type=int, default=2, help="Number of games to play (default: 2).")
    parser.add_argument(
        "--halluci-color",
        choices=["white", "black", "alternate"],
        default="alternate",
        help="Which color halluci-mate plays. 'alternate' flips each game starting with white.",
    )
    parser.add_argument("--stockfish-skill", type=int, default=0, help=f"Stockfish 'Skill Level' UCI option [{STOCKFISH_SKILL_MIN}-{STOCKFISH_SKILL_MAX}] (default: 0, weakest).")
    parser.add_argument("--stockfish-depth", type=int, default=1, help="Stockfish search depth per move (default: 1).")
    parser.add_argument("--stockfish-movetime", type=float, default=None, help="Stockfish time per move in seconds. Overrides --stockfish-depth if set.")
    parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES, help=f"Abort a game after this many plies (default: {DEFAULT_MAX_PLIES}).")
    parser.add_argument("--unconstrained", action="store_true", help="Disable legal-move masking for halluci-mate. Illegal samples fall back to constrained.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for halluci-mate (0.0 = greedy).")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling cutoff (0 = disabled).")
    parser.add_argument("--device", type=str, default=None, help="Torch device (default: cuda if available, else cpu).")
    parser.add_argument("--pgn-out", type=Path, default=None, help="Optional path to write all games as a single PGN file.")
    args = parser.parse_args()

    if not (STOCKFISH_SKILL_MIN <= args.stockfish_skill <= STOCKFISH_SKILL_MAX):
        parser.error(f"--stockfish-skill must be in [{STOCKFISH_SKILL_MIN}, {STOCKFISH_SKILL_MAX}]")

    halluci = ChessInferenceEngine.from_checkpoint(
        args.checkpoint,
        constrained=not args.unconstrained,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )

    stockfish_limit = chess.engine.Limit(time=args.stockfish_movetime) if args.stockfish_movetime is not None else chess.engine.Limit(depth=args.stockfish_depth)

    stockfish = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    try:
        stockfish.configure({"Skill Level": args.stockfish_skill})

        results: list[GameResult] = []
        for i in range(args.games):
            if args.halluci_color == "white":
                color = chess.WHITE
            elif args.halluci_color == "black":
                color = chess.BLACK
            else:
                color = chess.WHITE if i % 2 == 0 else chess.BLACK

            print(f"\n--- Game {i + 1}/{args.games}: halluci-mate as {'White' if color == chess.WHITE else 'Black'} ---")
            result = _play_one_game(
                halluci,
                stockfish,
                halluci_color=color,
                stockfish_limit=stockfish_limit,
                max_plies=args.max_plies,
            )
            results.append(result)
            print(f"Result: {result.result}  plies: {result.ply_count}  termination: {result.termination}")
    finally:
        stockfish.quit()

    _summarize(results)

    if args.pgn_out is not None:
        args.pgn_out.write_text("\n\n".join(r.pgn for r in results) + "\n")
        print(f"\nPGN written to {args.pgn_out}")


if __name__ == "__main__":
    main()
