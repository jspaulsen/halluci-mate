"""``legal_rate`` evaluator: run the model unconstrained on positions, record top-1 legality.

Source signal for Phase 2 (legality) DPO progress. The evaluator drives the
``Predictor`` through a flat list of positions and records one
``PerLegalRateRecord`` per position. Position sources are mutually exclusive:

* ``--positions <file>`` — one FEN per line; the model has no move history.
* ``--sample-from-games <pgn>`` ``--n N`` — reservoir-sample N (game, ply) pairs
  from a PGN, replay moves up to each ply, and score the resulting position
  with the real prefix on the board.

The per-position record schema (`docs/eval_harness.md` §Per-legal-rate record)
captures only the unconstrained top-1 + a legality bit; it deliberately does
not duplicate the ``vs_stockfish`` per-move shape.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import chess
import chess.pgn

from halluci_mate.eval.records import Evaluator, PerLegalRateRecord
from halluci_mate.eval.runs import RunWriter
from halluci_mate.game import Game, Perspective

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from halluci_mate.inference import Predictor


@dataclass(frozen=True)
class LegalRateConfig:
    """Knobs for ``run_legal_rate``.

    Exactly one of ``positions_path`` / ``sample_from_games_path`` must be set;
    enforced in ``__post_init__`` so a misconfigured run fails before any
    inference work happens. ``sample_n`` and ``seed`` only carry meaning under
    the PGN-sampling source.
    """

    positions_path: Path | None = None
    sample_from_games_path: Path | None = None
    sample_n: int = 10_000
    seed: int = 0

    def __post_init__(self) -> None:
        sources = [self.positions_path, self.sample_from_games_path]
        set_count = sum(s is not None for s in sources)
        if set_count != 1:
            raise ValueError("exactly one of positions_path / sample_from_games_path must be set")
        if self.sample_from_games_path is not None and self.sample_n < 1:
            raise ValueError(f"sample_n must be >= 1; got {self.sample_n}")


@dataclass(frozen=True)
class _Position:
    """One sampled position handed to the evaluator's inner loop."""

    position_id: str
    game: Game


def run_legal_rate(
    *,
    engine: Predictor,
    config: LegalRateConfig,
    run_dir: Path,
    run_id: str,
    checkpoint: str,
    extra_config: dict[str, object] | None = None,
) -> int:
    """Run the model unconstrained on positions sourced per ``config`` and write a run dir.

    Returns the number of records emitted (one per position). Caller manages
    the lifecycle of ``engine``. ``extra_config`` is merged into ``config.json``;
    same pattern as ``run_vs_stockfish`` so the CLI can persist the engine's
    effective sampling parameters without putting them on ``LegalRateConfig``.
    """
    writer = RunWriter(run_dir)
    writer.write_config(_build_config_payload(config=config, run_id=run_id, checkpoint=checkpoint, extra=extra_config))

    event_id = 0
    with writer:
        for position in _iter_positions(config):
            prediction = engine.predict_with_metadata(position.game, constrained=False, record_top_k=0)
            writer.append_record(
                PerLegalRateRecord(
                    run_id=run_id,
                    event_id=event_id,
                    evaluator=Evaluator.LEGAL_RATE,
                    checkpoint=checkpoint,
                    position_id=position.position_id,
                    fen=position.game.board.fen(),
                    model_move=prediction.raw_sample_move_uci,
                    legal=prediction.raw_sample_legal,
                )
            )
            event_id += 1
            position.game.reset_cache()
    return event_id


def _build_config_payload(*, config: LegalRateConfig, run_id: str, checkpoint: str, extra: dict[str, object] | None) -> dict[str, object]:
    payload: dict[str, object] = {
        "evaluator": Evaluator.LEGAL_RATE.value,
        "run_id": run_id,
        "checkpoint": checkpoint,
    }
    payload.update(asdict(config))
    # ``Path`` does not survive ``json.dumps``; normalize to strings so the
    # on-disk config round-trips. Done here (not on the dataclass) so the
    # in-memory config keeps real ``Path`` ergonomics.
    for key in ("positions_path", "sample_from_games_path"):
        value = payload.get(key)
        if value is not None:
            payload[key] = str(value)
    if extra:
        payload.update(extra)
    return payload


def _iter_positions(config: LegalRateConfig) -> Iterator[_Position]:
    if config.positions_path is not None:
        yield from _iter_fen_file(config.positions_path)
        return
    assert config.sample_from_games_path is not None  # guaranteed by __post_init__
    yield from _iter_pgn_samples(config.sample_from_games_path, n=config.sample_n, seed=config.seed)


def _iter_fen_file(path: Path) -> Iterator[_Position]:
    with path.open(encoding="utf-8") as fp:
        for lineno, raw in enumerate(fp, start=1):
            fen = raw.strip()
            if not fen:
                continue
            try:
                board = chess.Board(fen)
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: invalid FEN {fen!r}: {exc}") from exc
            game = Game(board=board, perspective=_perspective_for_turn(board.turn))
            yield _Position(position_id=f"{path.name}:{lineno}", game=game)


def _iter_pgn_samples(path: Path, *, n: int, seed: int) -> Iterator[_Position]:
    """Yield ``n`` random (game, ply) positions reservoir-sampled from ``path``.

    Each sample is the *pre-move* state at a given ply (i.e. the model is
    asked to predict the move at that ply). The sample is materialized by
    replaying moves onto a fresh ``chess.Board`` so ``Game.move_stack`` carries
    the real prefix into ``predict_with_metadata``.

    The reservoir is built eagerly: the entire PGN is read before the first
    record is emitted, so a multi-GB Lichess corpus produces a long opening
    silence followed by a fast scoring phase. Acceptable at v1's documented
    10k-position scale; future work to stream + cap memory if the source
    PGN grows past tens of thousands of games.
    """
    rng = random.Random(seed)
    entries = _reservoir_sample_pgn(path, n=n, rng=rng)
    for entry in entries:
        board = chess.Board()
        for move in entry.moves:
            board.push(move)
        game = Game(board=board, perspective=_perspective_for_turn(board.turn))
        yield _Position(position_id=f"{entry.game_id}:ply{entry.ply}", game=game)


@dataclass(frozen=True)
class _ReservoirEntry:
    game_id: str
    ply: int
    moves: tuple[chess.Move, ...]


def _reservoir_sample_pgn(path: Path, *, n: int, rng: random.Random) -> list[_ReservoirEntry]:
    """One-pass reservoir sample of (game, ply) positions across all PGN games.

    Memory cost is proportional to ``n * mean_ply``: each entry snapshots the
    move list up to its ply (so the entry can replay the position on demand).
    At ``n=10_000`` and mean ply ~80 that's ~800k ``chess.Move`` references,
    in the tens-of-MB range. A leaner shape would store ``(game_index, ply)``
    indices and reread the PGN to materialize the prefix; revisit if memory
    becomes a real problem at scale.
    """
    reservoir: list[_ReservoirEntry] = []
    seen = 0
    with path.open(encoding="utf-8") as pgn_fp:
        for game_index, game in enumerate(iter(lambda: chess.pgn.read_game(pgn_fp), None)):
            game_id = f"game-{game_index:04d}"
            moves: list[chess.Move] = []
            for node in game.mainline():
                entry = _ReservoirEntry(game_id=game_id, ply=len(moves), moves=tuple(moves))
                seen += 1
                if len(reservoir) < n:
                    reservoir.append(entry)
                else:
                    j = rng.randint(0, seen - 1)
                    if j < n:
                        reservoir[j] = entry
                moves.append(node.move)
    return reservoir


def _perspective_for_turn(turn: chess.Color) -> Perspective:
    return Perspective.WHITE if turn == chess.WHITE else Perspective.BLACK
