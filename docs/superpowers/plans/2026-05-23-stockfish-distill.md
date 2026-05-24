# Stockfish Distillation Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-stage pipeline that generates a from-scratch pretraining corpus of full Stockfish games (seeded from high-Elo Lichess openings) and tokenizes it into the existing `train/eval/test.parquet` training format.

**Architecture:** Stage 1 (`generate_distill_games.py` → `distill/engine_selfplay.py` + `distill/raw_shards.py`) plays full-strength Stockfish games with a MultiPV "wobble" for diversity and writes resumable raw game shards to disk. Stage 2 (`prepare_distill_data.py` → `data_preparation.py`) tokenizes those raw shards into stratified splits, reusing the existing pipeline with `elo_bucket` dropped from the stratification key.

**Tech Stack:** Python 3.12, `python-chess` (`chess.engine` UCI for Stockfish), `datasets`/`pyarrow` (parquet shards), `pydantic` (row schema/validation), `typer` (CLIs), `pytest`. All Python via `uv run`.

**Spec:** `docs/superpowers/specs/2026-05-23-stockfish-distill-design.md`

**Conventions for every task:** Run `uv run pytest <path> -v` for tests, `uv run ruff check . && uv run ruff format . && uv run ty check` before each commit. Catch only specific exceptions. Type-annotate every signature. Keep functions under 40 lines.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/halluci_mate/distill/__init__.py` | Empty package marker (no barrel imports). |
| `src/halluci_mate/distill/engine_selfplay.py` | `SelfPlayConfig`, score helpers, `select_wobble_move`, `_AdjudicationState`, `play_seed`, `SelfPlayGame`. |
| `src/halluci_mate/distill/raw_shards.py` | `EngineMeta`, `RawGameRow` (pydantic), `game_to_row`, `ShardWriter`, `read_raw_games`. |
| `src/halluci_mate/distill/seeds.py` | `SeedOpening`, `iter_seed_openings` (filtered Lichess stream → first-N-ply seeds). |
| `src/halluci_mate/distill/generate.py` | `run_generation` (serial, resumable driver wiring seeds → `play_seed` → `ShardWriter`). |
| `src/halluci_mate/distill/tokenize.py` | `process_engine_game` + `tokenize_engine_shards` (raw shards → tokenized shards; `distill → core` deps only). |
| `src/halluci_mate/data_preparation.py` | **Modify (core-only):** parameterize `build_stratified_splits`/`save_splits` stratify columns. No `distill` import. |
| `scripts/generate_distill_games.py` | Stage 1 typer CLI; opens Stockfish (optionally a `--workers` pool) and calls the driver. |
| `scripts/prepare_distill_data.py` | Stage 2 typer CLI; tokenize raw shards → splits. |
| `tests/distill/engine_selfplay_test.py` | Unit tests for config, helpers, wobble, adjudication, `play_seed` (stub engine). |
| `tests/distill/raw_shards_test.py` | Unit tests for row schema + `ShardWriter` (atomicity, resume). |
| `tests/distill/seeds_test.py` | Unit tests for seed filtering/extraction (in-memory fake stream). |
| `tests/distill/generate_test.py` | Unit tests for `run_generation` driver + resume (stub engine). |
| `tests/data_preparation_test.py` | **Modify/add:** stratify-column parameterization of `build_stratified_splits`. |
| `tests/distill/tokenize_test.py` | Unit tests for `process_engine_game` + `tokenize_engine_shards`. |
| `tests/integration/distill_pipeline_test.py` | End-to-end with real Stockfish at shallow depth (excluded from default `pytest` run via `norecursedirs`). |

---

## Task 1: Package marker + `SelfPlayConfig`

**Files:**
- Create: `src/halluci_mate/distill/__init__.py`
- Create: `src/halluci_mate/distill/engine_selfplay.py`
- Test: `tests/distill/engine_selfplay_test.py`

- [ ] **Step 1: Create the empty package marker**

```python
# src/halluci_mate/distill/__init__.py
```

(Leave the file empty — minimal `__init__.py` per CLAUDE.md.)

- [ ] **Step 2: Write the failing test**

```python
# tests/distill/engine_selfplay_test.py
from __future__ import annotations

import pytest

from halluci_mate.distill.engine_selfplay import SelfPlayConfig


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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/distill/engine_selfplay_test.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError` for `SelfPlayConfig`.

- [ ] **Step 4: Write the implementation**

```python
# src/halluci_mate/distill/engine_selfplay.py
"""Full-strength Stockfish self-play with a MultiPV diversity 'wobble'.

Seeds from a human opening prefix, then plays both sides with Stockfish,
sampling among near-equal moves so a fixed-budget (otherwise deterministic)
engine produces a diverse corpus. Games are adjudicated (resign / draw / cap)
to stay finite and avoid dead-drawn shuffles.
"""

from __future__ import annotations

from dataclasses import dataclass

# Mate scores collapse to a finite cp so adjudication/selection arithmetic
# never sees ``None`` (mirrors vs_stockfish ``_MATE_SCORE_CP``).
_MATE_SCORE_CP = 100_000


@dataclass(frozen=True)
class SelfPlayConfig:
    """Knobs for one self-play generation run. ``depth`` is the cost dial."""

    depth: int = 18
    multipv: int = 4
    wobble_cp: int = 30
    wobble_temp: float = 50.0
    seed_plies: int = 12
    resign_cp: int = 700
    resign_plies: int = 4
    draw_cp: int = 15
    draw_plies: int = 8
    draw_min_ply: int = 60
    max_plies: int = 200

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1; got {self.depth}")
        if self.multipv < 1:
            raise ValueError(f"multipv must be >= 1; got {self.multipv}")
        if self.wobble_cp < 0:
            raise ValueError(f"wobble_cp must be >= 0; got {self.wobble_cp}")
        if self.wobble_temp <= 0:
            raise ValueError(f"wobble_temp must be > 0; got {self.wobble_temp}")
        if self.seed_plies < 0:
            raise ValueError(f"seed_plies must be >= 0; got {self.seed_plies}")
        if self.max_plies < 1:
            raise ValueError(f"max_plies must be >= 1; got {self.max_plies}")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/distill/engine_selfplay_test.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add src/halluci_mate/distill/__init__.py src/halluci_mate/distill/engine_selfplay.py tests/distill/engine_selfplay_test.py
git commit -m "feat(distill): add SelfPlayConfig with validation"
```

---

## Task 2: Score helpers (`_stm_cp`, `_best_white_cp`, `_pv_first_move`, `_softmax`)

These convert `chess.engine` analysis into the integers the wobble and adjudicator need. `analyse(..., multipv=K)` returns a `list[InfoDict]` ordered best-first; each `InfoDict` has a `"score"` (`PovScore`) and a `"pv"` (move list).

**Files:**
- Modify: `src/halluci_mate/distill/engine_selfplay.py`
- Test: `tests/distill/engine_selfplay_test.py`

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/distill/engine_selfplay_test.py
import chess
import chess.engine

from halluci_mate.distill.engine_selfplay import (
    _best_white_cp,
    _pv_first_move,
    _softmax,
    _stm_cp,
)


def _info(cp: int, pv_uci: str | None) -> chess.engine.InfoDict:
    info: chess.engine.InfoDict = {"score": chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE)}
    if pv_uci is not None:
        info["pv"] = [chess.Move.from_uci(pv_uci)]
    return info


def test_stm_cp_white_to_move_is_white_relative() -> None:
    assert _stm_cp(_info(80, "e2e4"), chess.WHITE) == 80


def test_stm_cp_black_to_move_flips_sign() -> None:
    # +80 white-relative is -80 from Black's perspective.
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/distill/engine_selfplay_test.py -v`
Expected: FAIL — helpers not defined.

- [ ] **Step 3: Implement the helpers**

```python
# add to src/halluci_mate/distill/engine_selfplay.py
import math

import chess
import chess.engine


def _stm_cp(info: chess.engine.InfoDict, turn: chess.Color) -> int:
    """Side-to-move-relative centipawns for one analysis line (mates clamped)."""
    score = info["score"].pov(turn).score(mate_score=_MATE_SCORE_CP)
    assert score is not None  # ``score(mate_score=...)`` only returns None for MateGiven
    return score


def _best_white_cp(infos: list[chess.engine.InfoDict]) -> int:
    """White-relative centipawns of the best line (``infos[0]``), mates clamped."""
    score = infos[0]["score"].white().score(mate_score=_MATE_SCORE_CP)
    assert score is not None
    return score


def _pv_first_move(info: chess.engine.InfoDict) -> chess.Move | None:
    """First move of an analysis line's principal variation, or ``None``."""
    pv = info.get("pv")
    if not pv:
        return None
    return pv[0]


def _softmax(values: list[float]) -> list[float]:
    """Numerically-stable softmax over ``values``."""
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]
```

(Place the `import math` / `import chess` lines with the module's other top-level imports, not inline.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/distill/engine_selfplay_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/halluci_mate/distill/engine_selfplay.py tests/distill/engine_selfplay_test.py
git commit -m "feat(distill): add score/softmax helpers for self-play"
```

---

## Task 3: `select_wobble_move`

Picks one move: filter to lines within `wobble_cp` of the best STM score, then softmax-sample by `score / wobble_temp`. RNG is injected so selection is deterministic in tests.

**Files:**
- Modify: `src/halluci_mate/distill/engine_selfplay.py`
- Test: `tests/distill/engine_selfplay_test.py`

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/distill/engine_selfplay_test.py
import random

from halluci_mate.distill.engine_selfplay import SelfPlayConfig, select_wobble_move


def test_wobble_filters_out_of_band_moves() -> None:
    # Black to move: white-relative scores invert. Make e2e4 best for Black.
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/distill/engine_selfplay_test.py -v`
Expected: FAIL — `select_wobble_move` not defined.

- [ ] **Step 3: Implement**

```python
# add to src/halluci_mate/distill/engine_selfplay.py
import random  # with the other top-level imports


def select_wobble_move(
    infos: list[chess.engine.InfoDict],
    turn: chess.Color,
    config: SelfPlayConfig,
    rng: random.Random,
) -> chess.Move:
    """Sample a near-best move from MultiPV analysis (the diversity 'wobble').

    Keeps lines within ``config.wobble_cp`` of the best side-to-move score,
    then softmax-samples by ``cp / config.wobble_temp`` (lower temp favors the
    best move; larger temp approaches uniform). ``rng`` makes this reproducible.
    """
    candidates = [(move, _stm_cp(info, turn)) for info in infos if (move := _pv_first_move(info)) is not None]
    if not candidates:
        raise ValueError("analysis returned no candidate moves")
    best_cp = max(cp for _, cp in candidates)
    in_band = [(move, cp) for move, cp in candidates if best_cp - cp <= config.wobble_cp]
    weights = _softmax([cp / config.wobble_temp for _, cp in in_band])
    return rng.choices([move for move, _ in in_band], weights=weights, k=1)[0]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/distill/engine_selfplay_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/halluci_mate/distill/engine_selfplay.py tests/distill/engine_selfplay_test.py
git commit -m "feat(distill): add MultiPV wobble move selection"
```

---

## Task 4: `_AdjudicationState`

Tracks consecutive-ply runs to decide resign / draw / continue from the per-ply best-line white-relative eval.

**Files:**
- Modify: `src/halluci_mate/distill/engine_selfplay.py`
- Test: `tests/distill/engine_selfplay_test.py`

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/distill/engine_selfplay_test.py
from halluci_mate.distill.engine_selfplay import _AdjudicationState


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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/distill/engine_selfplay_test.py -v`
Expected: FAIL — `_AdjudicationState` not defined.

- [ ] **Step 3: Implement**

```python
# add to src/halluci_mate/distill/engine_selfplay.py
from typing import Literal

Outcome = Literal["white", "black", "draw"]
Termination = Literal["natural", "adjudicated-win", "adjudicated-draw", "max-plies"]


@dataclass
class _AdjudicationState:
    """Rolling consecutive-ply counters for resign/draw adjudication."""

    decisive_sign: int = 0
    decisive_run: int = 0
    draw_run: int = 0

    def update(self, white_cp: int, ply: int, config: SelfPlayConfig) -> tuple[Termination, Outcome] | None:
        """Feed the best-line white-relative eval; return a verdict or ``None``."""
        sign = 1 if white_cp >= config.resign_cp else -1 if white_cp <= -config.resign_cp else 0
        if sign != 0 and sign == self.decisive_sign:
            self.decisive_run += 1
        else:
            self.decisive_sign = sign
            self.decisive_run = 1 if sign != 0 else 0
        if self.decisive_run >= config.resign_plies:
            return ("adjudicated-win", "white" if sign > 0 else "black")

        if ply >= config.draw_min_ply and abs(white_cp) < config.draw_cp:
            self.draw_run += 1
        else:
            self.draw_run = 0
        if self.draw_run >= config.draw_plies:
            return ("adjudicated-draw", "draw")
        return None
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/distill/engine_selfplay_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/halluci_mate/distill/engine_selfplay.py tests/distill/engine_selfplay_test.py
git commit -m "feat(distill): add self-play adjudication state machine"
```

---

## Task 5: `play_seed` + `SelfPlayGame`

The one-game loop: replay the human seed prefix, then analyse → adjudicate → wobble → push until termination. Driven by an injected engine (a stub in tests; real Stockfish in production).

**Files:**
- Modify: `src/halluci_mate/distill/engine_selfplay.py`
- Test: `tests/distill/engine_selfplay_test.py`

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/distill/engine_selfplay_test.py
from halluci_mate.distill.engine_selfplay import SelfPlayGame, play_seed


class _ScriptedEngine:
    """Returns MultiPV analysis from a fixed white-relative eval per call.

    ``white_cp_fn(board)`` lets a test drive adjudication; the analysis always
    offers the current legal moves as separate PV lines so wobble has choices.
    """

    def __init__(self, white_cp_fn) -> None:
        self._white_cp_fn = white_cp_fn

    def analyse(self, board: chess.Board, limit: chess.engine.Limit, *, multipv: int) -> list[chess.engine.InfoDict]:
        del limit
        white_cp = self._white_cp_fn(board)
        infos: list[chess.engine.InfoDict] = []
        for move in list(board.legal_moves)[:multipv]:
            infos.append({"score": chess.engine.PovScore(chess.engine.Cp(white_cp), chess.WHITE), "pv": [move]})
        return infos


def test_play_seed_includes_seed_prefix_and_terminates_on_cap() -> None:
    engine = _ScriptedEngine(lambda board: 0)  # balanced -> no resign
    config = SelfPlayConfig(seed_plies=2, max_plies=6, draw_min_ply=999, multipv=3, wobble_cp=10_000)
    game = play_seed(engine, ("game-x", ["e2e4", "e7e5"]), config, random.Random(0))
    assert isinstance(game, SelfPlayGame)
    assert game.moves_uci[:2] == ["e2e4", "e7e5"]
    assert len(game.moves_uci) == 6
    assert game.termination == "max-plies"
    assert game.outcome == "draw"
    assert game.seed_source == "game-x"
    assert game.seed_plies == 2


def test_play_seed_adjudicates_white_win() -> None:
    engine = _ScriptedEngine(lambda board: 5000)  # white crushing every ply
    config = SelfPlayConfig(seed_plies=0, resign_cp=700, resign_plies=2, max_plies=50, multipv=2, wobble_cp=10_000)
    game = play_seed(engine, ("game-y", []), config, random.Random(0))
    assert game.termination == "adjudicated-win"
    assert game.outcome == "white"


def test_play_seed_reproducible_with_same_rng_seed() -> None:
    engine = _ScriptedEngine(lambda board: 0)
    config = SelfPlayConfig(seed_plies=0, max_plies=8, draw_min_ply=999, multipv=4, wobble_cp=10_000)
    g1 = play_seed(engine, ("g", []), config, random.Random(123))
    g2 = play_seed(engine, ("g", []), config, random.Random(123))
    assert g1.moves_uci == g2.moves_uci
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/distill/engine_selfplay_test.py -v`
Expected: FAIL — `play_seed` / `SelfPlayGame` not defined.

- [ ] **Step 3: Implement**

```python
# add to src/halluci_mate/distill/engine_selfplay.py
from typing import Protocol

# White-relative result strings -> outcome label; "*" (no result) -> draw.
_RESULT_TO_OUTCOME: dict[str, Outcome] = {"1-0": "white", "0-1": "black", "1/2-1/2": "draw"}


class _AnalysisEngine(Protocol):
    """The single ``chess.engine`` call ``play_seed`` makes (multipv analyse)."""

    def analyse(self, board: chess.Board, limit: chess.engine.Limit, *, multipv: int) -> list[chess.engine.InfoDict]: ...


@dataclass(frozen=True)
class SelfPlayGame:
    """One generated game: full move list (incl. seed prefix) + outcome."""

    seed_source: str
    seed_plies: int
    moves_uci: list[str]
    outcome: Outcome
    termination: Termination


# A SeedOpening is (seed_source, seed_moves_uci) — see distill.seeds.
SeedOpening = tuple[str, list[str]]


def play_seed(engine: _AnalysisEngine, seed: SeedOpening, config: SelfPlayConfig, rng: random.Random) -> SelfPlayGame:
    """Play one full-strength game from a human opening seed."""
    seed_source, seed_moves = seed
    prefix = seed_moves[: config.seed_plies]
    board = chess.Board()
    for uci in prefix:
        board.push(chess.Move.from_uci(uci))
    moves = list(prefix)

    adjudicator = _AdjudicationState()
    limit = chess.engine.Limit(depth=config.depth)
    termination, outcome = _play_loop(engine, board, moves, adjudicator, config, rng, limit)
    return SelfPlayGame(seed_source=seed_source, seed_plies=len(prefix), moves_uci=moves, outcome=outcome, termination=termination)


def _play_loop(
    engine: _AnalysisEngine,
    board: chess.Board,
    moves: list[str],
    adjudicator: _AdjudicationState,
    config: SelfPlayConfig,
    rng: random.Random,
    limit: chess.engine.Limit,
) -> tuple[Termination, Outcome]:
    """Mutate ``board``/``moves`` until termination; return (termination, outcome)."""
    while True:
        if board.is_game_over(claim_draw=True):
            return "natural", _RESULT_TO_OUTCOME.get(board.result(claim_draw=True), "draw")
        if board.ply() >= config.max_plies:
            return "max-plies", "draw"
        infos = engine.analyse(board, limit, multipv=config.multipv)
        verdict = adjudicator.update(_best_white_cp(infos), board.ply(), config)
        if verdict is not None:
            return verdict
        move = select_wobble_move(infos, board.turn, config, rng)
        board.push(move)
        moves.append(move.uci())
```

(`_play_loop` keeps `play_seed` under the 40-line cap.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/distill/engine_selfplay_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/halluci_mate/distill/engine_selfplay.py tests/distill/engine_selfplay_test.py
git commit -m "feat(distill): add play_seed one-game self-play loop"
```

---

## Task 6: Raw shard schema + `ShardWriter`

Durable artifact layer. `RawGameRow` validates on read; `ShardWriter` buffers games, writes atomic parquet shards, and tracks a resumable `games_written` cursor in `manifest.json`.

**Files:**
- Create: `src/halluci_mate/distill/raw_shards.py`
- Test: `tests/distill/raw_shards_test.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/distill/raw_shards_test.py
from __future__ import annotations

from pathlib import Path

from halluci_mate.distill.engine_selfplay import SelfPlayConfig, SelfPlayGame
from halluci_mate.distill.raw_shards import EngineMeta, RawGameRow, ShardWriter, read_raw_games

META = EngineMeta(depth=18, multipv=4, wobble_cp=30, sf_version="test", nnue="test.nnue")


def _game(source: str) -> SelfPlayGame:
    return SelfPlayGame(seed_source=source, seed_plies=1, moves_uci=["e2e4", "e7e5"], outcome="draw", termination="max-plies")


def test_row_roundtrips_through_model() -> None:
    row = RawGameRow(game_id="g0", seed_source="s", seed_plies=1, moves_uci=["e2e4"], outcome="draw", termination="natural", **META.as_columns())
    assert RawGameRow.model_validate(row.model_dump()) == row


def test_shardwriter_writes_full_and_partial_shards(tmp_path: Path) -> None:
    with ShardWriter(tmp_path, META, shard_size=2) as writer:
        for i in range(5):
            writer.add(f"g{i}", _game(f"s{i}"))
    shards = sorted(tmp_path.glob("shard_*.parquet"))
    assert len(shards) == 3  # 2 + 2 + 1
    games = list(read_raw_games(tmp_path))
    assert [g.game_id for g in games] == ["g0", "g1", "g2", "g3", "g4"]
    assert all(g.outcome == "draw" for g in games)


def test_shardwriter_resumes_from_manifest(tmp_path: Path) -> None:
    with ShardWriter(tmp_path, META, shard_size=2) as writer:
        writer.add("g0", _game("s0"))
        writer.add("g1", _game("s1"))
    # Reopen: cursor should report 2 already written, next shard index 1.
    writer2 = ShardWriter(tmp_path, META, shard_size=2)
    assert writer2.games_written == 2
    with writer2:
        writer2.add("g2", _game("s2"))
    assert sorted(p.name for p in tmp_path.glob("shard_*.parquet")) == ["shard_00000.parquet", "shard_00001.parquet"]
    assert [g.game_id for g in read_raw_games(tmp_path)] == ["g0", "g1", "g2"]


def test_shardwriter_no_partial_file_left_on_clean_exit(tmp_path: Path) -> None:
    with ShardWriter(tmp_path, META, shard_size=2) as writer:
        writer.add("g0", _game("s0"))
    assert list(tmp_path.glob("*.tmp")) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/distill/raw_shards_test.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/halluci_mate/distill/raw_shards.py
"""Durable raw-game shard layer for the distillation generator.

Games are written as atomic parquet shards plus a ``manifest.json`` holding a
resumable ``games_written`` cursor. Re-tokenizing/re-splitting reads these
shards and never re-runs Stockfish.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 — used at runtime
from typing import TYPE_CHECKING, Literal, Self

from datasets import Dataset, load_dataset
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from types import TracebackType
    from collections.abc import Iterator

    from halluci_mate.distill.engine_selfplay import SelfPlayGame

MANIFEST_FILENAME = "manifest.json"
_SHARD_GLOB = "shard_*.parquet"


@dataclass(frozen=True)
class EngineMeta:
    """Engine provenance recorded on every row (Stockfish output is build-dependent)."""

    depth: int
    multipv: int
    wobble_cp: int
    sf_version: str
    nnue: str

    def as_columns(self) -> dict[str, object]:
        return {"engine_depth": self.depth, "engine_multipv": self.multipv, "engine_wobble_cp": self.wobble_cp, "engine_sf_version": self.sf_version, "engine_nnue": self.nnue}


class RawGameRow(BaseModel):
    """One on-disk game row (flattened engine columns for robust parquet typing)."""

    model_config = ConfigDict(frozen=True)

    game_id: str
    seed_source: str
    seed_plies: int
    moves_uci: list[str]
    outcome: Literal["white", "black", "draw"]
    termination: Literal["natural", "adjudicated-win", "adjudicated-draw", "max-plies"]
    engine_depth: int
    engine_multipv: int
    engine_wobble_cp: int
    engine_sf_version: str
    engine_nnue: str


def game_to_row(game_id: str, game: SelfPlayGame, meta: EngineMeta) -> RawGameRow:
    return RawGameRow(
        game_id=game_id,
        seed_source=game.seed_source,
        seed_plies=game.seed_plies,
        moves_uci=game.moves_uci,
        outcome=game.outcome,
        termination=game.termination,
        **meta.as_columns(),
    )


def read_raw_games(raw_dir: Path) -> Iterator[RawGameRow]:
    """Yield validated rows across all shards in ``raw_dir`` (shard order)."""
    shards = sorted(str(p) for p in raw_dir.glob(_SHARD_GLOB))
    if not shards:
        return
    data = load_dataset("parquet", data_files=shards, split="train")
    for row in data:
        yield RawGameRow.model_validate(row)


class ShardWriter:
    """Buffer games and flush atomic parquet shards with a resumable manifest."""

    def __init__(self, raw_dir: Path, meta: EngineMeta, shard_size: int) -> None:
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._meta = meta
        self._shard_size = shard_size
        self._buffer: list[dict[str, object]] = []
        self._next_index = len(list(raw_dir.glob(_SHARD_GLOB)))
        self.games_written = self._read_cursor()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        # Flush remaining games only on a clean exit so a crash doesn't write a
        # short shard that would corrupt the games_written cursor on resume.
        if exc_type is None and self._buffer:
            self._flush()

    def add(self, game_id: str, game: SelfPlayGame) -> None:
        self._buffer.append(game_to_row(game_id, game, self._meta).model_dump())
        if len(self._buffer) >= self._shard_size:
            self._flush()

    def _flush(self) -> None:
        path = self.raw_dir / f"shard_{self._next_index:05d}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        Dataset.from_list(self._buffer).to_parquet(str(tmp))
        os.replace(tmp, path)  # atomic publish
        self._next_index += 1
        self.games_written += len(self._buffer)
        self._buffer.clear()
        self._write_cursor()

    def _read_cursor(self) -> int:
        manifest = self.raw_dir / MANIFEST_FILENAME
        if not manifest.exists():
            return 0
        return int(json.loads(manifest.read_text(encoding="utf-8"))["games_written"])

    def _write_cursor(self) -> None:
        manifest = self.raw_dir / MANIFEST_FILENAME
        tmp = manifest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"games_written": self.games_written}) + "\n", encoding="utf-8")
        os.replace(tmp, manifest)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/distill/raw_shards_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/halluci_mate/distill/raw_shards.py tests/distill/raw_shards_test.py
git commit -m "feat(distill): add atomic resumable raw game shard writer"
```

---

## Task 7: Seed stream (`iter_seed_openings`)

Filtered Lichess stream → `(seed_source, first-N-ply UCI)` seeds. Reuses `passes_highelo_filter` and `parse_movetext`; rejects non-rapid/classical, non-Normal, too-short, and unparseable games.

**Files:**
- Create: `src/halluci_mate/distill/seeds.py`
- Test: `tests/distill/seeds_test.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/distill/seeds_test.py
from __future__ import annotations

from halluci_mate.distill.seeds import iter_seed_openings, is_rapid_or_classical

_OK_GAME = {
    "Event": "Rated Rapid game",
    "Termination": "Normal",
    "Site": "https://lichess.org/abc",
    "WhiteElo": "2400", "BlackElo": "2410",
    "WhiteRatingDiff": "5", "BlackRatingDiff": "-5",
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/distill/seeds_test.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/halluci_mate/distill/seeds.py
"""Turn a filtered Lichess stream into self-play opening seeds.

Reuses the high-Elo filter and PGN parsing from ``data_preparation`` /
``pgn_to_uci``; keeps only Rapid + Classical Normal-termination games long
enough to supply ``seed_plies`` opening moves.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from halluci_mate.data_preparation import passes_highelo_filter
from halluci_mate.pgn_to_uci import parse_movetext

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

logger = logging.getLogger(__name__)

# Lichess ``Event`` substrings for the long time controls we seed from.
_RAPID_CLASSICAL = ("rapid", "classical")

SeedOpening = tuple[str, list[str]]


def is_rapid_or_classical(event: str) -> bool:
    """True if the Lichess ``Event`` names a rapid or classical game (not blitz/bullet)."""
    lowered = event.lower()
    return any(tc in lowered for tc in _RAPID_CLASSICAL)


def iter_seed_openings(stream: Iterable[dict], seed_plies: int, min_elo: int, max_rating_diff: int, max_elo_gap: int) -> Iterator[SeedOpening]:
    """Yield ``(seed_source, first seed_plies UCI moves)`` for qualifying games."""
    for sample in stream:
        if sample.get("Termination") != "Normal" or not is_rapid_or_classical(sample.get("Event", "")):
            continue
        if not passes_highelo_filter(sample, min_elo, max_rating_diff, max_elo_gap):
            continue
        try:
            moves = parse_movetext(sample["movetext"])
        except (ValueError, KeyError):
            continue
        if len(moves) < seed_plies:
            continue
        yield sample.get("Site", "unknown"), moves[:seed_plies]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/distill/seeds_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/halluci_mate/distill/seeds.py tests/distill/seeds_test.py
git commit -m "feat(distill): add high-Elo rapid/classical seed extraction"
```

---

## Task 8: `run_generation` driver

Serial, resumable orchestration: skip already-written seeds, play each through `play_seed` with a per-game seeded RNG (reproducible regardless of order), feed the `ShardWriter`.

**Files:**
- Create: `src/halluci_mate/distill/generate.py`
- Test: `tests/distill/generate_test.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/distill/generate_test.py
from __future__ import annotations

from pathlib import Path

import chess
import chess.engine

from halluci_mate.distill.engine_selfplay import SelfPlayConfig
from halluci_mate.distill.generate import run_generation
from halluci_mate.distill.raw_shards import EngineMeta, read_raw_games

META = EngineMeta(depth=4, multipv=2, wobble_cp=30, sf_version="test", nnue="t")


class _BalancedEngine:
    def analyse(self, board: chess.Board, limit: chess.engine.Limit, *, multipv: int) -> list[chess.engine.InfoDict]:
        del limit
        return [{"score": chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE), "pv": [m]} for m in list(board.legal_moves)[:multipv]]


def _seeds(n: int) -> list[tuple[str, list[str]]]:
    return [(f"s{i}", ["e2e4", "e7e5"]) for i in range(n)]


def test_run_generation_writes_all_games(tmp_path: Path) -> None:
    config = SelfPlayConfig(seed_plies=2, max_plies=4, draw_min_ply=999, multipv=2, wobble_cp=10_000)
    written = run_generation(seeds=_seeds(5), engine=_BalancedEngine(), config=config, meta=META, raw_dir=tmp_path, shard_size=2)
    assert written == 5
    games = list(read_raw_games(tmp_path))
    assert [g.game_id for g in games] == [f"game-{i:08d}" for i in range(5)]


def test_run_generation_resumes_without_duplicates(tmp_path: Path) -> None:
    config = SelfPlayConfig(seed_plies=2, max_plies=4, draw_min_ply=999, multipv=2, wobble_cp=10_000)
    run_generation(seeds=_seeds(2), engine=_BalancedEngine(), config=config, meta=META, raw_dir=tmp_path, shard_size=2)
    # Resume: pass the full seed list again; first 2 must be skipped via the cursor.
    written = run_generation(seeds=_seeds(5), engine=_BalancedEngine(), config=config, meta=META, raw_dir=tmp_path, shard_size=2)
    assert written == 3
    ids = [g.game_id for g in read_raw_games(tmp_path)]
    assert ids == [f"game-{i:08d}" for i in range(5)]
    assert len(ids) == len(set(ids))
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/distill/generate_test.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/halluci_mate/distill/generate.py
"""Serial, resumable driver: seeds -> play_seed -> raw shards.

Game ids are the global seed index (zero-padded) so a resumed run continues
the same numbering. The per-game RNG is seeded from the game id, making the
wobble reproducible independent of processing order.
"""

from __future__ import annotations

import itertools
import logging
import random
from typing import TYPE_CHECKING

from tqdm import tqdm

from halluci_mate.distill.engine_selfplay import play_seed
from halluci_mate.distill.raw_shards import ShardWriter

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from halluci_mate.distill.engine_selfplay import SelfPlayConfig, SeedOpening, _AnalysisEngine
    from halluci_mate.distill.raw_shards import EngineMeta

logger = logging.getLogger(__name__)


def run_generation(
    *,
    seeds: Iterable[SeedOpening],
    engine: _AnalysisEngine,
    config: SelfPlayConfig,
    meta: EngineMeta,
    raw_dir: Path,
    shard_size: int,
) -> int:
    """Play games for ``seeds`` into ``raw_dir``; return total games written.

    On resume, ``ShardWriter.games_written`` says how many leading seeds were
    already processed; those are skipped so the run continues deterministically.
    """
    with ShardWriter(raw_dir, meta, shard_size=shard_size) as writer:
        start = writer.games_written
        if start:
            logger.info("Resuming: skipping %d already-written games", start)
        for index, seed in enumerate(itertools.islice(seeds, start, None), start=start):
            game_id = f"game-{index:08d}"
            game = play_seed(engine, seed, config, random.Random(game_id))
            writer.add(game_id, game)
        produced = writer.games_written - start
    logger.info("Generation complete: %d new games (%d total)", produced, start + produced)
    return produced
```

(Note: `tqdm` is already a dependency; wrap the loop with `tqdm` once a total is known in the CLI. The driver returns the count of *new* games for this invocation.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/distill/generate_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/halluci_mate/distill/generate.py tests/distill/generate_test.py
git commit -m "feat(distill): add resumable serial generation driver"
```

---

## Task 9: Stage 1 CLI (`generate_distill_games.py`)

Wires the real Stockfish engine to the driver. Single-process by default; `--workers` opens one Stockfish per pool worker (engine objects are not picklable, so each worker creates its own in an initializer). The pool path is covered by the integration test (Task 12), not unit tests.

**Files:**
- Create: `scripts/generate_distill_games.py`

- [ ] **Step 1: Implement the CLI**

```python
# scripts/generate_distill_games.py
"""Stage 1: generate a Stockfish self-play distillation corpus (raw game shards).

Streams high-Elo Rapid+Classical Lichess openings as seeds, plays both sides
with full-strength Stockfish + a MultiPV wobble, and writes resumable raw
parquet shards. Re-run with the same args to resume an interrupted job.

Usage:
    uv run python scripts/generate_distill_games.py --num-games 100000 --raw-dir data/distill/_raw
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import chess.engine
import typer
from datasets import load_dataset

from halluci_mate.distill.engine_selfplay import SelfPlayConfig
from halluci_mate.distill.generate import run_generation
from halluci_mate.distill.raw_shards import EngineMeta
from halluci_mate.distill.seeds import iter_seed_openings
from halluci_mate.logging_setup import configure_script_logging

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/distill/_raw")
DEFAULT_NUM_GAMES = 1_000_000
DEFAULT_SHARD_SIZE = 10_000
DEFAULT_MIN_ELO = 2000
DEFAULT_MAX_RATING_DIFF = 30
DEFAULT_MAX_ELO_GAP = 200


def _engine_meta(engine: chess.engine.SimpleEngine, config: SelfPlayConfig) -> EngineMeta:
    ident = engine.id
    return EngineMeta(depth=config.depth, multipv=config.multipv, wobble_cp=config.wobble_cp, sf_version=ident.get("name", "unknown"), nnue=str(ident.get("NNUE", "unknown")))


def main(
    stockfish_path: Annotated[str, typer.Option(help="Path to the Stockfish binary")] = "stockfish",
    num_games: Annotated[int, typer.Option(help="Target number of games to generate")] = DEFAULT_NUM_GAMES,
    raw_dir: Annotated[Path, typer.Option(help="Output directory for raw game shards")] = DEFAULT_RAW_DIR,
    shard_size: Annotated[int, typer.Option(help="Games per shard file")] = DEFAULT_SHARD_SIZE,
    depth: Annotated[int, typer.Option(help="Stockfish search depth (cost dial)")] = 18,
    threads: Annotated[int, typer.Option(help="Stockfish Threads per engine")] = 1,
    hash_mb: Annotated[int, typer.Option(help="Stockfish Hash (MB) per engine")] = 256,
    min_elo: Annotated[int, typer.Option(help="Minimum Elo for both seed players")] = DEFAULT_MIN_ELO,
    max_rating_diff: Annotated[int, typer.Option(help="Max |RatingDiff| for either seed player")] = DEFAULT_MAX_RATING_DIFF,
    max_elo_gap: Annotated[int, typer.Option(help="Max |WhiteElo - BlackElo| for the seed")] = DEFAULT_MAX_ELO_GAP,
) -> None:
    """Generate the Stage 1 raw self-play corpus."""
    configure_script_logging(__name__)
    config = SelfPlayConfig(depth=depth)

    stream = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)
    seeds = iter_seed_openings(stream, config.seed_plies, min_elo, max_rating_diff, max_elo_gap)
    capped_seeds = (seed for seed, _ in zip(seeds, range(num_games), strict=False))

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        engine.configure({"Threads": threads, "Hash": hash_mb})
        meta = _engine_meta(engine, config)
        run_generation(seeds=capped_seeds, engine=engine, config=config, meta=meta, raw_dir=raw_dir, shard_size=shard_size)
    finally:
        engine.quit()


if __name__ == "__main__":
    typer.run(main)
```

- [ ] **Step 2: Smoke-check it imports and shows help**

Run: `uv run python scripts/generate_distill_games.py --help`
Expected: typer help text listing the options; exit 0. (No real generation yet.)

- [ ] **Step 3: Lint/type/format then commit**

```bash
uv run ruff check . && uv run ruff format . && uv run ty check
git add scripts/generate_distill_games.py
git commit -m "feat(distill): add Stage 1 generate-distill-games CLI"
```

> **Note on `--workers` (deferred):** parallelism is a later enhancement. When added, use `multiprocessing.Pool(workers, initializer=_open_engine)` where `_open_engine` stores a per-process `SimpleEngine` in a module global, and an `imap_unordered` worker returns `SelfPlayGame`s the main process feeds to `ShardWriter`. Resume after a parallel crash may re-play a few seeds; Task 10's `dedup_by_game_id` removes the resulting duplicate ids at tokenization time. Do not implement this until the serial path is validated end-to-end.

---

## Task 10: Stratification parameterization (core, `data_preparation.py`)

Parameterize the stratification columns so engine data (which has no `elo_bucket`) can split on `result | opening_family`. This is a **core-only** change — no `distill` import — preserving the `distill → core` dependency direction. The engine tokenization itself lands in Task 10b.

**Files:**
- Modify: `src/halluci_mate/data_preparation.py`
- Test: `tests/data_preparation_test.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/data_preparation_test.py
from pathlib import Path

from datasets import Dataset

from halluci_mate.data_preparation import build_stratified_splits


def test_build_stratified_splits_accepts_custom_columns(tmp_path: Path) -> None:
    shard_dir = tmp_path / "_shards"
    shard_dir.mkdir()
    rows = [{"input_ids": [1, 2], "attention_mask": [1, 1], "result": "white" if i % 2 else "draw", "opening_family": "e4"} for i in range(20)]
    Dataset.from_list(rows).to_parquet(str(shard_dir / "shard_00000.parquet"))
    train, eval_, test = build_stratified_splits(shard_dir, eval_size=4, test_size=2, seed=42, stratify_columns=("result", "opening_family"))
    assert len(train) + len(eval_) + len(test) == 20
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/data_preparation_test.py -v`
Expected: FAIL — `build_stratified_splits` has no `stratify_columns` parameter.

- [ ] **Step 3: Implement — parameterize stratification**

In `src/halluci_mate/data_preparation.py`, add a module constant near `_METADATA_COLUMNS`:

```python
# Default stratification key for human (Lichess) data; engine data drops
# ``elo_bucket`` since self-play continuations have no single rating.
_DEFAULT_STRATIFY_COLUMNS: tuple[str, ...] = ("elo_bucket", "result", "opening_family")
```

Change `build_stratified_splits` to accept and use the columns (replace the hardcoded key line):

```python
def build_stratified_splits(
    shard_dir: Path,
    eval_size: int,
    test_size: int,
    seed: int,
    stratify_columns: tuple[str, ...] = _DEFAULT_STRATIFY_COLUMNS,
) -> tuple[Dataset, Dataset, Dataset]:
    """Load all shards and split into stratified train/eval/test datasets."""
    shard_files = sorted(str(p) for p in shard_dir.glob("shard_*.parquet"))
    all_data = load_dataset("parquet", data_files=shard_files, split="train")

    stratum_indices: dict[str, list[int]] = defaultdict(list)
    for idx in range(len(all_data)):
        row = all_data[idx]
        key = "|".join(str(row[col]) for col in stratify_columns)
        stratum_indices[key].append(idx)
    # ... unchanged from here down ...
```

Thread the parameter through `save_splits`:

```python
def save_splits(
    shard_dir: Path,
    total_examples: int,
    output_dir: Path,
    stratify_columns: tuple[str, ...] = _DEFAULT_STRATIFY_COLUMNS,
) -> None:
    # ... unchanged until the build call ...
    train_data, eval_data, test_data = build_stratified_splits(shard_dir, eval_size, test_size, SHUFFLE_SEED, stratify_columns)
    # ... unchanged ...
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/data_preparation_test.py -v`
Expected: PASS (new test + existing ones still green).

- [ ] **Step 5: Commit**

```bash
git add src/halluci_mate/data_preparation.py tests/data_preparation_test.py
git commit -m "refactor(data): parameterize stratification columns for splits"
```

---

## Task 10b: Engine tokenization module (`distill/tokenize.py`)

`process_engine_game` (the engine analogue of `process_game`) and `tokenize_engine_shards`. Lives in `distill/` so it can import `read_raw_games` (from `distill/raw_shards.py`) and core helpers (`game_to_sequences`, `classify_opening_family`, `write_shard`, `SHARD_SIZE`) without inverting the dependency direction.

**Files:**
- Create: `src/halluci_mate/distill/tokenize.py`
- Test: `tests/distill/tokenize_test.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/distill/tokenize_test.py
from __future__ import annotations

from pathlib import Path

from datasets import Dataset

from halluci_mate.data_preparation import create_tokenizer
from halluci_mate.distill.tokenize import process_engine_game, tokenize_engine_shards


def _raw_row(game_id: str, outcome: str) -> dict:
    return {
        "game_id": game_id,
        "seed_source": "s",
        "seed_plies": 2,
        "moves_uci": ["e2e4", "e7e5", "g1f3"],
        "outcome": outcome,
        "termination": "natural",
        "engine_depth": 18, "engine_multipv": 4, "engine_wobble_cp": 30,
        "engine_sf_version": "t", "engine_nnue": "t",
    }


def test_process_engine_game_emits_winner_perspective_sequence() -> None:
    examples = process_engine_game(_raw_row("g0", "white"), create_tokenizer())
    assert len(examples) == 1  # decisive -> one perspective
    assert examples[0]["result"] == "white"
    assert examples[0]["opening_family"] == "e4"
    assert examples[0]["termination_type"] == "decisive"
    assert "elo_bucket" not in examples[0]
    assert len(examples[0]["input_ids"]) > 0


def test_process_engine_game_draw_emits_two_perspectives() -> None:
    assert len(process_engine_game(_raw_row("g1", "draw"), create_tokenizer())) == 2


def test_tokenize_engine_shards_roundtrips(tmp_path: Path) -> None:
    raw_dir = tmp_path / "_raw"
    raw_dir.mkdir()
    Dataset.from_list([_raw_row(f"g{i}", "draw") for i in range(3)]).to_parquet(str(raw_dir / "shard_00000.parquet"))
    out_shards = tmp_path / "_tok"
    out_shards.mkdir()
    total = tokenize_engine_shards(raw_dir, create_tokenizer(), out_shards)
    assert total == 6  # 3 draws x 2 perspectives


def test_tokenize_engine_shards_dedups_repeated_game_ids(tmp_path: Path) -> None:
    raw_dir = tmp_path / "_raw"
    raw_dir.mkdir()
    Dataset.from_list([_raw_row("dup", "white"), _raw_row("dup", "white")]).to_parquet(str(raw_dir / "shard_00000.parquet"))
    out_shards = tmp_path / "_tok"
    out_shards.mkdir()
    assert tokenize_engine_shards(raw_dir, create_tokenizer(), out_shards) == 1  # second "dup" skipped
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/distill/tokenize_test.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/halluci_mate/distill/tokenize.py
"""Stage 2 tokenization: raw self-play shards -> tokenized training shards.

Engine analogue of ``data_preparation.process_game`` / ``stream_and_shard``.
Imports core helpers (one-directional: distill -> core); emits no
``elo_bucket`` because self-play continuations have no single rating.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from halluci_mate.data_preparation import SHARD_SIZE, write_shard
from halluci_mate.distill.raw_shards import read_raw_games
from halluci_mate.game_metadata import classify_opening_family
from halluci_mate.game_to_sequences import game_to_sequences

if TYPE_CHECKING:
    from pathlib import Path

    from halluci_mate.chess_tokenizer import ChessTokenizer

logger = logging.getLogger(__name__)


def process_engine_game(row: dict, tokenizer: ChessTokenizer) -> list[dict]:
    """Tokenize one raw self-play game row into training examples.

    Moves are already UCI (no PGN parsing). Decisive games yield one
    winner-perspective sequence; draws yield two (one per perspective).
    """
    moves = row["moves_uci"]
    outcome = row["outcome"]
    opening_family = classify_opening_family(moves[0])
    termination_type = "draw" if outcome == "draw" else "decisive"

    results: list[dict] = []
    for seq in game_to_sequences(moves, outcome):
        encoded = tokenizer(seq, add_special_tokens=False)
        results.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "result": outcome,
                "opening_family": opening_family,
                "termination_type": termination_type,
            }
        )
    return results


def tokenize_engine_shards(raw_dir: Path, tokenizer: ChessTokenizer, shard_dir: Path, *, dedup_by_game_id: bool = True) -> int:
    """Tokenize all raw shards into tokenized shards; return the example count.

    ``dedup_by_game_id`` drops games whose id was already seen — defends against
    the handful of duplicate games a parallel-generation crash+resume can leave.
    """
    buffer: list[dict] = []
    shard_index = 0
    total = 0
    seen: set[str] = set()
    for game in read_raw_games(raw_dir):
        if dedup_by_game_id and game.game_id in seen:
            continue
        seen.add(game.game_id)
        buffer.extend(process_engine_game(game.model_dump(), tokenizer))
        if len(buffer) >= SHARD_SIZE:
            write_shard(buffer, shard_dir, shard_index)
            total += len(buffer)
            buffer.clear()
            shard_index += 1
    if buffer:
        write_shard(buffer, shard_dir, shard_index)
        total += len(buffer)
    logger.info("Tokenized %d examples from %s", total, raw_dir)
    return total
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/distill/tokenize_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/halluci_mate/distill/tokenize.py tests/distill/tokenize_test.py
git commit -m "feat(distill): add engine-game tokenization with game-id dedup"
```

---

## Task 11: Stage 2 CLI (`prepare_distill_data.py`)

Reads raw shards → tokenizes → stratified splits on `result | opening_family`.

**Files:**
- Create: `scripts/prepare_distill_data.py`

- [ ] **Step 1: Implement the CLI**

```python
# scripts/prepare_distill_data.py
"""Stage 2: tokenize raw Stockfish self-play shards into train/eval/test splits.

Reads the raw game shards produced by ``generate_distill_games.py``, tokenizes
them with the chess tokenizer, and writes stratified Parquet splits consumable
by ``scripts/train.py`` unchanged. Stratifies on result + opening family
(engine self-play has no single Elo, so ``elo_bucket`` is dropped).

Usage:
    uv run python scripts/prepare_distill_data.py --raw-dir data/distill/_raw --output-dir data/distill
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Annotated

import typer

from halluci_mate.data_preparation import create_tokenizer, save_splits
from halluci_mate.distill.tokenize import tokenize_engine_shards
from halluci_mate.logging_setup import configure_script_logging

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/distill/_raw")
DEFAULT_OUTPUT_DIR = Path("data/distill")
_ENGINE_STRATIFY_COLUMNS = ("result", "opening_family")


def prepare_dataset(raw_dir: Path, output_dir: Path) -> None:
    """Tokenize raw shards, then build + save stratified splits."""
    tokenizer = create_tokenizer()
    shard_dir = output_dir / "_shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True)

    total_examples = tokenize_engine_shards(raw_dir, tokenizer, shard_dir)
    save_splits(shard_dir, total_examples, output_dir, stratify_columns=_ENGINE_STRATIFY_COLUMNS)


def main(
    raw_dir: Annotated[Path, typer.Option(help="Directory of raw game shards from Stage 1")] = DEFAULT_RAW_DIR,
    output_dir: Annotated[Path, typer.Option(help="Output directory for train/eval/test Parquet")] = DEFAULT_OUTPUT_DIR,
) -> None:
    """Prepare tokenized distillation splits from raw self-play shards."""
    configure_script_logging(__name__)
    prepare_dataset(raw_dir=raw_dir, output_dir=output_dir)


if __name__ == "__main__":
    typer.run(main)
```

- [ ] **Step 2: Smoke-check help**

Run: `uv run python scripts/prepare_distill_data.py --help`
Expected: typer help text; exit 0.

- [ ] **Step 3: Lint/type/format then commit**

```bash
uv run ruff check . && uv run ruff format . && uv run ty check
git add scripts/prepare_distill_data.py
git commit -m "feat(distill): add Stage 2 prepare-distill-data CLI"
```

---

## Task 12: Integration test (real Stockfish, shallow)

End-to-end at tiny scale, exercising the real engine + both CLIs' core functions. Lives under `tests/integration/`, which `pyproject.toml` excludes from the default `pytest` run (`norecursedirs = ["integration"]`), so it only runs when invoked explicitly.

**Files:**
- Create: `tests/integration/distill_pipeline_test.py`

- [ ] **Step 1: Write the test**

```python
# tests/integration/distill_pipeline_test.py
"""End-to-end distillation pipeline test against a real Stockfish binary.

Skipped automatically when no ``stockfish`` binary is on PATH. Run explicitly:
    uv run pytest tests/integration/distill_pipeline_test.py -v
"""

from __future__ import annotations

import shutil
from pathlib import Path

import chess.engine
import pytest

from halluci_mate.data_preparation import create_tokenizer
from halluci_mate.distill.engine_selfplay import SelfPlayConfig
from halluci_mate.distill.generate import run_generation
from halluci_mate.distill.raw_shards import EngineMeta, read_raw_games
from halluci_mate.distill.tokenize import tokenize_engine_shards

_STOCKFISH = shutil.which("stockfish")
pytestmark = pytest.mark.skipif(_STOCKFISH is None, reason="stockfish binary not on PATH")

_SEEDS = [("g-ruy", ["e2e4", "e7e5", "g1f3", "b8c6"]), ("g-fr", ["e2e4", "e7e6", "d2d4", "d7d5"])]


def test_generate_then_tokenize(tmp_path: Path) -> None:
    config = SelfPlayConfig(depth=6, multipv=3, max_plies=24, draw_min_ply=10, draw_plies=4, seed_plies=4)
    raw_dir = tmp_path / "_raw"
    engine = chess.engine.SimpleEngine.popen_uci(_STOCKFISH)
    try:
        engine.configure({"Threads": 1, "Hash": 16})
        meta = EngineMeta(depth=config.depth, multipv=config.multipv, wobble_cp=config.wobble_cp, sf_version="sf", nnue="nnue")
        written = run_generation(seeds=_SEEDS, engine=engine, config=config, meta=meta, raw_dir=raw_dir, shard_size=10)
    finally:
        engine.quit()

    assert written == 2
    games = list(read_raw_games(raw_dir))
    assert len(games) == 2
    assert all(g.moves_uci[:4] == seed[1] for g, seed in zip(games, _SEEDS, strict=True))

    shard_dir = tmp_path / "_tok"
    shard_dir.mkdir()
    total = tokenize_engine_shards(raw_dir, create_tokenizer(), shard_dir)
    assert total >= 2
    assert sorted(shard_dir.glob("shard_*.parquet"))
```

- [ ] **Step 2: Run it (only if Stockfish is installed)**

Run: `uv run pytest tests/integration/distill_pipeline_test.py -v`
Expected: PASS, or SKIP if `stockfish` is not on PATH. (Install via `sudo apt install stockfish` to run for real.)

- [ ] **Step 3: Confirm default suite still excludes it**

Run: `uv run pytest -q`
Expected: the integration test does not appear (excluded by `norecursedirs`); all unit tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/distill_pipeline_test.py
git commit -m "test(distill): add real-Stockfish end-to-end integration test"
```

---

## Task 13: Final verification gate

Run the project's full pre-commit gate over the whole change set.

- [ ] **Step 1: Full suite**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest`
Expected: all green.

- [ ] **Step 2: Per CLAUDE.md workflow, before opening a PR**

Invoke, in order: `@agent code-simplifier`, `@agent verify-app`, `@agent build-validator`, `@agent code-architect` (must return APPROVE), then `/review-changes`, then `/commit-push-pr`.

---

## Self-Review Notes (author)

- **Spec coverage:** seed filter (Task 7), full-strength + MultiPV wobble (Tasks 2-3, 5), adjudication (Task 4), resumable shardable raw artifact (Tasks 6, 8), engine version recorded (Task 6 `EngineMeta`), `elo_bucket`-dropped stratification (Task 10), Phase-1 format reuse via `game_to_sequences` (Task 10b), error handling for bad seeds (Task 7) and atomic writes (Task 6), unit + integration tests (Tasks 1-12). Parallel `--workers` is explicitly deferred (Task 9 note) — the spec calls the generator *shardable/distributable* but does not require the pool in the first cut.
- **Dependency direction:** all `distill → core` edges. `data_preparation` (core) gets only the pure stratify-column parameter (Task 10); engine tokenization that needs `read_raw_games` lives in `distill/tokenize.py` (Task 10b), so no core module imports `distill`.
- **Type consistency:** `SeedOpening = tuple[str, list[str]]` defined in both `engine_selfplay` and `seeds` (identical alias; `seeds` is the public source). `Outcome`/`Termination` literals match between `engine_selfplay` and `RawGameRow`. `EngineMeta.as_columns()` keys match `RawGameRow` field names (`engine_depth`, ...). `run_generation` is keyword-only and matches its tests.
- **Deferred-by-design:** Stockfish `--workers` pool; any dedup beyond `game_id`. No placeholders remain.
