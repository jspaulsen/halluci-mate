# Inference-Time Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a depth-2 minimax search that picks a move from the LM's legal top-K, scored by a pluggable leaf evaluator (material-count first), packaged as a `Predictor` drop-in for the `vs_stockfish` harness.

**Architecture:** A new `src/halluci_mate/search/` package: `leaf.py` (the `LeafEvaluator` seam + `MaterialEvaluator`), `minimax.py` (`run_search` returning a rich `SearchResult`), `predictor.py` (`SearchPredictor`, implements `inference.Predictor`). Search reuses the existing `Predictor` protocol to read the policy's top-K, explores on copied boards (no KV-cache changes), and maps its choice onto the existing `MovePrediction` with no record/metric/DPO schema change. `predict` is added to the `Predictor` protocol so third-party callers can build against the abstraction.

**Tech Stack:** Python 3.12, `python-chess`, HuggingFace Transformers, pytest, typer (CLI), `uv` for all commands.

**Spec:** `docs/superpowers/specs/2026-05-23-inference-search-design.md`

---

## File Structure

- Create `src/halluci_mate/search/__init__.py` — empty (repo convention: no barrel imports).
- Create `src/halluci_mate/search/leaf.py` — `LeafEvaluator` protocol, `MaterialEvaluator`, `PIECE_VALUES`, `MATE_VALUE`.
- Create `src/halluci_mate/search/minimax.py` — `CandidateScore`, `SearchResult`, `run_search`.
- Create `src/halluci_mate/search/predictor.py` — `SearchPredictor`.
- Modify `src/halluci_mate/inference.py` — add `predict` to the `Predictor` protocol.
- Modify `scripts/eval.py` — `--search` / `--search-k` flags + wiring on `vs-stockfish`.
- Modify the existing eval test stubs (add a trivial `predict`): `tests/eval/evaluators/vs_stockfish_test.py`, `tests/eval/evaluators/legal_rate_test.py`, `tests/scripts/eval_test.py`.
- Create `tests/helpers/search_policy.py` — `ScriptedPolicy` test double (shared by minimax/predictor tests).
- Create `tests/search/__init__.py`, `tests/search/leaf_test.py`, `tests/search/minimax_test.py`, `tests/search/predictor_test.py`.

All commands run from the repo root with `uv run`.

---

## Task 1: Add `predict` to the `Predictor` protocol

**Files:**
- Modify: `src/halluci_mate/inference.py` (the `Predictor` protocol, ~lines 57–71)
- Modify: `tests/eval/evaluators/vs_stockfish_test.py` (`_StubEngine` + two local `_IllegalEngine` classes)
- Modify: `tests/eval/evaluators/legal_rate_test.py` (`_ConstantEngine`)
- Modify: `tests/scripts/eval_test.py` (`_StubEngine`)

- [ ] **Step 1: Add `predict` to the protocol (the failing change)**

In `src/halluci_mate/inference.py`, inside `class Predictor(Protocol)`, add a second method after `predict_with_metadata`:

```python
    def predict(self, game: Game, constrained: bool | None = None) -> chess.Move:
        """Return a legal move, or raise ``IllegalMoveError`` (unconstrained illegal sample)."""
        ...
```

(`chess` and `Game` are already imported in this module; `Game` under `TYPE_CHECKING`, which is fine with `from __future__ import annotations`.)

- [ ] **Step 2: Run the type checker to see it fail**

Run: `uv run ty check`
Expected: FAIL — the eval test stubs passed to `Predictor`-typed params (`_StubEngine`, `_ConstantEngine`, `_IllegalEngine`) now lack `predict`.

- [ ] **Step 3: Add a trivial `predict` to each existing stub**

`ChessInferenceEngine` already implements `predict`, so only the stubs need updating. Each stub gains a `predict` that returns a legal move (the harness never calls it on these stubs; it just has to satisfy the protocol). All four files already use `from __future__ import annotations`, so the `chess.Move` return annotation needs no new import.

In `tests/eval/evaluators/vs_stockfish_test.py`, add to `_StubEngine` (after `predict_with_metadata`):

```python
    def predict(self, game: Game, constrained: bool | None = None) -> chess.Move:
        del constrained
        return list(game.board.legal_moves)[0]
```

Add the same method to **both** local `_IllegalEngine` classes (in `test_terminates_on_illegal_move` and `test_analyze_on_illegal_move_only_fills_before_fields`):

```python
        def predict(self, game: Game, constrained: bool | None = None) -> chess.Move:
            del constrained
            return next(iter(game.board.legal_moves))
```

In `tests/eval/evaluators/legal_rate_test.py`, add to `_ConstantEngine`:

```python
    def predict(self, game: Game, constrained: bool | None = None) -> chess.Move:
        del constrained
        return next(iter(game.board.legal_moves))
```

In `tests/scripts/eval_test.py`, add to `_StubEngine`:

```python
    def predict(self, game: Game, constrained: bool | None = None) -> chess.Move:
        del constrained
        return next(iter(game.board.legal_moves))
```

- [ ] **Step 4: Run type check + the affected suites to verify green**

Run: `uv run ty check && uv run pytest tests/eval tests/scripts -q`
Expected: PASS (type check clean; existing tests unaffected — `predict` is unused by them).

- [ ] **Step 5: Commit**

```bash
git add src/halluci_mate/inference.py tests/eval/evaluators/vs_stockfish_test.py tests/eval/evaluators/legal_rate_test.py tests/scripts/eval_test.py
git commit -m "feat(inference): add predict to the Predictor protocol"
```

---

## Task 2: Material leaf evaluator

**Files:**
- Create: `src/halluci_mate/search/__init__.py` (empty)
- Create: `src/halluci_mate/search/leaf.py`
- Create: `tests/search/__init__.py` (empty)
- Test: `tests/search/leaf_test.py`

- [ ] **Step 1: Create the empty package init files**

Create `src/halluci_mate/search/__init__.py` with no content, and `tests/search/__init__.py` with no content (mirrors `eval/`).

- [ ] **Step 2: Write the failing tests**

Create `tests/search/leaf_test.py`:

```python
"""Tests for the material leaf evaluator."""

from __future__ import annotations

import chess

from halluci_mate.search.leaf import MATE_VALUE, MaterialEvaluator


def test_start_position_is_balanced() -> None:
    evaluator = MaterialEvaluator()
    board = chess.Board()
    assert evaluator.evaluate(board, pov=chess.WHITE) == 0.0
    assert evaluator.evaluate(board, pov=chess.BLACK) == 0.0


def test_material_advantage_is_pov_signed() -> None:
    evaluator = MaterialEvaluator()
    # Standard start position with the black queen removed: White is +9.
    board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert evaluator.evaluate(board, pov=chess.WHITE) == 9.0
    assert evaluator.evaluate(board, pov=chess.BLACK) == -9.0


def test_checkmate_is_decisive_for_the_mating_side() -> None:
    evaluator = MaterialEvaluator()
    board = chess.Board()
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:  # fool's mate; White is mated
        board.push_uci(uci)
    assert board.is_checkmate()
    assert board.turn == chess.WHITE
    assert evaluator.evaluate(board, pov=chess.WHITE) == -MATE_VALUE
    assert evaluator.evaluate(board, pov=chess.BLACK) == MATE_VALUE


def test_stalemate_is_zero() -> None:
    evaluator = MaterialEvaluator()
    board = chess.Board("k7/8/1Q6/8/8/8/8/7K b - - 0 1")
    assert board.is_stalemate()
    assert evaluator.evaluate(board, pov=chess.WHITE) == 0.0


def test_insufficient_material_is_zero() -> None:
    evaluator = MaterialEvaluator()
    board = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1")  # K vs K
    assert board.is_insufficient_material()
    assert evaluator.evaluate(board, pov=chess.WHITE) == 0.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/search/leaf_test.py -q`
Expected: FAIL with `ModuleNotFoundError: halluci_mate.search.leaf`.

- [ ] **Step 4: Implement `leaf.py`**

Create `src/halluci_mate/search/leaf.py`:

```python
"""Leaf evaluators for inference-time search.

A ``LeafEvaluator`` scores a board from a given side's point of view (higher is
better for that side). ``MaterialEvaluator`` is the v1 implementation:
piece-count with checkmate / draw terminal handling.
"""

from __future__ import annotations

from typing import Protocol

import chess

# Classic piece weights; the king carries no material value.
PIECE_VALUES: dict[chess.PieceType, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

# Finite sentinel for a decisive (checkmate) leaf: large enough to dominate any
# material swing, but a plain float so scores stay ordinary numbers.
MATE_VALUE = 1_000_000.0


class LeafEvaluator(Protocol):
    """Scores a leaf board from ``pov``'s perspective (higher = better for pov)."""

    def evaluate(self, board: chess.Board, *, pov: chess.Color) -> float: ...


class MaterialEvaluator:
    """Piece-count leaf eval with checkmate / draw terminal handling."""

    def evaluate(self, board: chess.Board, *, pov: chess.Color) -> float:
        if board.is_checkmate():
            # The side to move has been checkmated.
            return -MATE_VALUE if board.turn == pov else MATE_VALUE
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        white = _material(board, chess.WHITE)
        black = _material(board, chess.BLACK)
        score = white - black
        return score if pov == chess.WHITE else -score


def _material(board: chess.Board, color: chess.Color) -> float:
    return sum(value * len(board.pieces(piece_type, color)) for piece_type, value in PIECE_VALUES.items())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/search/leaf_test.py -q`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add src/halluci_mate/search/__init__.py src/halluci_mate/search/leaf.py tests/search/__init__.py tests/search/leaf_test.py
git commit -m "feat(search): add material leaf evaluator"
```

---

## Task 3: Depth-2 minimax search

**Files:**
- Create: `src/halluci_mate/search/minimax.py`
- Create: `tests/helpers/search_policy.py`
- Test: `tests/search/minimax_test.py`

- [ ] **Step 1: Create the scripted policy test double**

Create `tests/helpers/search_policy.py`:

```python
"""Scripted ``Predictor`` stub for search tests.

Returns a top-K keyed by board *placement* (``board_fen()``, ignoring clocks
and side-to-move so keys stay stable); falls back to the first legal moves in
board order. Implements the ``inference.Predictor`` protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import chess

from halluci_mate.eval.records import TopKEntry
from halluci_mate.inference import MovePrediction

if TYPE_CHECKING:
    from halluci_mate.game import Game


class ScriptedPolicy:
    def __init__(
        self,
        script: dict[str, list[tuple[str, float]]] | None = None,
        *,
        fallback_k: int = 3,
        mask_used: bool = True,
    ) -> None:
        self._script = script or {}
        self._fallback_k = fallback_k
        self._mask_used = mask_used

    def _entries(self, board: chess.Board, record_top_k: int) -> list[TopKEntry]:
        key = board.board_fen()
        if key in self._script:
            return [TopKEntry(move=uci, logprob=logprob) for uci, logprob in self._script[key]]
        moves = list(board.legal_moves)[: max(record_top_k, self._fallback_k)]
        return [TopKEntry(move=move.uci(), logprob=-float(i)) for i, move in enumerate(moves)]

    def predict_with_metadata(self, game: Game, *, constrained: bool | None = None, record_top_k: int = 5) -> MovePrediction:
        del constrained
        entries = self._entries(game.board, record_top_k)
        top_uci = entries[0].move
        return MovePrediction(
            played_move=chess.Move.from_uci(top_uci),
            model_move_uci=top_uci,
            raw_sample_move_uci=top_uci,
            raw_sample_legal=True,
            model_top_k=entries[:record_top_k] if record_top_k > 0 else [],
            mask_used=self._mask_used,
        )

    def predict(self, game: Game, constrained: bool | None = None) -> chess.Move:
        del constrained
        return next(iter(game.board.legal_moves))
```

- [ ] **Step 2: Write the failing tests**

Create `tests/search/minimax_test.py`:

```python
"""Tests for depth-2 minimax search over the policy's top-K."""

from __future__ import annotations

import chess
import pytest

from halluci_mate.game import Game, Perspective
from halluci_mate.search.leaf import MaterialEvaluator
from halluci_mate.search.minimax import run_search
from tests.helpers.search_policy import ScriptedPolicy

# Back-rank position: Ra1-a8 is mate; g1f1 is a quiet alternative.
_BACK_RANK_FEN = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
_BACK_RANK_KEY = "6k1/5ppp/8/8/8/8/5PPP/R5K1"

# Trap position: White Qxd5 wins a pawn but hangs the queen to exd5.
_TRAP_FEN = "6k1/5ppp/4p3/3p4/8/8/6PP/3Q2K1 w - - 0 1"
_TRAP_ROOT_KEY = "6k1/5ppp/4p3/3p4/8/8/6PP/3Q2K1"
_TRAP_AFTER_QXD5_KEY = "6k1/5ppp/4p3/3Q4/8/8/6PP/6K1"

# Quiet equal-material position for tie-break checks.
_QUIET_FEN = "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1"
_QUIET_KEY = "6k1/5ppp/8/8/8/8/5PPP/6K1"


def _white_game(fen: str) -> Game:
    return Game(board=chess.Board(fen), perspective=Perspective.WHITE)


def test_picks_mate_over_quiet_move_and_overrides_policy_argmax() -> None:
    # Sanity: a1a8 really is mate from this position.
    sanity = chess.Board(_BACK_RANK_FEN)
    sanity.push_uci("a1a8")
    assert sanity.is_checkmate()

    # Policy ranks the quiet move first; search must override to the mate.
    policy = ScriptedPolicy({_BACK_RANK_KEY: [("g1f1", -0.1), ("a1a8", -0.5)]})
    result = run_search(policy, MaterialEvaluator(), _white_game(_BACK_RANK_FEN), k=2)

    assert result.policy_argmax == chess.Move.from_uci("g1f1")
    assert result.chosen == chess.Move.from_uci("a1a8")


def test_avoids_hanging_a_piece_via_opponent_min() -> None:
    policy = ScriptedPolicy(
        {
            _TRAP_ROOT_KEY: [("d1d5", -0.1), ("d1d2", -0.5)],  # policy greedily prefers the capture
            _TRAP_AFTER_QXD5_KEY: [("e6d5", -0.1)],            # opponent recaptures the queen
        }
    )
    result = run_search(policy, MaterialEvaluator(), _white_game(_TRAP_FEN), k=2)

    assert result.policy_argmax == chess.Move.from_uci("d1d5")
    assert result.chosen == chess.Move.from_uci("d1d2")


def test_candidate_that_is_terminal_after_my_move_is_scored_without_a_reply() -> None:
    # Only the mate is offered; pushing it ends the game (no reply layer).
    policy = ScriptedPolicy({_BACK_RANK_KEY: [("a1a8", -0.1)]})
    result = run_search(policy, MaterialEvaluator(), _white_game(_BACK_RANK_FEN), k=1)
    assert result.chosen == chess.Move.from_uci("a1a8")


def test_tie_break_prefers_logprob_then_uci() -> None:
    leaf = MaterialEvaluator()

    # Equal score (both quiet, material 0) and equal logprob -> lower UCI wins.
    by_uci = ScriptedPolicy({_QUIET_KEY: [("g1h1", -0.5), ("g1f1", -0.5)]})
    assert run_search(by_uci, leaf, _white_game(_QUIET_FEN), k=2).chosen == chess.Move.from_uci("g1f1")

    # Equal score, different logprob -> higher logprob wins.
    by_logprob = ScriptedPolicy({_QUIET_KEY: [("g1f1", -0.9), ("g1h1", -0.1)]})
    assert run_search(by_logprob, leaf, _white_game(_QUIET_FEN), k=2).chosen == chess.Move.from_uci("g1h1")


def test_k_below_one_raises() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        run_search(ScriptedPolicy(), MaterialEvaluator(), _white_game(_QUIET_FEN), k=0)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/search/minimax_test.py -q`
Expected: FAIL with `ModuleNotFoundError: halluci_mate.search.minimax`.

- [ ] **Step 4: Implement `minimax.py`**

Create `src/halluci_mate/search/minimax.py`:

```python
"""Depth-2 minimax search over the policy's legal top-K.

For each of the policy's top-K candidate moves, ``run_search`` asks the policy
for the opponent's top-K replies, scores each resulting leaf with a
``LeafEvaluator``, takes the opponent's worst-for-us reply (min), and finally
the best candidate (argmax). Leaf scoring is board-only, so the cost is
``1 + k`` forwards. See
``docs/superpowers/specs/2026-05-23-inference-search-design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess

from halluci_mate.game import Game, Perspective

if TYPE_CHECKING:
    from halluci_mate.eval.records import TopKEntry
    from halluci_mate.inference import MovePrediction, Predictor
    from halluci_mate.search.leaf import LeafEvaluator


@dataclass(frozen=True)
class CandidateScore:
    """One root candidate move with its policy logprob and minimax score."""

    move: chess.Move
    policy_logprob: float
    score: float


@dataclass(frozen=True)
class SearchResult:
    """Full output of a search; the seam for future search->policy distillation."""

    candidates: list[CandidateScore]  # score-sorted, best first
    chosen: chess.Move
    policy_argmax: chess.Move
    root_prediction: MovePrediction


def run_search(policy: Predictor, leaf: LeafEvaluator, game: Game, *, k: int) -> SearchResult:
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    pov = chess.WHITE if game.perspective == Perspective.WHITE else chess.BLACK
    root = policy.predict_with_metadata(game, constrained=True, record_top_k=k)
    candidates = _legal_candidates(root.model_top_k, game.board)
    if not candidates:
        raise ValueError(f"policy returned no legal top-K candidates for {game.board.fen()}")
    scored = [_score_candidate(policy, leaf, game, move, logprob, pov, k) for move, logprob in candidates]
    scored.sort(key=lambda candidate: (-candidate.score, -candidate.policy_logprob, candidate.move.uci()))
    return SearchResult(candidates=scored, chosen=scored[0].move, policy_argmax=candidates[0][0], root_prediction=root)


def _score_candidate(policy: Predictor, leaf: LeafEvaluator, game: Game, move: chess.Move, logprob: float, pov: chess.Color, k: int) -> CandidateScore:
    board_after = game.board.copy(stack=True)  # keep move history so the branch tokenizes correctly
    board_after.push(move)
    if board_after.is_game_over():
        return CandidateScore(move=move, policy_logprob=logprob, score=leaf.evaluate(board_after, pov=pov))
    branch = Game(board=board_after, perspective=game.perspective)
    replies = policy.predict_with_metadata(branch, constrained=True, record_top_k=k)
    reply_moves = [reply for reply, _ in _legal_candidates(replies.model_top_k, board_after)]
    return CandidateScore(move=move, policy_logprob=logprob, score=_min_reply_score(board_after, reply_moves, leaf, pov))


def _min_reply_score(board_after: chess.Board, reply_moves: list[chess.Move], leaf: LeafEvaluator, pov: chess.Color) -> float:
    if not reply_moves:  # defensive: non-terminal board always has legal replies
        return leaf.evaluate(board_after, pov=pov)
    scores: list[float] = []
    for reply in reply_moves:
        leaf_board = board_after.copy(stack=False)  # material eval needs no history
        leaf_board.push(reply)
        scores.append(leaf.evaluate(leaf_board, pov=pov))
    return min(scores)  # opponent chooses the reply that minimises our score


def _legal_candidates(top_k: list[TopKEntry], board: chess.Board) -> list[tuple[chess.Move, float]]:
    """Parse top-K UCI entries to legal moves, dropping any that don't apply."""
    candidates: list[tuple[chess.Move, float]] = []
    for entry in top_k:
        move = _parse_legal(entry.move, board)
        if move is not None:
            candidates.append((move, entry.logprob))
    return candidates


def _parse_legal(uci: str, board: chess.Board) -> chess.Move | None:
    try:
        move = chess.Move.from_uci(uci)
    except chess.InvalidMoveError:
        return None
    return move if move in board.legal_moves else None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/search/minimax_test.py -q`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add src/halluci_mate/search/minimax.py tests/helpers/search_policy.py tests/search/minimax_test.py
git commit -m "feat(search): add depth-2 minimax over policy top-K"
```

---

## Task 4: SearchPredictor

**Files:**
- Create: `src/halluci_mate/search/predictor.py`
- Test: `tests/search/predictor_test.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/search/predictor_test.py`:

```python
"""Tests for SearchPredictor (the Predictor-protocol wrapper around search)."""

from __future__ import annotations

import chess
import pytest

from halluci_mate.game import Game, Perspective
from halluci_mate.search.leaf import MaterialEvaluator
from halluci_mate.search.predictor import SearchPredictor
from tests.helpers.search_policy import ScriptedPolicy

# Back-rank position: a1a8 is mate; policy ranks the quiet g1f1 first.
_BACK_RANK_FEN = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
_BACK_RANK_KEY = "6k1/5ppp/8/8/8/8/5PPP/R5K1"


def _white_game() -> Game:
    return Game(board=chess.Board(_BACK_RANK_FEN), perspective=Perspective.WHITE)


def test_prediction_replaces_move_but_preserves_policy_metadata() -> None:
    policy = ScriptedPolicy({_BACK_RANK_KEY: [("g1f1", -0.1), ("a1a8", -0.5)]}, mask_used=False)
    predictor = SearchPredictor(policy=policy, leaf=MaterialEvaluator(), k=2)

    prediction = predictor.predict_with_metadata(_white_game(), record_top_k=1)

    # model_move is the search pick (the mate), not the policy argmax.
    assert prediction.model_move_uci == "a1a8"
    assert prediction.played_move == chess.Move.from_uci("a1a8")
    # The rest passes through from the policy's own root prediction.
    assert prediction.raw_sample_move_uci == "g1f1"
    assert prediction.mask_used is False
    assert [entry.move for entry in prediction.model_top_k] == ["g1f1"]  # sliced to record_top_k=1


def test_predict_returns_the_chosen_move() -> None:
    policy = ScriptedPolicy({_BACK_RANK_KEY: [("g1f1", -0.1), ("a1a8", -0.5)]})
    predictor = SearchPredictor(policy=policy, leaf=MaterialEvaluator(), k=2)
    assert predictor.predict(_white_game()) == chess.Move.from_uci("a1a8")


def test_k_below_one_raises() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        SearchPredictor(policy=ScriptedPolicy(), leaf=MaterialEvaluator(), k=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/search/predictor_test.py -q`
Expected: FAIL with `ModuleNotFoundError: halluci_mate.search.predictor`.

- [ ] **Step 3: Implement `predictor.py`**

Create `src/halluci_mate/search/predictor.py`:

```python
"""SearchPredictor: wrap a policy in depth-2 search, exposing the Predictor API.

Implements ``inference.Predictor`` so it is a drop-in for ``ChessInferenceEngine``
in the eval harness and in third-party play loops.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from halluci_mate.search.minimax import run_search

if TYPE_CHECKING:
    import chess

    from halluci_mate.game import Game
    from halluci_mate.inference import MovePrediction, Predictor
    from halluci_mate.search.leaf import LeafEvaluator


class SearchPredictor:
    """Selects moves by depth-2 minimax over a wrapped policy's legal top-K."""

    def __init__(self, policy: Predictor, leaf: LeafEvaluator, *, k: int) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}")
        self.policy = policy
        self.leaf = leaf
        self.k = k

    def predict_with_metadata(self, game: Game, *, constrained: bool | None = None, record_top_k: int = 5) -> MovePrediction:
        # Search always reasons over the legal (masked) top-K; ``constrained`` is
        # accepted for Predictor parity and ignored.
        del constrained
        result = run_search(self.policy, self.leaf, game, k=self.k)
        root = result.root_prediction
        return replace(
            root,
            played_move=result.chosen,
            model_move_uci=result.chosen.uci(),
            model_top_k=root.model_top_k[:record_top_k],
        )

    def predict(self, game: Game, constrained: bool | None = None) -> chess.Move:
        # The chosen move always comes from the legal top-K, so this never raises.
        del constrained
        return run_search(self.policy, self.leaf, game, k=self.k).chosen
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/search/predictor_test.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/halluci_mate/search/predictor.py tests/search/predictor_test.py
git commit -m "feat(search): add SearchPredictor Predictor wrapper"
```

---

## Task 5: CLI wiring on `vs-stockfish`

**Files:**
- Modify: `scripts/eval.py` (imports near lines 32–45; `vs_stockfish_cmd` flags + body)
- Test: `tests/scripts/eval_test.py`

- [ ] **Step 1: Write the failing test**

This file invokes the CLI in-process via `eval_cli.main([...])` and patches the engines with `_patch_engines(monkeypatch, stockfish)`. `json`, `CONFIG_FILENAME`, `_StubStockfish`, `_patch_engines`, and `eval_cli` are all already imported/defined in the file — reuse them. Add this test:

```python
def test_vs_stockfish_search_records_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--search` wraps the engine and records the search parameters in config.json."""
    stockfish = _StubStockfish()
    _patch_engines(monkeypatch, stockfish)
    evals_dir = tmp_path / "evals"

    eval_cli.main(
        [
            "vs-stockfish",
            "--checkpoint",
            "stub-ckpt",
            "--games",
            "1",
            "--max-plies",
            "4",
            "--halluci-color",
            "white",
            "--evals-dir",
            str(evals_dir),
            "--search",
            "--search-k",
            "4",
        ]
    )

    run_dirs = [p for p in evals_dir.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    config = json.loads((run_dirs[0] / CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert config["search"] is True
    assert config["search_k"] == 4
    assert config["search_leaf"] == "material"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/eval_test.py::test_vs_stockfish_search_records_config -q`
Expected: FAIL — `--search` is an unknown option (typer exits non-zero).

- [ ] **Step 3: Add the imports**

In `scripts/eval.py`, change the inference import and add the search imports:

```python
from halluci_mate.inference import ChessInferenceEngine, Predictor
from halluci_mate.search.leaf import MaterialEvaluator
from halluci_mate.search.predictor import SearchPredictor
```

- [ ] **Step 4: Add the flags to `vs_stockfish_cmd`**

Add two parameters to `vs_stockfish_cmd` (alongside `blunder_threshold_cp`, keeping the trailing `) -> None:`):

```python
    search: Annotated[bool, typer.Option("--search/--no-search", help="Wrap the model in depth-2 minimax search over its top-K (material leaf eval).")] = False,
    search_k: Annotated[int, typer.Option(help="Number of top-K candidates search considers per move (default: 3). Only used with --search.")] = 3,
```

- [ ] **Step 5: Wire the predictor in the command body**

In `vs_stockfish_cmd`, after `engine = ChessInferenceEngine.from_checkpoint(...)`, build the predictor and extend `extra_config`, then pass the predictor to `run_vs_stockfish`:

```python
    extra_config: dict[str, object] = {"temperature": engine.temperature, "top_k": engine.top_k}

    predictor: Predictor = engine
    if search:
        predictor = SearchPredictor(policy=engine, leaf=MaterialEvaluator(), k=search_k)
        extra_config.update({"search": True, "search_k": search_k, "search_leaf": "material"})
```

Then change the existing `run_vs_stockfish(engine=engine, ...)` call to `run_vs_stockfish(engine=predictor, ...)`. (Leave the existing `extra_config` assignment line replaced by the block above so it is defined once.)

- [ ] **Step 6: Run the new test + the full eval/scripts suites**

Run: `uv run pytest tests/scripts/eval_test.py -q && uv run ty check`
Expected: PASS (new test green, type check clean).

- [ ] **Step 7: Commit**

```bash
git add scripts/eval.py tests/scripts/eval_test.py
git commit -m "feat(eval): add --search flag to vs-stockfish"
```

---

## Task 6: Smoke test through the harness

**Files:**
- Test: `tests/eval/evaluators/vs_stockfish_test.py` (add one test; reuses `_StubEngine` + `_StubStockfish`)

- [ ] **Step 1: Write the smoke test**

In `tests/eval/evaluators/vs_stockfish_test.py`, add an import at the top (with the other imports):

```python
from halluci_mate.search.leaf import MaterialEvaluator
from halluci_mate.search.predictor import SearchPredictor
```

Then add the test (it wraps the existing `_StubEngine` policy in search and drives a real run):

```python
def test_search_predictor_drives_a_run(tmp_path: Path) -> None:
    """SearchPredictor is a drop-in Predictor: a full run completes and writes records."""
    run_dir = tmp_path / "run"
    config = VsStockfishConfig(games=1, max_plies=6, halluci_color="white")
    predictor = SearchPredictor(policy=_StubEngine(), leaf=MaterialEvaluator(), k=3)

    outcomes = run_vs_stockfish(
        engine=predictor,
        stockfish=_StubStockfish(),
        config=config,
        run_dir=run_dir,
        run_id=DEFAULT_RUN_ID,
        checkpoint=DEFAULT_CHECKPOINT,
    )

    assert len(outcomes) == 1
    records = RunReader(run_dir).read_records()
    move_records = [r for r in records if isinstance(r, PerMoveRecord)]
    assert move_records  # at least one model decision was recorded
    # Every recorded model move is legal in the position it was played from.
    for record in move_records:
        board = chess.Board(record.fen_before)
        assert chess.Move.from_uci(record.model_move) in board.legal_moves
```

- [ ] **Step 2: Run the smoke test**

Run: `uv run pytest tests/eval/evaluators/vs_stockfish_test.py::test_search_predictor_drives_a_run -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/eval/evaluators/vs_stockfish_test.py
git commit -m "test(search): smoke-test SearchPredictor through vs_stockfish"
```

---

## Task 7: Full verification + workflow gates

**Files:** none (verification only)

- [ ] **Step 1: Run the full local gate**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest`
Expected: all PASS. Fix anything that fails before proceeding (the PostToolUse hook auto-formats on edit, so `ruff format --check` should already be clean).

- [ ] **Step 2: Run the project workflow gates (per CLAUDE.md)**

In order, do not skip:
- `/test-and-fix` — iterate pytest + ty + ruff until all pass.
- `@agent code-simplifier` — simplify the new search modules (dead code, duplication, verbose patterns).
- `@agent verify-app` — full suite PASS/FAIL report.
- `@agent build-validator` — install / imports / deps PASS/FAIL report.
- `@agent code-architect` — module-boundary + structural review → expect APPROVE (search depends only on `chess`, `game`, and the `inference.Predictor`/`MovePrediction` surface; no imports from `eval/`).

- [ ] **Step 3: Ship**

- `/review-changes` — review the diff for logic errors, edge cases, style.
- `/commit-push-pr` — open the PR. PR title: `feat(search): inference-time depth-2 search over LM top-K`. Body: what changed (the search package + `predict` on the protocol + `--search` flag), why (catch one-move tactical blunders without retraining), and how to test it (`uv run pytest tests/search`, plus a real A/B is the documented follow-on eval run).

---

## Notes for the implementer

- **Never load pretrained weights** for the model — not relevant here (search wraps an existing checkpoint), but the repo trains from scratch.
- **All Python via `uv run`** — never bare `python`/`pytest`.
- **Stage files by name** — never `git add -A` (the working tree has unrelated untracked `dpo/`, `scripts/serve_vllm.py`, `.zed/` that must stay out of these commits).
- `search/` must not import from `eval/`. The harness depends on search only through the `Predictor` protocol.
- The branch is `jpaulsendev/inference-search`, already created and stacked on the DPO branch with the spec commit.
