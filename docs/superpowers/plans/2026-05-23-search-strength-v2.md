# Inference-Time Search v2 (Net-Strength) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework inference-time search so it *raises* v2d's score_rate vs Stockfish instead of lowering it, by fixing the fake opponent model, the material-only horizon, and unconditional re-ranking diagnosed in the v1 eval.

**Architecture:** The LM proposes root candidate moves (its top-K); a thin, board-only search verifies them. The opponent reply layer goes full-width (all legal moves, no LM call below the root); a fixed-POV quiescence search resolves forcing captures/checks; the leaf adds a small king-safety term; and `SearchPredictor` applies a margin gate so it only overrides the policy argmax when search is confident by τ pawn-equivalents.

**Tech Stack:** Python 3.12, `python-chess`, `pytest`, `typer` (CLI), `uv` for all commands. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-23-search-strength-v2-design.md`

**Per the project workflow (CLAUDE.md):** all Python runs via `uv run`; the PostToolUse hook auto-formats/lints on every edit; commit specific files by name (never `git add -A`); conventional-commit subjects. End each commit message with the trailer:
`Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`

---

## File Structure

- `src/halluci_mate/search/minimax.py` — **rewrite** the opponent layer to full-width + LM-free; **add** `quiesce()` and the `quiescence`/`qdepth` knobs. Drop `_min_reply_score`. (Tasks 1–2)
- `src/halluci_mate/search/leaf.py` — **add** `MaterialKingSafetyEvaluator` + king-safety helpers and weight constants; keep `MaterialEvaluator`. (Task 4)
- `src/halluci_mate/search/predictor.py` — **add** the margin gate and `margin`/`quiescence`/`qdepth` params. (Task 3)
- `scripts/eval.py` — **add** `--search-leaf`, `--search-margin`, `--search-quiescence`, `--search-qdepth`; a `SEARCH_LEAVES` registry; record the knobs in `extra_config`. (Task 5)
- `tests/search/minimax_test.py`, `tests/search/leaf_test.py`, `tests/search/predictor_test.py` — updated/added tests alongside each task.
- `tests/scripts/eval_test.py` — registry unit test (it already does `import scripts.eval as eval_cli`). (Task 5)

`run_search` stays a pure scorer; the gate is a `SearchPredictor`-level selection policy so `SearchResult` is untouched (preserves the future distillation seam).

---

## Task 1: Full-width, LM-free opponent layer (static leaf)

Replace the policy-generated opponent replies with *all* legal replies, scored by the static leaf. The LM is now called only at the root. This is the core "real opponent model" fix; quiescence is added in Task 2.

**Files:**
- Modify: `src/halluci_mate/search/minimax.py`
- Test: `tests/search/minimax_test.py`

- [ ] **Step 1: Update the trap test to stop scripting the opponent reply**

The opponent layer no longer consults the policy, so the recapture is found full-width. Replace `test_avoids_hanging_a_piece_via_opponent_min` in `tests/search/minimax_test.py` with:

```python
def test_avoids_hanging_a_piece_via_opponent_min() -> None:
    # Opponent replies are now searched full-width (no scripted reply needed):
    # after Qxd5, ...exd5 recaptures the queen, so search keeps the quiet Qd2.
    policy = ScriptedPolicy({_TRAP_ROOT_KEY: [("d1d5", -0.1), ("d1d2", -0.5)]})
    result = run_search(policy, MaterialEvaluator(), _white_game(_TRAP_FEN), k=2)

    assert result.policy_argmax == chess.Move.from_uci("d1d5")
    assert result.chosen == chess.Move.from_uci("d1d2")
```

Also delete the now-unused `_TRAP_AFTER_QXD5_KEY` module constant.

- [ ] **Step 2: Run the search tests to see the suite still describes current behavior**

Run: `uv run pytest tests/search/minimax_test.py -v`
Expected: PASS (the edited test still passes against the *current* implementation — Qxd5 is scripted-recaptured today; we're about to change *how* that recapture is found).

- [ ] **Step 3: Rewrite `minimax.py` to search opponent replies full-width**

Replace the entire body of `src/halluci_mate/search/minimax.py` with:

```python
"""Depth-2 minimax: the LM proposes root candidates, board search verifies them.

The model's top-K supplies the root candidate moves; the opponent's reply is
searched full-width over *all* legal moves (the leaf is board-only, so the LM is
not queried below the root), and each candidate is scored by the opponent's
worst-for-us reply. See
``docs/superpowers/specs/2026-05-23-search-strength-v2-design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess

from halluci_mate.game import Perspective

if TYPE_CHECKING:
    from halluci_mate.eval.records import TopKEntry
    from halluci_mate.game import Game
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
    scored = [_score_candidate(leaf, game.board, move, logprob, pov) for move, logprob in candidates]
    scored.sort(key=lambda candidate: (-candidate.score, -candidate.policy_logprob, candidate.move.uci()))
    return SearchResult(candidates=scored, chosen=scored[0].move, policy_argmax=candidates[0][0], root_prediction=root)


def _score_candidate(leaf: LeafEvaluator, board_before: chess.Board, move: chess.Move, logprob: float, pov: chess.Color) -> CandidateScore:
    board_after = board_before.copy(stack=False)  # no move history needed below the root
    board_after.push(move)
    if board_after.is_game_over():
        return CandidateScore(move=move, policy_logprob=logprob, score=leaf.evaluate(board_after, pov=pov))
    # Opponent minimizes our score over ALL legal replies (full-width, board-only).
    score = min(_reply_value(board_after, reply, leaf, pov) for reply in board_after.legal_moves)
    return CandidateScore(move=move, policy_logprob=logprob, score=score)


def _reply_value(board_after: chess.Board, reply: chess.Move, leaf: LeafEvaluator, pov: chess.Color) -> float:
    child = board_after.copy(stack=False)
    child.push(reply)
    return leaf.evaluate(child, pov=pov)


def _legal_candidates(top_k: list[TopKEntry], board: chess.Board) -> list[tuple[chess.Move, float]]:
    """Parse top-K UCI entries to legal moves, dropping any that don't apply."""
    return [(move, entry.logprob) for entry in top_k if (move := _parse_legal(entry.move, board)) is not None]


def _parse_legal(uci: str, board: chess.Board) -> chess.Move | None:
    try:
        move = chess.Move.from_uci(uci)
    except chess.InvalidMoveError:
        return None
    return move if move in board.legal_moves else None
```

- [ ] **Step 4: Run the search tests to verify full-width behavior passes**

Run: `uv run pytest tests/search/minimax_test.py -v`
Expected: PASS — all five tests, now exercising the full-width opponent layer. (`test_avoids_hanging_a_piece_via_opponent_min` finds `...exd5` by enumeration; the mate/terminal/tie-break tests are unchanged in outcome.)

- [ ] **Step 5: Confirm the wider suite still passes**

Run: `uv run pytest tests/search tests/eval -q`
Expected: PASS (predictor tests still pass — `SearchPredictor` is unchanged this task).

- [ ] **Step 6: Lint, type-check, commit**

Run: `uv run ruff check . && uv run ty check`
Expected: clean.

```bash
git add src/halluci_mate/search/minimax.py tests/search/minimax_test.py
git commit -m "feat(search): search opponent replies full-width, LM-free

Drop the policy-generated reply layer; the leaf is board-only so the
opponent's worst reply is found by enumerating all legal moves. Fixes
the v1 'fake opponent' regression and cuts cost to one LM forward/move.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Quiescence (capture + check-evasion extension)

Resolve forcing tactics past the depth-2 horizon so search stops walking into recaptures and forced mates. Adds `quiesce()` and the `quiescence`/`qdepth` knobs (default on, depth 4).

**Files:**
- Modify: `src/halluci_mate/search/minimax.py`
- Test: `tests/search/minimax_test.py`

- [ ] **Step 1: Write the failing quiescence tests**

Add to `tests/search/minimax_test.py` (and add `MATE_VALUE` to the leaf import: `from halluci_mate.search.leaf import MATE_VALUE, MaterialEvaluator`, and `from halluci_mate.search.minimax import quiesce, run_search`):

```python
# Black to move; ...exd5 wins the hanging White queen. Static eval over-credits White.
_HANGING_Q_FEN = "6k1/8/4p3/3Q4/8/8/8/6K1 b - - 0 1"

# White to move and in check (Re1+). Only Rxe1, then ...Rxe1# — a forced mate two
# plies deep, reachable only through the check-evasion + recapture extension.
_FORCED_MATE_FEN = "4r1k1/8/8/8/8/8/5PPP/3Rr1K1 w - - 0 1"


def test_quiescence_resolves_a_hanging_capture() -> None:
    board = chess.Board(_HANGING_Q_FEN)
    leaf = MaterialEvaluator()
    assert leaf.evaluate(board, pov=chess.WHITE) == 8.0  # static stand-pat over-credits White
    assert quiesce(board, leaf, chess.WHITE, 4) == -1.0  # ...exd5 leaves White down


def test_quiescence_depth_zero_is_static() -> None:
    board = chess.Board(_HANGING_Q_FEN)
    assert quiesce(board, MaterialEvaluator(), chess.WHITE, 0) == 8.0


def test_quiescence_sees_forced_mate_through_check() -> None:
    board = chess.Board(_FORCED_MATE_FEN)
    assert quiesce(board, MaterialEvaluator(), chess.WHITE, 4) == -MATE_VALUE


def test_run_search_threads_quiescence_flag() -> None:
    policy = ScriptedPolicy({_TRAP_ROOT_KEY: [("d1d5", -0.1), ("d1d2", -0.5)]})
    for quiescence in (True, False):
        result = run_search(policy, MaterialEvaluator(), _white_game(_TRAP_FEN), k=2, quiescence=quiescence)
        assert result.chosen == chess.Move.from_uci("d1d2")


def test_negative_qdepth_raises() -> None:
    with pytest.raises(ValueError, match="qdepth must be >= 0"):
        run_search(ScriptedPolicy(), MaterialEvaluator(), _white_game(_QUIET_FEN), k=1, qdepth=-1)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/search/minimax_test.py -k "quiescence or threads or qdepth" -v`
Expected: FAIL — `ImportError: cannot import name 'quiesce'` (and `run_search` has no `quiescence`/`qdepth` kwargs).

- [ ] **Step 3: Add quiescence and thread the knobs through `run_search`**

In `src/halluci_mate/search/minimax.py`: add the constant `DEFAULT_QDEPTH = 4` below the imports, and replace `run_search`, `_score_candidate`, and `_reply_value` with:

```python
def run_search(policy: Predictor, leaf: LeafEvaluator, game: Game, *, k: int, quiescence: bool = True, qdepth: int = DEFAULT_QDEPTH) -> SearchResult:
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if qdepth < 0:
        raise ValueError(f"qdepth must be >= 0; got {qdepth}")
    pov = chess.WHITE if game.perspective == Perspective.WHITE else chess.BLACK
    root = policy.predict_with_metadata(game, constrained=True, record_top_k=k)
    candidates = _legal_candidates(root.model_top_k, game.board)
    if not candidates:
        raise ValueError(f"policy returned no legal top-K candidates for {game.board.fen()}")
    scored = [_score_candidate(leaf, game.board, move, logprob, pov, quiescence=quiescence, qdepth=qdepth) for move, logprob in candidates]
    scored.sort(key=lambda candidate: (-candidate.score, -candidate.policy_logprob, candidate.move.uci()))
    return SearchResult(candidates=scored, chosen=scored[0].move, policy_argmax=candidates[0][0], root_prediction=root)


def _score_candidate(leaf: LeafEvaluator, board_before: chess.Board, move: chess.Move, logprob: float, pov: chess.Color, *, quiescence: bool, qdepth: int) -> CandidateScore:
    board_after = board_before.copy(stack=False)  # no move history needed below the root
    board_after.push(move)
    if board_after.is_game_over():
        return CandidateScore(move=move, policy_logprob=logprob, score=leaf.evaluate(board_after, pov=pov))
    # Opponent minimizes our score over ALL legal replies (full-width, board-only).
    score = min(_reply_value(board_after, reply, leaf, pov, quiescence=quiescence, qdepth=qdepth) for reply in board_after.legal_moves)
    return CandidateScore(move=move, policy_logprob=logprob, score=score)


def _reply_value(board_after: chess.Board, reply: chess.Move, leaf: LeafEvaluator, pov: chess.Color, *, quiescence: bool, qdepth: int) -> float:
    child = board_after.copy(stack=False)
    child.push(reply)
    if quiescence:
        return quiesce(child, leaf, pov, qdepth)
    return leaf.evaluate(child, pov=pov)


def quiesce(board: chess.Board, leaf: LeafEvaluator, pov: chess.Color, qdepth: int) -> float:
    """Fixed-POV minimax that extends only forcing moves to a quiet leaf.

    The leaf is always scored from ``pov``, so a node maximizes when
    ``board.turn == pov`` and minimizes otherwise. Captures are always
    extended; when the side to move is in check, all legal moves (evasions)
    are extended. ``qdepth`` caps the recursion for guaranteed termination.
    """
    legal = list(board.legal_moves)
    if not legal:
        return leaf.evaluate(board, pov=pov)  # checkmate or stalemate
    stand_pat = leaf.evaluate(board, pov=pov)
    if qdepth == 0:
        return stand_pat
    in_check = board.is_check()
    if in_check:
        forcing = legal
    else:
        forcing = [move for move in legal if board.is_capture(move)]
        if not forcing:
            return stand_pat  # quiet position
    maximizing = board.turn == pov
    best = (float("-inf") if maximizing else float("inf")) if in_check else stand_pat
    for move in forcing:
        child = board.copy(stack=False)
        child.push(move)
        value = quiesce(child, leaf, pov, qdepth - 1)
        best = max(best, value) if maximizing else min(best, value)
    return best
```

- [ ] **Step 4: Run the quiescence tests to verify they pass**

Run: `uv run pytest tests/search/minimax_test.py -v`
Expected: PASS — all tests, including the three `quiesce` tests, the threading test, and the negative-qdepth guard.

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check . && uv run ty check`
Expected: clean.

```bash
git add src/halluci_mate/search/minimax.py tests/search/minimax_test.py
git commit -m "feat(search): add fixed-POV quiescence over forcing moves

Extend captures and check-evasions to a quiet leaf so search resolves
recaptures and forced mates past the depth-2 horizon. Default on,
qdepth 4. Rejects the v1 mate-in-2 walk-in.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Margin gate in `SearchPredictor`

Only override the policy argmax when search's score beats it by τ pawn-equivalents. τ→∞ reproduces the baseline; τ=0 always trusts search.

**Files:**
- Modify: `src/halluci_mate/search/predictor.py`
- Test: `tests/search/predictor_test.py`

- [ ] **Step 1: Write the failing gate tests**

In `tests/search/predictor_test.py`, change `_white_game` to take a FEN and add the gate tests. Replace the helper and add constants/tests:

```python
def _white_game(fen: str = _BACK_RANK_FEN) -> Game:
    return Game(board=chess.Board(fen), perspective=Perspective.WHITE)


# Equal-material position; Qxd5 wins a truly free pawn (no recapture) -> +1 edge
# over the quiet Qd1. Lets us probe the margin gate with a small, known edge.
_FREE_PAWN_FEN = "6k1/6pp/8/3p4/8/8/3Q2PP/6K1 w - - 0 1"
_FREE_PAWN_KEY = "6k1/6pp/8/3p4/8/8/3Q2PP/6K1"


def test_gate_overrides_when_edge_clears_margin() -> None:
    policy = ScriptedPolicy({_FREE_PAWN_KEY: [("d2d1", -0.1), ("d2d5", -0.5)]})
    predictor = SearchPredictor(policy=policy, leaf=MaterialEvaluator(), k=2, margin=0.5)
    assert predictor.predict(_white_game(_FREE_PAWN_FEN)) == chess.Move.from_uci("d2d5")


def test_gate_keeps_argmax_when_edge_below_margin() -> None:
    policy = ScriptedPolicy({_FREE_PAWN_KEY: [("d2d1", -0.1), ("d2d5", -0.5)]})
    predictor = SearchPredictor(policy=policy, leaf=MaterialEvaluator(), k=2, margin=2.0)
    assert predictor.predict(_white_game(_FREE_PAWN_FEN)) == chess.Move.from_uci("d2d1")


def test_negative_margin_raises() -> None:
    with pytest.raises(ValueError, match="margin must be >= 0"):
        SearchPredictor(policy=ScriptedPolicy(), leaf=MaterialEvaluator(), k=2, margin=-1.0)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/search/predictor_test.py -k "gate or margin" -v`
Expected: FAIL — `SearchPredictor.__init__` has no `margin` parameter.

- [ ] **Step 3: Add the gate to `SearchPredictor`**

Replace the body of `src/halluci_mate/search/predictor.py` with:

```python
"""SearchPredictor: wrap a policy in depth-2 search, exposing the Predictor API.

Implements ``inference.Predictor`` so it is a drop-in for ``ChessInferenceEngine``
in the eval harness and in third-party play loops. A margin gate decides the
played move: search overrides the policy argmax only when its score beats the
argmax by ``margin`` pawn-equivalents (``margin`` large -> always keep argmax,
i.e. the bare policy; ``margin=0`` -> always trust search).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from halluci_mate.search.minimax import DEFAULT_QDEPTH, run_search

if TYPE_CHECKING:
    import chess

    from halluci_mate.game import Game
    from halluci_mate.inference import MovePrediction, Predictor
    from halluci_mate.search.leaf import LeafEvaluator
    from halluci_mate.search.minimax import SearchResult


class SearchPredictor:
    """Selects moves by depth-2 minimax over a wrapped policy's legal top-K."""

    def __init__(self, policy: Predictor, leaf: LeafEvaluator, *, k: int, margin: float = 0.0, quiescence: bool = True, qdepth: int = DEFAULT_QDEPTH) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}")
        if margin < 0:
            raise ValueError(f"margin must be >= 0; got {margin}")
        if qdepth < 0:
            raise ValueError(f"qdepth must be >= 0; got {qdepth}")
        self.policy = policy
        self.leaf = leaf
        self.k = k
        self.margin = margin
        self.quiescence = quiescence
        self.qdepth = qdepth

    def predict_with_metadata(self, game: Game, *, constrained: bool | None = None, record_top_k: int = 5) -> MovePrediction:
        # Search always reasons over the legal (masked) top-K; ``constrained`` is
        # accepted for Predictor parity and ignored.
        del constrained
        result = self._search(game)
        played = self._gated_move(result)
        root = result.root_prediction
        return replace(
            root,
            played_move=played,
            model_move_uci=played.uci(),
            model_top_k=root.model_top_k[:record_top_k],
        )

    def predict(self, game: Game, constrained: bool | None = None) -> chess.Move:
        # The chosen move always comes from the legal top-K, so this never raises.
        del constrained
        return self._gated_move(self._search(game))

    def _search(self, game: Game) -> SearchResult:
        return run_search(self.policy, self.leaf, game, k=self.k, quiescence=self.quiescence, qdepth=self.qdepth)

    def _gated_move(self, result: SearchResult) -> chess.Move:
        best = result.candidates[0]  # highest minimax score (score-sorted)
        argmax = next(candidate for candidate in result.candidates if candidate.move == result.policy_argmax)
        return best.move if (best.score - argmax.score) >= self.margin else argmax.move
```

- [ ] **Step 4: Run the predictor tests to verify they pass**

Run: `uv run pytest tests/search/predictor_test.py -v`
Expected: PASS — gate tests plus the existing metadata/predict/k-guard tests (default `margin=0.0` always plays the search best, so the back-rank-mate tests are unchanged).

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check . && uv run ty check`
Expected: clean.

```bash
git add src/halluci_mate/search/predictor.py tests/search/predictor_test.py
git commit -m "feat(search): margin-gate the override of the policy argmax

SearchPredictor keeps the policy argmax unless search beats it by
margin pawn-equivalents. margin->inf is the bare policy (cannot regress
below baseline); margin=0 always trusts search.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: King-safety leaf evaluator

Add `MaterialKingSafetyEvaluator`: material plus a small pawn-shield / open-file term, kept well below a minor piece so material dominates.

**Files:**
- Modify: `src/halluci_mate/search/leaf.py`
- Test: `tests/search/leaf_test.py`

- [ ] **Step 1: Write the failing king-safety tests**

Add to `tests/search/leaf_test.py` (extend the import to `from halluci_mate.search.leaf import MATE_VALUE, MaterialEvaluator, MaterialKingSafetyEvaluator` and add `import pytest`):

```python
# Equal material (3 pawns each). White's king is fully shielded (f2,g2,h2);
# Black's pawns are on the queenside (a7,b7,c7), leaving f/g/h open by its king.
_KING_SAFETY_FEN = "6k1/ppp5/8/8/8/8/5PPP/6K1 w - - 0 1"


def test_king_safety_rewards_shield_and_penalizes_open_files() -> None:
    board = chess.Board(_KING_SAFETY_FEN)
    assert MaterialEvaluator().evaluate(board, pov=chess.WHITE) == 0.0  # material is equal
    evaluator = MaterialKingSafetyEvaluator()
    white = evaluator.evaluate(board, pov=chess.WHITE)
    assert white > 0.0  # White's safer king scores higher
    assert evaluator.evaluate(board, pov=chess.BLACK) == pytest.approx(-white)  # POV-antisymmetric


def test_king_safety_eval_is_symmetric_at_the_start() -> None:
    assert MaterialKingSafetyEvaluator().evaluate(chess.Board(), pov=chess.WHITE) == 0.0


def test_king_safety_eval_is_decisive_on_checkmate() -> None:
    board = chess.Board()
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:  # fool's mate; White is mated
        board.push_uci(uci)
    assert MaterialKingSafetyEvaluator().evaluate(board, pov=chess.WHITE) == -MATE_VALUE
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/search/leaf_test.py -k king_safety -v`
Expected: FAIL — `ImportError: cannot import name 'MaterialKingSafetyEvaluator'`.

- [ ] **Step 3: Implement `MaterialKingSafetyEvaluator` and helpers**

Append to `src/halluci_mate/search/leaf.py` (after `MaterialEvaluator`, before or after `_material`):

```python
# King-safety weights, in pawn-equivalents. Kept well below a minor piece (3.0)
# so material always dominates; these only color otherwise-comparable lines.
W_KING_SHIELD = 0.15  # bonus per friendly pawn shielding the king
W_KING_OPEN_FILE = 0.25  # penalty per open/semi-open file on or next to the king


class MaterialKingSafetyEvaluator:
    """Material plus a small king-safety term (pawn shield + open files)."""

    def __init__(self) -> None:
        self._material = MaterialEvaluator()

    def evaluate(self, board: chess.Board, *, pov: chess.Color) -> float:
        if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
            return self._material.evaluate(board, pov=pov)  # terminal: material handles mate/draw
        base = self._material.evaluate(board, pov=pov)
        safety = _king_safety(board, chess.WHITE) - _king_safety(board, chess.BLACK)
        return base + (safety if pov == chess.WHITE else -safety)


def _king_safety(board: chess.Board, color: chess.Color) -> float:
    return W_KING_SHIELD * _king_shield(board, color) - W_KING_OPEN_FILE * _king_open_files(board, color)


def _king_shield(board: chess.Board, color: chess.Color) -> int:
    """Count friendly pawns on the three files around the king, on the two ranks in front of it."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    forward = 1 if color == chess.WHITE else -1
    pawn = chess.Piece(chess.PAWN, color)
    shield = 0
    for df in (-1, 0, 1):
        file = king_file + df
        if not 0 <= file <= 7:
            continue
        for dr in (1, 2):
            rank = king_rank + forward * dr
            if 0 <= rank <= 7 and board.piece_at(chess.square(file, rank)) == pawn:
                shield += 1
    return shield


def _king_open_files(board: chess.Board, color: chess.Color) -> int:
    """Count files on/next to the king that hold no friendly pawn (semi-open or open)."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    king_file = chess.square_file(king_sq)
    pawn = chess.Piece(chess.PAWN, color)
    exposed = 0
    for df in (-1, 0, 1):
        file = king_file + df
        if not 0 <= file <= 7:
            continue
        if not any(board.piece_at(chess.square(file, rank)) == pawn for rank in range(8)):
            exposed += 1
    return exposed
```

- [ ] **Step 4: Run the leaf tests to verify they pass**

Run: `uv run pytest tests/search/leaf_test.py -v`
Expected: PASS — king-safety tests plus the existing `MaterialEvaluator` tests.

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check . && uv run ty check`
Expected: clean.

```bash
git add src/halluci_mate/search/leaf.py tests/search/leaf_test.py
git commit -m "feat(search): add material + king-safety leaf evaluator

Small pawn-shield / open-file term (sub-minor-piece weight) for quiet
positions; delegates terminal mate/draw scoring to the material eval.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: CLI wiring

Expose the new knobs on `vs-stockfish` and record them in `config.json`.

**Files:**
- Modify: `scripts/eval.py`
- Test: `tests/scripts/eval_test.py`

- [ ] **Step 1: Write the failing registry test**

Add to `tests/scripts/eval_test.py` (it already does `import scripts.eval as eval_cli`; add `from halluci_mate.search.leaf import MaterialEvaluator, MaterialKingSafetyEvaluator`):

```python
def test_search_leaves_registry_maps_names_to_classes() -> None:
    assert eval_cli.SEARCH_LEAVES["material"] is MaterialEvaluator
    assert eval_cli.SEARCH_LEAVES["material-king-safety"] is MaterialKingSafetyEvaluator
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/scripts/eval_test.py -k registry -v`
Expected: FAIL — `AttributeError: module 'scripts.eval' has no attribute 'SEARCH_LEAVES'`.

- [ ] **Step 3: Wire the knobs into `scripts/eval.py`**

In `scripts/eval.py`:

(a) Replace the existing leaf import (`from halluci_mate.search.leaf import MaterialEvaluator`) with:
```python
from halluci_mate.search.leaf import MaterialEvaluator, MaterialKingSafetyEvaluator
```

(b) Add the registry next to `DEFAULT_EVALS_DIR`:
```python
# Name -> leaf evaluator class for the --search-leaf flag. Keys match the CLI
# choices; values are zero-arg constructors.
SEARCH_LEAVES = {"material": MaterialEvaluator, "material-king-safety": MaterialKingSafetyEvaluator}
```

(c) In `vs_stockfish_cmd`, replace the two existing search parameters with these four (keep `--search` and `--search-k`):
```python
    search: Annotated[bool, typer.Option("--search/--no-search", help="Wrap the model in depth-2 minimax search over its top-K.")] = False,
    search_k: Annotated[int, typer.Option(help="Number of top-K candidates search considers per move (default: 3). Only used with --search.")] = 3,
    search_leaf: Annotated[str, typer.Option(help=f"Leaf evaluator for search: one of {sorted(SEARCH_LEAVES)} (default: material-king-safety).")] = "material-king-safety",
    search_margin: Annotated[float, typer.Option(help="Override the policy argmax only when search beats it by this many pawn-equivalents (default: 1.0). 0 = always trust search.")] = 1.0,
    search_quiescence: Annotated[bool, typer.Option("--search-quiescence/--no-search-quiescence", help="Extend captures/checks to a quiet leaf (default: on).")] = True,
    search_qdepth: Annotated[int, typer.Option(help="Quiescence depth cap (default: 4). Only used with --search.")] = 4,
```

(d) Replace the existing `predictor: Predictor = engine` line **and** the `if search:` block below it (currently `eval.py:129-132`) with the following (one `predictor` declaration only — do not leave the old one):
```python
    predictor: Predictor = engine
    if search:
        if search_leaf not in SEARCH_LEAVES:
            raise typer.BadParameter(f"--search-leaf must be one of {sorted(SEARCH_LEAVES)}; got {search_leaf!r}")
        predictor = SearchPredictor(
            policy=engine,
            leaf=SEARCH_LEAVES[search_leaf](),
            k=search_k,
            margin=search_margin,
            quiescence=search_quiescence,
            qdepth=search_qdepth,
        )
        extra_config.update(
            {
                "search": True,
                "search_k": search_k,
                "search_leaf": search_leaf,
                "search_margin": search_margin,
                "search_quiescence": search_quiescence,
                "search_qdepth": search_qdepth,
            }
        )
```

- [ ] **Step 4: Run the registry test to verify it passes**

Run: `uv run pytest tests/scripts/eval_test.py -k registry -v`
Expected: PASS.

- [ ] **Step 5: Verify the CLI parses and search runs end-to-end (2-game smoke)**

Run:
```bash
uv run python scripts/eval.py vs-stockfish \
  --checkpoint jspaulsen/halluci-mate-v2d --checkpoint-tag v2d-smoke \
  --evals-dir /tmp/eval-smoke-v2 --games 2 \
  --stockfish-skill 5 --stockfish-depth 12 --sf-analyze \
  --halluci-color alternate --temperature 0.0 --device cuda:1 \
  --search --search-k 3 --search-leaf material-king-safety --search-margin 1.0
```
Expected: prints a `=== Summary ===` with 2 games played and an `Artifacts:` path; `config.json` in that run dir contains the six `search*` keys. (GPU 1 is the free device; GPU 0 may be occupied.)

- [ ] **Step 6: Lint, type-check, full suite, commit**

Run: `uv run ruff check . && uv run ty check && uv run pytest -q`
Expected: clean / all pass.

```bash
git add scripts/eval.py tests/scripts/eval_test.py
git commit -m "feat(eval): expose search leaf/margin/quiescence knobs on vs-stockfish

Add --search-leaf, --search-margin, --search-quiescence, --search-qdepth;
record all of them in config.json so runs are self-describing.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Verification & ship (project workflow)

Run the CLAUDE.md pre-commit workflow over the whole change set before opening a PR.

- [ ] **Step 1: `/test-and-fix`** — iterate pytest + ty + ruff until all pass.
- [ ] **Step 2: `@agent code-simplifier`** — dead code, duplication, verbose patterns (verifies tests + lint still pass).
- [ ] **Step 3: `@agent verify-app`** — full suite PASS/FAIL report (read-only).
- [ ] **Step 4: `@agent build-validator`** — install/imports/deps PASS/FAIL report.
- [ ] **Step 5: `@agent code-architect`** — module-boundary / structural review → must return APPROVE.
- [ ] **Step 6: `/review-changes`** — diff review for logic errors, edge cases, style.
- [ ] **Step 7:** Update `docs/inference_search.md` and the spec's "Status" line if anything diverged during implementation; commit any doc fixes.
- [ ] **Step 8: `/commit-push-pr`** — open the PR (only after verify-app + build-validator pass and code-architect APPROVEs). PR description: what changed (search v2), why (v1 regression + root cause), how to test (the measurement runbook below).

---

## Measurement & tuning runbook (operational — not code; run after Task 6 or alongside)

The implementation ships sensible defaults; these runs decide the final τ and whether king-safety earns its place. Compare every run against the v2d baseline `evals/2026-05-18T20-34-08_v2b-dpo-broad-sharp-welcoming-skink-500_vs-stockfish` (score_rate 0.090). **Ship bar: score_rate > 0.090.** Use GPU 1 (`--device cuda:1`).

Per the spec's increment order, measure at 30 games first, then confirm winners at 100:

1. **Core (full-width + quiescence, material leaf), gate off:**
   `--search --search-k 3 --search-leaf material --search-margin 0 --games 30`
2. **τ sweep** (material leaf): rerun with `--search-margin` in {0.5, 1.0, 2.0}; pick the best score_rate.
3. **King-safety leaf** at the best τ: `--search-leaf material-king-safety`; keep only if score_rate improves.
4. **Confirm** the best variant at `--games 100`; compare to baseline against the ship bar.

Per-variant diagnostic (reuse the override-vs-agreement CPL split from the v1 analysis): a healthy variant has override-CPL ≤ agreement-CPL. One-time read-only **root top-K coverage** check over the baseline `records.jsonl` (how often a better move is even in the top-K) tells you whether to raise `--search-k`.

Record the winning config and its headline numbers in the memory catalog (`project_inference_search_eval.md`), updating the "search v1 regresses" note with the v2 result.
```

