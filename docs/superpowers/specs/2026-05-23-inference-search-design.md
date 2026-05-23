# Inference-Time Search Over LM Top-K — Design

Status: approved design, not yet implemented.
Supersedes the future-improvement note in `docs/inference_search.md` for the
scope captured here (depth-2, material leaf, pluggable-leaf seam).

## Goal

Add a depth-2 minimax search that selects a move from the LM's top-K
candidates, scoring leaves with a pluggable evaluator (material-count first).
The search is packaged as a drop-in `Predictor` so it runs through the existing
`vs_stockfish` harness with no changes to records, metrics, or DPO export. The
purpose of this PR is a correct, tested first cut; the end-to-end Elo-lift
measurement is a follow-on eval run.

## Scope

In scope:
- Depth-2 minimax over the policy's legal top-K (my move → opponent reply → leaf).
- A `LeafEvaluator` seam with one concrete implementation: `MaterialEvaluator`.
- A `SearchPredictor` implementing the `inference.Predictor` protocol, with a
  `predict` convenience for play loops.
- Adding `predict` to the `inference.Predictor` protocol (+ trivial `predict`
  on the existing eval test stubs to keep type-checking).
- `--search` / `--search-k` flags on `scripts/eval.py vs-stockfish`.
- Unit tests on hand-built positions + a smoke run through the harness with a
  stub policy.

Out of scope (named future work, designed for but not built):
- DPO distillation of search into the policy (search pick = chosen,
  policy argmax = rejected). The data needed for it lives in `SearchResult`
  today; lighting it up later is an additive schema change only.
- Configurable depth (> 2).
- Additional leaf evaluators (material + positional, trained value head,
  LM-logprob leaf).
- KV-cache snapshotting for sibling-branch reuse.
- The real Elo-lift A/B eval run.

## Integration point

`scripts/eval.py` builds a `ChessInferenceEngine` and hands it to
`run_vs_stockfish` as a `Predictor`. Search slots in by wrapping that engine in
a `SearchPredictor` that *also* satisfies `Predictor`. The harness, the
record/metrics layer, and the DPO exporter are untouched.

## Cost

Depth-2 with a board-only (material) leaf is `1 + k` forwards per model move:

- The root forward (my top-K) is the pass the harness performs anyway.
- Forwarding each candidate move `m` produces the *opponent's* reply
  distribution as a side effect of the same pass (the model predicts the next
  move in the sequence).
- The `k²` leaf scores are pure board arithmetic — no forward.

At k=3–5 that is ≈ 40–60 ms/move, comfortably under the 250 ms budget in
`docs/inference_search.md`. **v1 therefore needs no KV-cache snapshotting.**
Branches use cold forwards on copied boards; the live `Game` is never
perturbed.

## Module layout

```
src/halluci_mate/search/
  __init__.py        # minimal
  leaf.py            # LeafEvaluator protocol + MaterialEvaluator
  minimax.py         # run_search(...) -> SearchResult; SearchResult, CandidateScore
  predictor.py       # SearchPredictor (implements inference.Predictor)
tests/search/
  __init__.py
  leaf_test.py
  minimax_test.py
  predictor_test.py
```

`search/` depends on `chess`, `game.Game` / `game.Perspective`, and
`inference.Predictor` / `inference.MovePrediction`. It does **not** import from
`eval/`; the harness depends on search only through the `Predictor` protocol.

## Components & interfaces

### `LeafEvaluator` (protocol) — the pluggable seam

```python
class LeafEvaluator(Protocol):
    def evaluate(self, board: chess.Board, *, pov: chess.Color) -> float: ...
```

Higher = better for `pov`. `MaterialEvaluator` is the one concrete impl.

### `run_search` — pure function, no I/O

```python
def run_search(policy: Predictor, leaf: LeafEvaluator, game: Game, *, k: int) -> SearchResult
```

### `SearchResult` — the future-distillation seam

```python
@dataclass(frozen=True)
class CandidateScore:
    move: chess.Move
    policy_logprob: float    # from root top-K
    score: float             # minimax value, my POV, after opponent's best reply

@dataclass(frozen=True)
class SearchResult:
    candidates: list[CandidateScore]   # score-sorted, descending
    chosen: chess.Move                 # argmax score
    policy_argmax: chess.Move          # what plain inference would have played
    root_prediction: MovePrediction    # carries model_top_k / raw_sample_* / mask_used
```

`run_search` computes everything `SearchResult` holds in order to run minimax,
so the distillation data (`policy_argmax`, per-candidate `score`) is free.

### `SearchPredictor` — implements `Predictor`, wraps a policy

```python
class SearchPredictor:
    def __init__(self, policy: Predictor, leaf: LeafEvaluator, *, k: int) -> None  # k >= 1
    def predict_with_metadata(self, game, *, constrained=None, record_top_k=5) -> MovePrediction
    def predict(self, game, constrained=None) -> chess.Move  # Predictor.predict; always legal, never raises
```

`predict` is a thin wrapper over `predict_with_metadata` that returns
`played_move`. For search the chosen move always comes from the legal top-K, so
`played_move` is never `None` and `predict` never raises `IllegalMoveError`. The
`constrained` argument is accepted for signature parity and ignored, like
`predict_with_metadata`'s.

### `predict` added to the `Predictor` protocol

`predict(self, game, constrained=None) -> chess.Move` is added to the
`inference.Predictor` protocol (it currently declares only
`predict_with_metadata`). No *in-repo* consumer calls `predict` through the
protocol today — but third-party callers do: the Lichess bot builds against the
inference engine via `predict()`, and is exactly the consumer that types against
the `Predictor` abstraction and swaps implementations
(`ChessInferenceEngine` ↔ `SearchPredictor`) behind it. Putting `predict` on the
protocol is what makes `SearchPredictor` a genuine drop-in there, rather than a
convenience that only concrete-typed code can reach.

`predict`'s protocol contract is "return a legal move or raise
`IllegalMoveError`." `ChessInferenceEngine` may raise (an unconstrained illegal
sample); `SearchPredictor` never does.

Consequence: every object that *structurally* satisfies `Predictor` now needs
`predict`. `ChessInferenceEngine` already has it. The existing eval test stubs
(`_StubEngine` in `vs_stockfish_test` / `eval_test`, `_ConstantEngine` in
`legal_rate_test`, the two local `_IllegalEngine` classes) each gain a trivial
`predict` to keep type-checking — a small, mechanical update this PR absorbs.
Search test stubs are written with `predict` from the start.

## Algorithm (depth-2)

POV = the model's side, derived from `game.perspective`
(`Perspective.WHITE` → `chess.WHITE`, etc.). At a model decision,
`board.turn == pov`.

1. `root = policy.predict_with_metadata(game, constrained=True, record_top_k=k)`.
   Candidates = the legal moves parsed from `root.model_top_k` (up to `k`).
2. For each candidate `m`: copy the board with `stack=True` (so tokenization
   sees the move history), push `m`.
   - If that position is game-over → no reply forward;
     `score(m) = leaf.evaluate(board_after_m, pov=pov)` (handles mate/stalemate).
   - Else build a fresh `Game(board=board_after_m, perspective=game.perspective,
     cache=None)` and call `policy.predict_with_metadata(...)`. For each reply
     `r` (legal top-K), `score(m, r) = leaf.evaluate(board_after_m + r, pov=pov)`.
     `score(m) = min_r score(m, r)` (opponent minimizes my score).
3. `chosen = argmax_m score(m)`; tie-break by `policy_logprob` descending, then
   UCI string — fully deterministic.

## Leaf evaluator (material)

Piece values: Q=9, R=5, B=N=3, P=1, K=0. `MATE_VALUE = 1_000_000.0` as a named
module constant.

- Checkmate → `-MATE_VALUE` if `board.turn == pov` else `+MATE_VALUE`.
- Stalemate / insufficient material → `0.0`.
- Otherwise white-relative material sum, sign-flipped to `pov`.

## MovePrediction mapping (no schema change)

```python
def predict_with_metadata(self, game, *, constrained=None, record_top_k=5):
    result = run_search(self.policy, self.leaf, game, k=self.k)   # queries policy constrained
    root = result.root_prediction
    return replace(root,
                   played_move=result.chosen,
                   model_move_uci=result.chosen.uci(),
                   model_top_k=root.model_top_k[:record_top_k])
```

`model_top_k`, `raw_sample_*`, and `mask_used` pass through from the policy's
own prediction. Only `model_move` changes. The resulting
`played_move != model_top_k[0]` divergence is the future distillation signal,
already captured in `SearchResult`.

Search **always queries the policy constrained** — candidates must be legal
moves, so searching over an unconstrained (possibly illegal) top-K is
meaningless. The `constrained` argument on `predict_with_metadata` is
intentionally ignored; this is documented in the method docstring. As a
consequence `mask_used` is `True` on search records.

`run_search` queries the root at breadth `k`, so the recorded `model_top_k`
holds the `k` search candidates (sliced to `record_top_k` when that is
smaller). Recording more than `k` candidates is not supported in v1, since
search only reasons over `k` — set `--record-top-k <= --search-k`.

## Public API / third-party usage

The package follows the repo convention: empty `__init__.py`, import from
submodules (mirrors `halluci_mate.eval`). Three tiers, in increasing control:

**1. Drop-in policy replacement.** `SearchPredictor` satisfies the
`inference.Predictor` protocol, so it goes anywhere a `ChessInferenceEngine`
goes:

```python
import chess
from halluci_mate.inference import ChessInferenceEngine
from halluci_mate.game import Game, Perspective
from halluci_mate.search.predictor import SearchPredictor
from halluci_mate.search.leaf import MaterialEvaluator

policy = ChessInferenceEngine.from_checkpoint("runs-v2a-ft/.../checkpoint-9687")
engine = SearchPredictor(policy=policy, leaf=MaterialEvaluator(), k=3)

game = Game(board=chess.Board(), perspective=Perspective.WHITE)
move = engine.predict(game)                        # chess.Move, always legal
pred = engine.predict_with_metadata(game)          # MovePrediction (model_move may != model_top_k[0])
```

**2. Full result.** `run_search` returns the per-candidate scores and the
policy argmax — for analysis or the future distillation path:

```python
from halluci_mate.search.minimax import run_search

result = run_search(policy, MaterialEvaluator(), game, k=3)   # SearchResult
result.chosen          # chess.Move
result.policy_argmax   # what the LM alone would have played
for c in result.candidates:        # score-sorted CandidateScore list
    ...                            # c.move, c.policy_logprob, c.score
```

**3. Custom leaf.** Any object matching `LeafEvaluator` plugs in without
touching the package:

```python
class MyLeaf:
    def evaluate(self, board: chess.Board, *, pov: chess.Color) -> float: ...

engine = SearchPredictor(policy=policy, leaf=MyLeaf(), k=5)
```

## CLI wiring

`scripts/eval.py vs-stockfish` gains:

- `--search / --no-search` (default off)
- `--search-k` (default 3)

After building `engine`, when `--search` is set:
`engine = SearchPredictor(policy=engine, leaf=MaterialEvaluator(), k=search_k)`,
then pass it to `run_vs_stockfish` as before. `extra_config` records
`{"search": True, "search_k": k, "search_leaf": "material"}` into `config.json`.

## Error handling & edge cases

- `k < 1` → `ValueError` in `SearchPredictor.__init__`.
- The root is always non-terminal: the game loop guards `is_game_over` before a
  model turn. A terminal root propagates `GameOverError` from the policy, same
  as today.
- Candidate moves from `model_top_k` are re-validated as legal via the existing
  parse helper (trust-but-verify the post-mask contract).
- A candidate move that is itself game-over is scored directly, with no reply
  layer.

## Testing (success criteria)

- **`leaf_test.py`**: start position = 0; side up a queen = ±9 by POV;
  fool's-mate board = ±MATE_VALUE; stalemate and K-vs-K = 0.
- **`minimax_test.py`** with a scripted **stub `Predictor`** (top-K keyed by
  FEN): (a) mate-in-1 chosen over a material grab; (b) a hanging-piece trap
  where policy-argmax ≠ search pick — proves search changes the decision;
  (c) terminal-after-my-move path; (d) deterministic tie-break.
- **`predictor_test.py`**: returned `MovePrediction` has `model_move` = search
  pick while `model_top_k` / `raw_sample_*` / `mask_used` are preserved from the
  policy; `k < 1` raises.
- **Smoke**: drive `run_vs_stockfish` with a `SearchPredictor` over a stub
  policy and a stub Stockfish (reusing the existing `vs_stockfish_test` stubs);
  assert records are produced. The real Elo-lift A/B is a follow-on run.

## Known limitations (accepted for v1)

- **Horizon effect.** Depth-2 + material has no quiescence; search may grab a
  piece that is recaptured at ply 3.
- **Opponent perspective.** Replies are generated with *my* perspective token,
  because the model was trained with a single `<WHITE>` / `<BLACK>` token for
  the whole game; flipping mid-sequence is out-of-distribution. Flipping for the
  opponent layer is an open question, not built.
- **Top-K coverage.** If the right move is never in the policy's top-K, search
  cannot find it. Measurable on existing eval records before scaling up.
- **No cache reuse.** Cold forwards on copied boards; acceptable at this
  latency. Snapshotting is named future work.
