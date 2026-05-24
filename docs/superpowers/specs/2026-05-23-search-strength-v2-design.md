# Inference-Time Search v2 — Net-Strength Redesign

Status: implemented on branch `jpaulsendev/inference-search` (2026-05-23); strength
A/B (the measurement runbook) still pending.
Builds on `docs/superpowers/specs/2026-05-23-inference-search-design.md` (the v1
depth-2 material search) and the negative eval result recorded after it.

## Background — why v1 must change

The v1 search (depth-2 minimax over the policy's legal top-K, material leaf,
opponent replies drawn from the same policy) **regressed** `jspaulsen/halluci-mate-v2d`
against Stockfish:

| Metric (100g, skill 5 / depth 12, --sf-analyze, t=0, alternate) | v2d (no search) | v2d + search k=3 |
|---|---|---|
| score_rate | 0.090 (1W/16D/83L) | 0.055 (1W/9D/90L) |
| CPL median | 14 | 19 |
| blunder endgame | 7.14% | 8.67% |

Eval dirs: baseline `evals/2026-05-18T20-34-08_v2b-dpo-broad-sharp-welcoming-skink-500_vs-stockfish`,
search `evals/2026-05-24T00-16-58_v2d-search_vs-stockfish`.

Root cause (within-run analysis of the search arm): search overrode the policy
argmax on **24.1%** of moves; the regression is entirely in those overrides
(CPL median 38 vs 14 on agreements; blunder 8.18% vs 4.75%). Three structural
causes:

1. **Fake opponent model.** Opponent replies were drawn from the same
   ~club-strength policy's top-K, so the `min`-over-replies missed Stockfish's
   real refutation. Candidates looked safe and were not.
2. **Material-only leaf, no quiescence.** Could not see recaptures or a forced
   mate one ply past the depth-2 horizon (e.g. `game-0080`: played a quiet rook
   move into `...Re1+ Rxe1 Rxe1#`, a mate the material leaf at ply 2 scored as
   merely "−2 material").
3. **Unconditional re-ranking.** Search replaced the argmax even on weak
   evidence, so when its model was wrong it actively chose worse.

## Goal & success criterion

Make inference-time search **net-improve** v2d's strength against Stockfish.

**Ship bar:** `score_rate` at skill 5 / depth 12 / `--sf-analyze` / t=0 /
alternate / 100 games **> 0.090** (strictly beats the no-search baseline).
Search's overrides must net-help, not net-hurt.

## Constraints

- **No external chess engine in the inference loop.** The leaf evaluator stays
  hand-crafted or LM/learned (project thesis, `docs/inference_search.md`).
  Stockfish is used only as the *eval opponent/oracle*, never inside search.
- **No retraining this iteration.** Inference-only levers: opponent model,
  quiescence, leaf terms, selection policy, depth/k. A trained value head is
  explicitly deferred (revisit only if these saturate below the bar).
- `SearchPredictor` stays a drop-in `inference.Predictor`; the harness,
  records, metrics, and DPO export are untouched.

## Approach (chosen)

"Make it a real search": the LM proposes root candidates; a thin, board-only
tactical search verifies them.

1. **Full-width, LM-free opponent layer.** Replace the policy-generated replies
   with *all* legal opponent moves, resolved by quiescence. The LM is called
   **only at the root** to propose the k candidates. (The v1 reply forward
   existed only to stay "LM-driven", but the leaf is board-only, so enumeration
   is both more correct and cheaper.)
2. **Quiescence with capture + check extension** to kill the horizon effect.
3. **Material + king-safety leaf** for quiet-position judgment (king-safety
   measured separately; quiescence handles the *forced* tactics).
4. **Margin gate** in `SearchPredictor`: override the policy argmax only when
   search's score beats it by τ pawn-equivalents.

Rejected alternatives: "blunder-filter only" (gate alone — too low a ceiling,
leaves the bad evaluation unfixed); "richer LM-driven search" (LM-logprob leaf /
opponent perspective-token flip / depth-3 — most moving parts, brings back
per-leaf forwards, perspective flip is out-of-distribution for this model).
Both are a subset / superset of the chosen approach and remain fallbacks.

## Cost

Per model move: **1 LM forward** (root candidates only) + board-only search.
Lower than v1's `1 + k` forwards. The reply layer enumerates all legal opponent
moves (~30) and each runs a quiescence search over forcing moves, capped at
`qdepth`; all `python-chess` arithmetic, no GPU. Quiescence is bounded by the
cap for guaranteed termination (perpetual-check safety).

## Module layout (changes only)

```
src/halluci_mate/search/
  leaf.py        # + MaterialKingSafetyEvaluator (MaterialEvaluator kept)
  minimax.py     # opponent layer -> full-width legal replies; + quiescence
  predictor.py   # + margin gate (selection policy)
scripts/eval.py  # + --search-leaf / --search-margin /
                 #   --search-quiescence / --search-qdepth
tests/search/    # + quiescence / king-safety / gate unit tests
```

`run_search` stays a pure scorer (no I/O); the margin gate is a
`SearchPredictor`-level selection policy, keeping `SearchResult` intact as the
future distillation seam.

## Algorithm (depth-2, quiescent leaf)

POV = the model's side (`game.perspective`). At a model decision
`board.turn == pov`.

```
root = policy.predict_with_metadata(game, constrained=True, record_top_k=k)   # only LM call
candidates = legal moves parsed from root.model_top_k        (up to k)
for m in candidates:
    push m onto a board copy (stack=False; no move history needed — the LM is not queried below the root)
    if board_after_m is game-over:
        score(m) = leaf.evaluate(board_after_m, pov)         # terminal (mate/draw)
    else:
        score(m) = min over ALL legal replies r of:
                       quiesce(board_after_m_then_r, pov, qdepth)
chosen = argmax_m score(m)        # tie-break: policy_logprob desc, then UCI (deterministic)
```

### Quiescence

`quiesce(board, pov, qdepth) -> float`, **fixed-POV minimax** with stand-pat
(the leaf is always scored from `pov`, so a node maximizes when
`board.turn == pov` and minimizes otherwise — no per-ply sign flip):

- If the board is terminal: return the leaf's mate/draw value (so a forced mate
  within the cap is found — this is what rejects `d4d7`).
- If `qdepth == 0` or the position is quiet (not in check and no capture to
  extend): return `leaf.evaluate(board, pov)`.
- Stand-pat = `leaf.evaluate(board, pov)`: a lower bound at a `pov`-to-move
  (max) node, an upper bound at an opponent-to-move (min) node. Skipped when the
  side to move is in check (it cannot decline to respond).
- Extend **forcing moves only**: captures, plus *all* legal moves when the side
  to move is in check (evasions). Recurse with `qdepth - 1`, taking the max
  (`pov` to move) or min (opponent to move). Generating quiet checking moves is
  a possible later refinement, not in v1.

`qdepth` is capped (default 4) to guarantee termination.

## Leaf evaluator

`LeafEvaluator` protocol unchanged: `evaluate(board, *, pov) -> float`, higher =
better for `pov`, units = pawn-equivalents.

`MaterialKingSafetyEvaluator`:
```
score = material(board)  +  W_KING_SAFETY * king_safety(board)        # then sign-flip to pov
```
- `material`: existing weights (Q=9, R=5, B=N=3, P=1), ±MATE_VALUE on
  checkmate, 0 on stalemate / insufficient material.
- `king_safety`: cheap, both-sides-differenced heuristic — pawn-shield
  intactness in front of each king plus a penalty for open/semi-open files
  bearing on the king. Kept minimal; `W_KING_SAFETY` is a named module constant
  (pawn-equivalent units) tuned empirically.

`MaterialEvaluator` is retained for ablation (`--search-leaf material`).

## Margin gate (selection policy in SearchPredictor)

`run_search` returns `SearchResult` with score-sorted `candidates`, the
unconditional `chosen` (best by score), and `policy_argmax`. `SearchPredictor`
applies the gate to pick the *played* move:

```
best   = candidates[0]                          # highest minimax score
argmax = the candidate whose move == policy_argmax
played = best.move  if (best.score - argmax.score) >= margin  else argmax
```

- τ (`margin`, pawn-equivalents) is a constructor/CLI param.
  **τ → ∞ ⇒ baseline policy** (never override); **τ = 0 ⇒ always trust search.**
  Brackets "cannot do worse than v2d" on one end; swept to find net-positive.
- No extra cost: `argmax`'s score is already in the candidate set.
- `MovePrediction` mapping is unchanged except `played_move`/`model_move_uci`
  now reflect the gated choice; `model_top_k` / `raw_sample_*` / `mask_used`
  still pass through from the policy. The `played_move != model_top_k[0]`
  divergence remains the distillation signal.

## CLI

`vs-stockfish` gains (alongside existing `--search` / `--search-k`):

- `--search-leaf {material, material-king-safety}` (default `material-king-safety`)
- `--search-margin FLOAT` (τ, pawn-equivalents; starting default `1.0`, refined by the τ sweep; `0` = always-override, large = baseline)
- `--search-quiescence / --no-search-quiescence` (default on)
- `--search-qdepth INT` (default 4)

`extra_config` records `search`, `search_k`, `search_leaf`, `search_margin`,
`search_quiescence`, `search_qdepth` into `config.json` so every run is
self-describing.

## Measurement & iteration

Three tiers, cheap → expensive:

1. **Unit tests (no LM).** Hand-built positions assert search avoids the known
   failures: the `game-0080` mate-in-2 (quiescence rejects `d4d7`); a recapture
   trap (don't grab a defended piece); a "gate holds" case (small edge → keep
   argmax). Deterministic, instant.
2. **Directional eval — 30g** vs skill-5 + `--sf-analyze` per variant
   (~minutes). Read override-rate and the override-vs-agreement CPL split before
   spending a full run.
3. **Headline confirm — 100g** (skill 5, depth 12, `--sf-analyze`, t=0,
   alternate) once a variant clears tier 2; compare to the v2d baseline against
   the ship bar (`score_rate > 0.090`).

One-time **root top-K coverage** check (read-only, over the existing baseline
`records.jsonl`): on positions v2d blundered, how often is a better move in the
LM top-K at all? Low coverage caps search's ceiling and argues for higher
`--search-k` (or flags the policy, not search, as the bottleneck).

**Increment order** (measure at tier 2 between each; stop when the bar is met):

1. Full-width LM-free opponent layer + quiescence (material leaf) — core fix.
2. Margin gate; sweep τ.
3. King-safety leaf; keep only if it adds marginal lift.

## Error handling & edge cases

- `k < 1` → `ValueError` (as today). `qdepth < 0` → `ValueError`.
- Root is always non-terminal (the game loop guards `is_game_over` before a
  model turn); a terminal root propagates as today.
- Candidate moves from `model_top_k` are re-validated legal via the existing
  parse helper.
- A candidate that is itself game-over is scored directly (no reply layer).
- Quiescence on a check with no legal evasions = checkmate → leaf mate value.
- Determinism preserved: same tie-break order; full-width reply enumeration is
  order-stable.

## Testing standards

`tests/search/<module>_test.py`, plain `assert`, `pytest.approx` for floats,
scripted stub `Predictor` keyed by FEN for search tests (no model load), real
`python-chess` boards. New quiescence, king-safety, and gate logic each get
focused unit tests. Run `uv run ruff check . && uv run ty check && uv run pytest`
before committing.

## Out of scope (named, not built)

- Trained value-head leaf (deferred; the higher-ceiling Phase-2 option).
- LM-logprob leaf and opponent perspective-token handling.
- Depth > 2; KV-cache snapshotting (cost is now 1 forward/move — moot).
- DPO distillation of search picks (the `SearchResult` seam still carries the
  data for it later).
