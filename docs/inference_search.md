# Inference-Time Search Over LM Top-K

A future-improvement note. Not implemented; not on the current roadmap.
Captured here so it can be evaluated against other phase-3+ work.

## TL;DR

The current model decides each move with one forward pass and `argmax`. Most
of its blunders are "I picked a plausible move and didn't notice my opponent
has a free piece next turn." A tiny game-tree search at inference time —
**LM as policy, simple function as value, depth 2** — would catch the bulk of
those blunders without any retraining. Plausible lift: **+200–400 Elo** for
the cheapest version, more with a better leaf evaluator.

This is the AlphaZero recipe scaled way down: a strong policy plus a small
search amplifies multiplicatively, and we already have the policy.

## Motivation

The skill-20 vs-Stockfish ladder (`evals/2026-04-30T*_skill20_*`) shows the
model winning **zero games** against full-strength Stockfish at depth ≥ 2. At
depth 2 it draws 50%; at depth 4 it converts almost nothing. The dominant
failure mode is a single tactical oversight per game: the LM picks a
positionally-reasonable move, the opponent collects material on the
following move, the model never recovers.

Search at depth 2 looks one ply past your own move. Almost every blunder of
the form "didn't notice the immediate refutation" disappears.

## Sketch

```
For my move:
  candidates ← top-K from LM(position)         # K ≈ 3–5
  for m in candidates:
    push m
    replies ← top-K from LM(position)          # opponent
    for r in replies:
      push r
      score(m, r) ← leaf_eval(position)
      pop r
    score(m) ← min over r of score(m, r)       # opponent picks worst-for-me
    pop m
  play argmax score(m)
```

Cost: K² forward passes per model move. At K=5 that is 25× a single forward
— from ~10 ms/move to ~250 ms/move on GPU. Still fast in absolute terms.

## Leaf evaluators, ranked by cost vs. lift

The project's thesis is **LM-driven prediction**. Using an external engine
(e.g. Stockfish) at the leaves would short-circuit that thesis at inference
time and is explicitly out of scope for this design. The options below stay
inside the "learned or hand-crafted, no engine" space.

| Evaluator | Implementation cost | Inference cost | Estimated lift |
|---|---|---|---|
| Material count (Q=9, R=5, B=N=3, P=1) | ~20 LOC | Negligible | +150–300 Elo |
| Material + simple positional (king safety, mobility, passed pawns) | ~100 LOC | Negligible | +250–400 Elo |
| Trained value head (predict outcome from FEN) | New training run | One forward / leaf | +400–700 Elo |
| LM logprob of "good continuation" tokens at the leaf | ~50 LOC | One forward / leaf | Untested; worth measuring |

The cheapest row — **K=3, depth=2, plain material eval** — is the obvious
first cut. ~150 LOC end-to-end and catches roughly 80% of one-move blunders.
The bottom row keeps inference end-to-end LM-driven (the search uses the
same model for both the policy and the leaf score) and is the most
on-thesis option once the cheap material eval saturates.

## Why this stacks with the rest of the plan

- **DPO improves the policy; search amplifies a better policy.** Quality DPO
  (Phase 3) widens the candidate set to include better moves; search picks
  the right one. Either alone is good; together they multiply.
- **The eval harness already captures the policy distribution.** `model_top_k`
  on every per-move record (`docs/eval_harness.md` §Per-move record) is
  exactly the input search would consume. No new data collection.
- **Adding a value head later is a drop-in.** The search wrapper does not
  change when the leaf evaluator does; same minimax, smarter leaves.

## Open questions / risks

- **KV cache management on push/pop branches.** Each branch invalidates the
  cache for its sibling. Options: snapshot the cache before each branch
  (memory cost K²), or accept the re-forward (compute cost). Solvable but
  this is where most implementation complexity lives.
- **Top-K coverage.** If the LM's top-K never contains the right move, search
  cannot save you. Empirically K=3 captures the right move on the strong
  majority of positions for chess transformers; K=1 is useless. Worth
  measuring on existing eval records before committing.
- **Determinism vs. exploration.** Greedy minimax over top-K is deterministic
  given the policy; the same opening gets the same line every game. Fine for
  evaluation, may want softmax-weighted leaf scoring for self-play data
  generation.
- **Inference-time complexity creep.** A search wrapper is a second moving
  part next to the LM. Every change to one needs to be tested against the
  other. The current single-forward inference is dead simple.

## Decision criteria

Reach for this when:
- Quality DPO (Phase 3) has shipped and the policy improvements are
  saturating.
- The legality DPO (Phase 2) has driven `legal_rate` high enough that
  unconstrained inference is viable — search on top of constrained decoding
  is fine, but the value-of-search story is clearer once masking is no
  longer doing the heavy lifting.
- A real strength target above ~2000 Elo is in scope; below that, more
  training data or a bigger model is probably easier per unit work.

Skip this in favor of a bigger model when:
- Compute is the bottleneck rather than implementation. Going 0.6B → 3B
  on imitation alone likely outperforms 0.6B + depth-2 search per
  engineering hour.
- The downstream use case wants single-forward inference latency
  (~10 ms/move) and cannot tolerate ~250 ms/move.

## References

- Ruoss et al., *Grandmaster-Level Chess Without Search* (2024) — the
  upper-bound recipe (270M model, distill Stockfish action-values, no search,
  ~2895 Elo). Search is the alternative to that distillation pipeline.
- Karvonen, *Chess-GPT* — small chess transformers + minimax-over-top-K is
  a community-known pattern; documented gains roughly match the table above.
- AlphaZero (Silver et al., 2018) — original "policy net + MCTS" recipe;
  the scaling intuition transfers even though MCTS is overkill here.
