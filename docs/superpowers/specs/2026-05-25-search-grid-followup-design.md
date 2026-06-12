# Search v2 — τ×k grid follow-up (design)

Date: 2026-05-25
Status: design (approved, pending implementation plan)
Branch: `jpaulsendev/inference-search`
Predecessors: `2026-05-23-search-strength-v2-design.md` (shipped, PR #21)

## Goal

The shipped v2 search default (`k=3`, `leaf=material-king-safety`, `τ (margin)=1.0`,
quiescence on, qdepth 4) scores **0.175** vs `jspaulsen/halluci-mate-v2d` (100g,
Stockfish skill 5 / depth 12, `--sf-analyze`, t=0, alternate). This follow-up sweeps
the two knobs left untuned — the number of root candidates `k` and the margin gate `τ`
— to see whether a different setting beats 0.175 without regressing the guardrails the
default already wins on.

**A "no improvement" outcome is a valid, documented result.** If nothing clears the
baseline within noise, the v2 defaults stand and we record the negative result rather
than shipping a change chasing 30g/100g noise.

### Why these two knobs, and why jointly

- `τ` (`search_margin`) is a *post-search* gate: search overrides the policy argmax only
  when it beats the argmax by τ pawn-equivalents. Lower τ → more overrides (riskier, per
  the v1 cautionary tale); higher τ → more conservative. Changing τ changes no search
  compute, only which move is played.
- `k` (`search_k`) is the number of root candidates the opponent layer scores. Each extra
  candidate triggers another full-width board-only opponent search, so per-move search
  cost grows ~linearly in `k` (the LM is still queried once per move regardless of `k`).
  The search overhead is CPU-bound board evaluation, dwarfed by Stockfish depth-12
  analysis time.
- They **interact**: more candidates (higher `k`) give search more chances to find an
  override that clears the gate, so the best τ likely shifts with `k`. Tuning them on a
  joint grid (rather than independently) captures that.

## Success criteria

- **Primary metric:** `score_rate` (consistent with all prior search work).
- **Guardrails (a winning cell must not regress these):**
  - CPL median ≤ 14 (current default value)
  - blunder rate ≤ 4.34% (current default value)
  - legal_rate ≈ 98% (search leaves the recorded policy top-1 untouched, so this is
    expected to be ~flat regardless)
- **Reported as context:** override-rate (fraction of decisions where the played move
  differs from the policy argmax `model_top_k[0]`), CPL p95, tactical_oversight.

## Protocol — held fixed for comparability

Everything except `k` and `τ` matches the 0.175 baseline run exactly:

| Knob | Value |
|---|---|
| checkpoint | `jspaulsen/halluci-mate-v2d` |
| stockfish_skill | 5 |
| stockfish_depth | 12 |
| sf_analyze | on |
| temperature | 0.0 |
| top_k | 0 |
| halluci_color | alternate |
| max_plies | 400 |
| search_leaf | material-king-safety |
| search_quiescence | on |
| search_qdepth | 4 |

Only `--search-k` and `--search-margin` vary across cells.

## Stage 1 — 30g directional sweep (270 games)

A throwaway bash loop runs one `eval.py vs-stockfish` per cell over the 9-cell grid:

```
k ∈ {3, 5, 8}   ×   τ ∈ {0.5, 1.0, 1.5}
```

Each cell uses a distinct `--checkpoint-tag` of the form `v2d-gridk<K>t<TT>` (no
underscores — the tag rule forbids them), e.g. `k=8, τ=0.5` → `v2d-gridk8t05`,
`k=3, τ=1.0` → `v2d-gridk3t10`.

The `k=3, τ=1.0` cell is included as an **in-sweep anchor**. 30g numbers are not directly
comparable to the 100g 0.175 baseline; the anchor calibrates the sweep's noise floor so
the other cells can be read relative to the known-good default at the same sample size.

### Decision rule (which cells advance to 100g)

At 30g, one game ≈ 0.033 score_rate, so the single argmax cell is not trustworthy. Read
the grid as a **surface**:

1. Rank cells by `score_rate` (headline).
2. Break near-ties and sanity-check the region using the more-stable CPL-median and
   blunder columns (these move less per-game than score_rate at n=30).
3. Identify the best-performing *region* of the grid and carry the **top 1–2 cells**
   forward to the confirm stage.

The leaderboard is presented to the user and the confirm picks are approved **before** the
100g budget is spent.

## Stage 2 — 100g confirm (≤200 games)

Re-run the chosen 1–2 cells at 100g, same protocol.

**Win condition:** a cell replaces the v2 default only if it beats 0.175 on `score_rate`
*and* holds the guardrails (CPL median ≤ 14, blunder ≤ 4.34%, legal_rate ~flat). The
existing 100g baseline run (`evals/2026-05-25T03-25-36_v2d-ab-ks-100_vs-stockfish`,
score 0.175) is reused as the comparison; `k=3, τ=1.0` is not re-run at 100g unless a
winning cell ties it and a fresh same-batch head-to-head is wanted.

## The only committed code — `scripts/sweep_summary.py` (read-only)

A small read-only analysis script. `compare.py` is a two-checkpoint side-by-side
dashboard and does not distinguish same-checkpoint runs that differ only in search config,
so the sweep needs its own collector.

Responsibilities:

- Scan run dirs under an evals dir by `--checkpoint-tag` prefix (e.g. `v2d-grid`).
- For each run: read `config.json` for the search knobs (`search_k`, `search_margin`,
  `search_leaf`, `games`); load aggregated metrics via the existing `eval.compare` /
  `eval.metrics` loaders (`score_rate`, blunder, CPL median/p95, legal_rate,
  tactical_oversight).
- Derive override-rate from records: fraction of model decisions where the played move
  (`model_move_uci`) differs from the policy argmax (`model_top_k[0]`), reusing the
  existing record-loading helper.
- Print a leaderboard table sorted by `score_rate`, with the guardrail and context columns
  alongside, one row per cell (`k`, `τ` from config).

Constraints: read-only (no writes to run dirs), reuses existing loaders rather than
reimplementing metric math, has a unit test (synthetic run dirs / records), and follows
the project code-style bar (full type annotations, specific exceptions, functions < 40
lines, module-level imports). Reused verbatim at the confirm stage.

Alternative considered: fold it in as an `eval.py sweep-summary` subcommand to keep one
CLI. Deferred to the implementation plan; a standalone script keeps eval (run) and
analysis concerns separate.

## Outputs

- Update the `project-inference-search-eval` memory with the grid result (winning cell +
  its 100g numbers, or the documented "defaults hold" negative result), and the new eval
  dir names.
- **Conditional follow-on (only if a cell wins):** a separate small change updating the
  shipped CLI defaults (`search_k` / `search_margin` in `scripts/eval.py`) and the v2
  design-spec note. Gated on the confirmed 100g number; not part of this experiment's
  scope until a winner is established.

## Cost & sequencing

`--sf-analyze` dominates wall time (~3× a non-analyzing run). Total: 270 sweep games +
up to 200 confirm games. Checkpoints with the user:

1. **Before launching Stage 1** (270 games).
2. **After the sweep, before Stage 2** — present the leaderboard, approve confirm picks.

The sweep loop may run in the background.

## Out of scope

- Other leaf evaluators, quiescence depth, or new search features (this is a tuning sweep
  of two existing knobs only).
- Re-running the v1 config (established as worse than the bare policy).
- Changing the eval harness or search implementation beyond the read-only summary reader.
