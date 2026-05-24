# Stockfish Distillation Corpus — Design

**Date:** 2026-05-23
**Status:** Approved (design); pending implementation plan
**Branch:** `worktree-stockfish-distill` (off `origin/main`)

## Goal

Generate a from-scratch pretraining corpus of **full Stockfish games** that the
model behavior-clones via plain cross-entropy, reusing the existing Phase-1
sequence format (`<WHITE> e2e4 … <EOS>`). This is distinct from the two existing
ways Stockfish meets the model:

- **Phase-1 SFT** — cross-entropy on full *human* Lichess games.
- **DPO** (`eval/dpo_export.py`) — `{chosen, rejected}` preference pairs that
  *nudge* an existing policy.

The distill corpus is a third thing: a **supervised teacher corpus** where
Stockfish is the teacher and the model clones whole engine games end-to-end.

## Locked decisions

| Decision | Choice |
| --- | --- |
| Learning signal | Full engine games, plain cross-entropy, Phase-1 `<WHITE> … <EOS>` format |
| Position source | Seed from real Lichess openings, then engine self-play to the end |
| Seed filter | High-Elo **Rapid + Classical** only (reuse `passes_highelo_filter`) |
| Engine strength | Full strength, fixed depth budget |
| Diversity | Small MultiPV "wobble" — softmax-sample among moves within a few cp of best |
| Scale / use | Large from-scratch corpus (1M+ games), pretrain on engine play alone |
| Architecture | Two-stage: resumable generator → raw game shards → tokenize/split |

## Architecture (two-stage)

```
Lichess stream ──► [Stage 1: engine self-play] ──► raw game shards (parquet) ──► [Stage 2: tokenize+split] ──► train/eval/test.parquet ──► train.py
   (opening seeds)   generate_distill_games.py        (durable artifact)          prepare_distill_data.py        (unchanged)
```

The cut point is deliberate. Generation is the expensive, fragile, distributable
part; raw games are a **durable artifact**. Re-tokenizing or re-splitting must
never re-run Stockfish. (This directly avoids the v2a "regen gotcha" where a
parquet→pgn/jsonl regeneration required replaying upstream work.)

## Stage 1 — Engine self-play generator

`scripts/generate_distill_games.py` (typer CLI) →
`src/halluci_mate/distill/engine_selfplay.py`.

### Seed stream

Reuse the existing streaming + filter path:

- `load_dataset("Lichess/standard-chess-games", split="train", streaming=True)`
- `Termination == "Normal"`
- `Event` contains `"rapid"` or `"classical"` (explicitly **not** blitz)
- `passes_highelo_filter(sample, min_elo, max_rating_diff, max_elo_gap)` from
  `data_preparation.py`, with the v2a defaults exposed as CLI options.

### Per-game procedure

1. Pull a seed; replay its first **~12 plies** (UCI) onto a `chess.Board`. Skip
   unparseable or too-short seeds (mirrors `process_game` returning `[]`).
2. Loop until termination:
   - `engine.analyse(board, Limit(depth=DEPTH), multipv=MULTIPV)` →
     list of `InfoDict`.
   - Keep candidate moves whose score is within `WOBBLE_CP` of the best move.
   - **Sample one candidate** (the "wobble"): softmax over the candidates'
     centipawn scores divided by `WOBBLE_TEMP` (lower temp → closer to always
     picking best; `WOBBLE_TEMP → ∞` → uniform over the in-band set). Push it.
3. **Adjudicate** to keep games finite and clean (full-strength self-play
   otherwise produces endless dead-drawn shuffles that are pure training noise):
   - **Resign:** `|eval| > RESIGN_CP` for `RESIGN_PLIES` consecutive plies →
     decisive result for the winning side.
   - **Draw:** `|eval| < DRAW_CP` for `DRAW_PLIES` plies after `DRAW_MIN_PLY`.
   - **Hard cap:** `MAX_PLIES`.
   - Natural terminations (checkmate, stalemate, insufficient material,
     threefold/fifty-move via `claim_draw=True`) take precedence.
4. Emit one raw record (see schema). `outcome` uses the project's existing
   `"white" | "black" | "draw"` vocabulary so Stage 2's `game_to_sequences`
   consumes it unchanged.

### Default knobs (depth is the dominant cost dial)

| Knob | Default | Notes |
| --- | --- | --- |
| `DEPTH` | 18 | Main cost dial |
| `MULTIPV` | 4 | Candidate breadth for wobble |
| `WOBBLE_CP` | 30 | cp band around best move |
| `WOBBLE_TEMP` | tune | Softmax temperature for in-band sampling |
| Seed plies | 12 | Human prefix retained before handoff |
| `RESIGN_CP` / `RESIGN_PLIES` | 700 / 4 | Adjudicated decisive |
| `DRAW_CP` / `DRAW_PLIES` / `DRAW_MIN_PLY` | 15 / 8 / 60 | Adjudicated draw |
| `MAX_PLIES` | 200 | Hard cap |

### Parallelism & resumability

- Multiprocessing pool; one `chess.engine.SimpleEngine` (Stockfish) per worker,
  `Threads=1`, modest `Hash`.
- Each worker owns its own shard files. A manifest tracks completed shards.
- Shards written **atomically** (temp file + rename); manifest updated *after*
  the rename — a crash mid-job resumes with no dupes or corruption.
- Per-game RNG seeded from the seed-game hash → the wobble is **reproducible**
  and a restart reproduces identical games.

### Raw shard schema

Parquet shards, one record per game (`src/halluci_mate/distill/raw_shards.py`,
pydantic model, style mirrors `eval/records.py`):

```jsonc
{
  "game_id": "...",
  "seed_source": "<lichess game id or hash>",
  "seed_plies": 12,
  "moves_uci": ["e2e4", "e7e5", ...],   // full game incl. seed
  "outcome": "white" | "black" | "draw",
  "termination": "natural" | "adjudicated-win" | "adjudicated-draw" | "max-plies",
  "engine": { "depth": 18, "multipv": 4, "wobble_cp": 30,
              "sf_version": "...", "nnue": "..." }
}
```

The engine version + NNUE net are recorded because Stockfish output is
build-dependent; the corpus is only reproducible against a pinned engine.

## Stage 2 — Tokenize & split

`scripts/prepare_distill_data.py` (typer CLI). Cheap; reuses
`data_preparation.py` almost verbatim.

- New `process_engine_game(record, tokenizer)` alongside `process_game`: skips
  PGN parsing (moves are already UCI), then
  `game_to_sequences(moves_uci, outcome) → tokenizer(...)`.
- **Stratification change:** the existing key is
  `elo_bucket | result | opening_family`. Engine continuations have no
  meaningful single Elo, so stratify on `result | opening_family` and **drop
  `elo_bucket`**. `opening_family` still derives from the seed's first move.
- Everything else is reused as-is: `build_stratified_splits`, 95/4/1
  `save_splits` → `train/eval/test.parquet`. `train.py` is untouched.

## Module layout

```
src/halluci_mate/distill/
  engine_selfplay.py   ← seeding, wobble selection, adjudication, one-game loop
  raw_shards.py        ← raw-shard pydantic schema + atomic reader/writer
src/halluci_mate/data_preparation.py   ← add process_engine_game
scripts/generate_distill_games.py      ← Stage 1 CLI (parallel, resumable)
scripts/prepare_distill_data.py        ← Stage 2 CLI (tokenize + split)
```

Score/cp helpers (white-relative cp, mate clamping, PV-first move) are patterned
on the existing `eval/evaluators/vs_stockfish.py`; shared helpers extracted if
the duplication is non-trivial.

## Error handling (per CLAUDE.md)

- Engine death → catch `chess.engine.EngineTerminatedError` / `EngineError`,
  log, respawn that worker's engine, drop the in-flight game, continue. Specific
  exception types only — never bare `except`.
- Bad/short seed → skip (yield nothing), like `process_game`.
- Atomic shard writes + post-rename manifest update for crash safety.

## Testing

- **Unit (stubbed engine, no real Stockfish):**
  - Wobble selection against a fabricated MultiPV `InfoDict` list — in-band
    filtering + RNG-seeded determinism.
  - Adjudication thresholds → termination labels.
  - One-game loop driven by a stub engine mirroring the `_StockfishEngine`
    Protocol stub in `tests/eval/evaluators/vs_stockfish_test.py`.
- **Stage 2:** `process_engine_game` on a known game → expected token sequences
  (real tokenizer, per CLAUDE.md "don't mock the tokenizer").
- **Integration (`tests/integration/`):** a few games through real Stockfish at
  shallow depth → a handful of parquet rows; plus a resumability test (write
  half the shards, simulate crash, restart, assert no dupes and completion).

## Out of scope

- Soft/policy distillation (MultiPV → KL soft targets) — richer but heavier; not
  this corpus.
- Per-position best-move SFT and additional DPO pairs — already covered by
  `dpo_export.py`.
- Training-time mixing with human data — this corpus is consumed standalone.
- Distributed/cluster orchestration — the generator is shardable and resumable
  so it *can* be fanned out, but the cluster harness itself is a separate concern.
```
