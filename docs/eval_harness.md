# Evaluation Harness Design

This document describes the design of the halluci-mate evaluation harness: the
set of tools used to measure model strength, legality, and move quality after
(and between) training runs.

## Goals

- **One shape for many evals.** vs-Stockfish games, tactical puzzles, legal-move
  rate, perplexity — all produce the same kind of artifact, so tooling written
  for one works for all.
- **Capture once, aggregate many times.** Replaying games against Stockfish is
  expensive; computing a new statistic from stored records is not. Metrics are
  pure functions over records — adding a new one never requires a re-run.
- **Feed downstream training.** Captured per-move data is the seed for DPO
  preference datasets (legality DPO in Phase 2, quality DPO in Phase 3). The
  harness is designed so that DPO pair generation is a post-hoc transform over
  existing eval artifacts, not a separate data-collection pipeline.

## Non-goals (v1)

- No MLflow / W&B integration. Artifacts are plain files on disk; a charting
  layer can be bolted on later without changing the collection format.
- No first-class checkpoint-vs-checkpoint comparison. Run twice, diff the JSON.
- No live dashboards, no self-play arena.

## Core design

```
┌─────────────┐       ┌──────────────┐       ┌──────────────┐
│  Evaluator  │──────▶│   records    │──────▶│   metrics    │
│ (collects)  │       │  (JSONL on   │       │ (pure funcs) │
│             │       │     disk)    │       │              │
└─────────────┘       └──────────────┘       └──────────────┘
                              │
                              ▼
                      ┌──────────────┐
                      │ DPO exporter │
                      │ (post-hoc)   │
                      └──────────────┘
```

**Evaluators** are thin. Their job is to drive the model through some workload
(play N games, solve M puzzles, run the model on K held-out positions) and
emit one structured record per event. They do not compute aggregates.

**Records** are append-only JSONL. One file per eval run. A record is a single
model decision — a move played, a puzzle attempted, a position scored. Schema
is uniform where it can be (see below) and extended per-evaluator where it
must be.

**Metrics** are pure functions `records -> stats`. They live separately from
evaluators and can be rerun at any time without replaying the eval. New
metric? New function over existing records. No re-run.

**DPO exporter** is a separate post-hoc tool. Given a run with per-move
Stockfish analysis, it emits preference pairs suitable for DPO training.
Detailed in [DPO seed data](#dpo-seed-data) below.

## Run artifact layout

Each eval invocation produces a directory:

```
evals/
  <run-id>/
    config.json        # evaluator name + full args + checkpoint path + git sha
    records.jsonl      # per-event records, append-only
    metrics.json       # computed aggregates (re-derivable from records)
    games.pgn          # optional, for game-based evaluators
```

**Run ID format**: `<timestamp>_<checkpoint-tag>_<evaluator>`, e.g.
`2026-04-19T20-15-00_marvelous-deer-608-ckpt9660_vs-stockfish`. Readable,
sortable, and unique enough for local use.

**Cross-run analysis groups by `run_id`, not by `checkpoint`.** The
`checkpoint` field is a free-form path kept for reproducibility; the
human-readable training-run identifier lives in the `<checkpoint-tag>`
portion of `run_id` and is what charts should axis on. There is no
structured checkpoint object on records — that comparison shape (e.g.
step within a single training run) is not a v1 use case.

**Location**: `evals/` at the repo root, gitignored. Mirrors the existing
`runs-v1/` layout for training runs.

## Record schema

All records share a small common header; the rest is evaluator-specific.

### Common fields (every record)

| Field | Type | Notes |
|-------|------|-------|
| `run_id` | string | matches the run directory name |
| `event_id` | int | monotonic within a run |
| `evaluator` | string | `vs_stockfish`, `puzzles`, `legal_rate`, `perplexity` |
| `checkpoint` | string | path or tag of the model checkpoint |

### Per-move record (`vs_stockfish`, optionally `puzzles`)

One record per model decision (i.e., only on plies where
`side_to_move == model_side`), not one per ply. The opponent's reply to
`model_move` is captured as `prior_opponent_move` on the next record for
the same `game_id`.

| Field | Type | Notes |
|-------|------|-------|
| `game_id` | string | groups moves into a game |
| `ply` | int | 0-indexed half-move number |
| `phase` | string | `opening` (ply < 20), `middle` (20–60), `endgame` (60+) |
| `side_to_move` | string | `white` \| `black` |
| `model_side` | string | which color the model is playing in this game |
| `fen_before` | string | position before the move |
| `legal_moves` | list[string] | all legal UCI moves from `fen_before` |
| `model_move` | string | UCI move the model actually played |
| `model_top_k` | list[{move, logprob}] | top-K from the *sampled-from* distribution (post-mask if `mask_used`, else unconstrained); K configurable |
| `mask_used` | bool | was constrained decoding enabled |
| `raw_sample_move` | string | unconstrained top-1; equals `model_move` when `mask_used` is false |
| `raw_sample_legal` | bool | was `raw_sample_move` legal in `fen_before` |
| `prior_opponent_move` | string \| null | move at `ply - 1` that produced `fen_before`; null only when model is White and `ply == 0` |
| `sf_best_move` | string \| null | only if `--sf-analyze` is set |
| `sf_eval_before_cp` | int \| null | centipawns, white-relative |
| `sf_eval_after_cp` | int \| null | centipawns, white-relative |
| `centipawn_loss` | int \| null | max(0, eval_before_stm - eval_after_stm) |
| `is_blunder` | bool \| null | `centipawn_loss > blunder_threshold` |

### Per-puzzle record (`puzzles`)

| Field | Type | Notes |
|-------|------|-------|
| `puzzle_id` | string | Lichess puzzle ID |
| `rating` | int | Lichess puzzle rating |
| `themes` | list[string] | e.g. `["fork", "mateIn2"]` |
| `fen` | string | starting position |
| `solution` | list[string] | expected move sequence |
| `model_attempt` | list[string] | moves the model played |
| `solved` | bool | did the attempt match the solution |

### Per-legal-rate record (`legal_rate`)

One unconstrained-prediction-on-a-position event.

| Field | Type | Notes |
|-------|------|-------|
| `position_id` | string | source identifier |
| `fen` | string | position |
| `model_move` | string | top-1 unconstrained prediction |
| `legal` | bool | true if `model_move` is legal in `fen` |

### Per-perplexity record (`perplexity`)

One scored continuation. No `model_move`/`legal` here — perplexity does
not sample from the model, only scores known token sequences.

| Field | Type | Notes |
|-------|------|-------|
| `position_id` | string | source identifier (sequence id) |
| `fen` | string | prefix position the continuation is scored against |
| `token_logprobs` | list[float] | per-token log-probabilities of the actual continuation |

## Evaluators (v1)

### `vs_stockfish`

Plays N games against Stockfish at a configured skill/depth. Already
prototyped in `scripts/run_vs_stockfish.py`; migrate this into the harness.

Flags:

- `--games N`
- `--stockfish-skill {0..20}`, `--stockfish-depth` or `--stockfish-movetime`
- `--halluci-color white|black|alternate`
- `--max-plies`
- `--temperature`, `--top-k`, `--unconstrained`
- `--sf-analyze` — **flag-gated.** When set, run Stockfish analysis on every
  position to capture `sf_eval_before_cp`, `sf_eval_after_cp`,
  `centipawn_loss`, `is_blunder`. Adds significant wall-time; off by default.
- `--blunder-threshold-cp` (default 200) — threshold for `is_blunder`.

### `puzzles`

Runs the model on a Lichess puzzle set. Top-1 prediction per expected move;
sequence must match the full solution to count as solved.

Flags:

- `--puzzle-db path.csv` — Lichess puzzles DB format
- `--rating-min`, `--rating-max`, `--sample N` — select a subset
- `--themes fork,mateIn2,…` — filter by theme

### `legal_rate`

Runs the model **unconstrained** on a set of positions and records whether the
top-1 sample is legal. Used to measure Phase 2 DPO progress.

Flags:

- `--positions path.fen` or `--sample-from-games games.pgn --n 10000`

### `perplexity`

Token-level cross-entropy on held-out game sequences. Cheapest metric; run
regularly.

Flags:

- `--data path/to/sequences.jsonl`
- `--max-sequences N`

## Metrics

Metrics live in `src/halluci_mate/eval/metrics.py` as pure functions.
Structure (not an interface to subclass — just a convention):

```python
def compute_win_rate(records: Iterable[Record]) -> WinRateStats: ...
def compute_legal_rate(records: Iterable[Record]) -> float: ...
def compute_centipawn_loss(records: Iterable[Record]) -> CplStats: ...
def compute_blunder_rate(records: Iterable[Record]) -> BlunderStats: ...
def compute_puzzle_accuracy(records: Iterable[Record]) -> PuzzleStats: ...
```

Each takes records and returns a stats dataclass. A top-level `compute_all`
dispatches based on the evaluator field in the records. Output is written to
`metrics.json` in the run directory.

### Stratification

Records carry enough dimensions to stratify aggregates without rerunning:

- **Game phase** via `phase` (opening / middle / endgame)
- **Model color** via `model_side`
- **Opponent strength** via `config.json` (skill level recorded there)
- **Puzzle rating** via rating bucket (<1200, 1200–1600, 1600–2000, 2000+)

Stratified metrics are computed alongside the overall aggregate — keep this as
dict output keyed by stratum, not separate functions.

## CLI shape

Single entry point, `scripts/eval.py`, with subcommands:

```
uv run python scripts/eval.py vs-stockfish \
    --checkpoint runs-v1/marvelous-deer-608/checkpoint-9660 \
    --games 10 --stockfish-skill 5 --sf-analyze

uv run python scripts/eval.py puzzles \
    --checkpoint <ckpt> --puzzle-db data/lichess_puzzles.csv --sample 500

uv run python scripts/eval.py legal-rate \
    --checkpoint <ckpt> --sample-from-games data/test.pgn --n 10000

uv run python scripts/eval.py perplexity \
    --checkpoint <ckpt> --data data/test.jsonl

uv run python scripts/eval.py report <run-id>
    # recomputes metrics.json from records.jsonl

uv run python scripts/eval.py export-dpo <run-id> \
    --output data/dpo_blunders.jsonl --threshold 200
```

## DPO seed data

The harness is designed so that Phase 2 and Phase 3 DPO data can be
generated without re-running games — both come from `vs_stockfish` records.

### Phase 2 (legality DPO)

Any per-move record where the unconstrained raw sample was illegal
(`mask_used` and not `raw_sample_legal`) is a negative example. Pair with
the constrained (legal) move that was actually played as the positive.

```
chosen:   (fen_before, model_move)        # legal, post-mask
rejected: (fen_before, raw_sample_move)   # illegal, unconstrained top-1
```

`raw_sample_move` carries the rejected move directly so the exporter does
not depend on the configured top-k size or on whether `model_top_k` was
captured pre- or post-mask.

### Phase 3 (quality DPO)

Any per-move record with `--sf-analyze` where the model's move has
`centipawn_loss > threshold` and Stockfish's best move is available is a
negative example. Pair with Stockfish's best move as the positive.

```
chosen:   (fen_before, sf_best_move)
rejected: (fen_before, model_move)
```

Only valid on records where `sf_best_move` and `centipawn_loss` are populated
(i.e., the run was done with `--sf-analyze`).

Keep the exporter deliberately simple — threshold on CPL, optional dedup by
FEN, output as JSONL `{prompt, chosen, rejected}` in whatever format the DPO
trainer consumes. Schema changes to the DPO format should live in the
exporter, not the collection records.

## Incremental build order

1. **Module skeleton.** `src/halluci_mate/eval/{__init__.py, records.py,
   runs.py}`. Record dataclasses, run-directory writer/reader, run-id helper.
2. **`vs_stockfish` evaluator.** Migrate `scripts/run_vs_stockfish.py` logic
   into `src/halluci_mate/eval/evaluators/vs_stockfish.py`. Emit per-move
   records. No Stockfish analysis yet.
3. **Metrics module.** `compute_win_rate`, `compute_legal_rate` from
   `raw_sample_legal`, basic stratification. `metrics.json` output.
4. **`scripts/eval.py` entry point** with the `vs-stockfish` and `report`
   subcommands. Point the existing ad-hoc script at this.
5. **`--sf-analyze` flag.** Per-move Stockfish analysis, CPL, blunder
   tagging.
6. **`perplexity` and `legal_rate` evaluators** — both are small.
7. **`puzzles` evaluator.** Lichess puzzle DB ingestion.
8. **`export-dpo` subcommand.** Starts with the legality flavor (no
   `--sf-analyze` required), then the quality flavor.

Each step produces something runnable. The shape of records is the contract
between collection and aggregation — stabilize it early, evolve it with
additive fields only.
