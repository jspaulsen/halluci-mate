# Search v2 τ×k Grid Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sweep the two untuned search knobs (`search_k`, `search_margin`/τ) on a 9-cell grid to find a setting that beats the shipped v2 default (score_rate 0.175 vs `v2d`), staged as a 30g directional sweep then a 100g confirm.

**Architecture:** One small committed read-only script (`scripts/sweep_summary.py`) collects and ranks same-checkpoint runs that differ only in search config — something `compare.py` (a two-checkpoint dashboard) cannot do. Everything else is an operational runbook: a throwaway bash loop drives `eval.py vs-stockfish` over the grid; the summary script ranks the results; the top 1–2 cells are re-run at 100g and compared to the existing 0.175 baseline.

**Tech Stack:** Python 3.12, uv, typer (CLI, matching `scripts/eval.py`), the existing `halluci_mate.eval` library (`runs.RunReader`, `compare.discover_runs/load_or_compute_metrics/headline_metrics`), pytest, Stockfish.

Spec: `docs/superpowers/specs/2026-05-25-search-grid-followup-design.md`.

---

## File Structure

- **Create** `scripts/sweep_summary.py` — read-only leaderboard reader. Responsibility: scan an evals dir by checkpoint-tag prefix, parse each run's search knobs + headline metrics + derived override-rate, print a score-sorted table. Writes nothing.
- **Create** `tests/scripts/sweep_summary_test.py` — unit tests for the reader (pure `override_rate` + on-disk `collect_summaries`).
- **(Conditional, Task 6 only)** `scripts/eval.py:104,106-108` — bump the shipped `search_k` / `search_margin` defaults, *only if* a cell wins at 100g.
- **(Recording)** the `project-inference-search-eval` memory file — append the grid result.

Tasks 1 is code (TDD). Tasks 2–5 are operational (long-running evals + analysis + recording) — no unit tests, exact commands instead. Task 6 is a conditional code change gated on the Stage-2 result.

---

### Task 1: `scripts/sweep_summary.py` — read-only leaderboard reader

**Files:**
- Create: `scripts/sweep_summary.py`
- Test: `tests/scripts/sweep_summary_test.py`

- [ ] **Step 1: Write the failing `override_rate` tests**

Create `tests/scripts/sweep_summary_test.py`:

```python
"""Unit tests for scripts/sweep_summary.py."""

from __future__ import annotations

from pathlib import Path

import scripts.sweep_summary as sweep_summary
from halluci_mate.eval.records import TopKEntry
from halluci_mate.eval.runs import RunWriter
from tests.helpers.eval_records import make_per_move_record


def test_override_rate_counts_played_vs_argmax_disagreements() -> None:
    records = [
        make_per_move_record(0, model_move="e2e4", model_top_k=[TopKEntry(move="e2e4", logprob=-0.1)]),  # agree
        make_per_move_record(1, model_move="d2d4", model_top_k=[TopKEntry(move="g1f3", logprob=-0.2)]),  # override
        make_per_move_record(2, model_move="c2c4", model_top_k=[TopKEntry(move="b1c3", logprob=-0.3)]),  # override
    ]
    assert sweep_summary.override_rate(records) == 2 / 3


def test_override_rate_is_none_without_eligible_records() -> None:
    assert sweep_summary.override_rate([]) is None
    assert sweep_summary.override_rate([make_per_move_record(0, model_top_k=[])]) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/scripts/sweep_summary_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.sweep_summary'`.

- [ ] **Step 3: Create `scripts/sweep_summary.py` with `override_rate` + `CellSummary`**

```python
"""Read-only leaderboard over a set of vs-stockfish search-sweep runs.

Scans an evals directory for runs whose checkpoint-tag starts with a given
prefix and prints one row per run: the search knobs (`search_k`,
`search_margin`) from `config.json`, the headline metrics (`score_rate`, CPL
median/p95, blunder/legal/tactical rates) via the shared `eval.compare`
loaders, and the override-rate derived from records (fraction of model
decisions where the played move differs from the policy argmax). Rows are
sorted by `score_rate`, best first. Writes nothing.

    uv run python scripts/sweep_summary.py --tag-prefix v2d-grid
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from halluci_mate.eval.compare import RunEntry, discover_runs, headline_metrics, load_or_compute_metrics
from halluci_mate.eval.records import Evaluator, PerMoveRecord, Record
from halluci_mate.eval.runs import RunReader

DEFAULT_EVALS_DIR = Path("evals")


@dataclass(frozen=True)
class CellSummary:
    """One sweep cell: its search knobs plus headline + override metrics."""

    run_id: str
    search_k: int
    search_margin: float
    games: int
    score_rate: float
    wins: int
    draws: int
    losses: int
    cpl_median: float | None
    cpl_p95: float | None
    blunder_rate: float | None
    legal_rate: float | None
    tactical_oversight: float | None
    override_rate: float | None


def override_rate(records: Iterable[Record]) -> float | None:
    """Fraction of model decisions where the played move differs from the policy argmax.

    The policy argmax is ``model_top_k[0].move``; the played move is
    ``model_move`` (search's choice when search is on). Decisions with an empty
    ``model_top_k`` are skipped. Returns ``None`` when there are no eligible
    decisions.
    """
    eligible = 0
    overrides = 0
    for record in records:
        if not isinstance(record, PerMoveRecord) or not record.model_top_k:
            continue
        eligible += 1
        if record.model_move != record.model_top_k[0].move:
            overrides += 1
    return overrides / eligible if eligible else None
```

- [ ] **Step 4: Run to verify the `override_rate` tests pass**

Run: `uv run pytest tests/scripts/sweep_summary_test.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Write the failing `collect_summaries` test**

Append to `tests/scripts/sweep_summary_test.py`:

```python
def _write_run(evals_dir: Path, run_id: str, *, search_k: int, search_margin: float, score_rate: float, records: list) -> None:
    writer = RunWriter(evals_dir / run_id)
    writer.write_config(
        {
            "evaluator": "vs_stockfish",
            "run_id": run_id,
            "checkpoint": "jspaulsen/halluci-mate-v2d",
            "games": 30,
            "search": True,
            "search_k": search_k,
            "search_margin": search_margin,
            "search_leaf": "material-king-safety",
        }
    )
    writer.write_metrics(
        {
            "win_rate": {"overall": {"score_rate": score_rate, "wins": 1, "draws": 2, "losses": 27}},
            "centipawn_loss": {"overall": {"median": 14.0, "p95": 178.0}},
            "blunder_rate": {"overall": {"rate": 0.043}},
            "legal_rate": {"overall": {"rate": 0.983}},
            "tactical_oversight_rate": {"overall": {"rate": 0.12}},
        }
    )
    with writer as open_writer:
        for record in records:
            open_writer.append_record(record)


def test_collect_summaries_sorts_by_score_and_parses_knobs(tmp_path: Path) -> None:
    agree = make_per_move_record(0, model_move="e2e4", model_top_k=[TopKEntry(move="e2e4", logprob=-0.1)])
    override = make_per_move_record(1, model_move="d2d4", model_top_k=[TopKEntry(move="g1f3", logprob=-0.2)])
    _write_run(tmp_path, "2026-05-25T01-00-00_v2d-gridk3t10_vs-stockfish", search_k=3, search_margin=1.0, score_rate=0.10, records=[agree, override])
    _write_run(tmp_path, "2026-05-25T02-00-00_v2d-gridk8t05_vs-stockfish", search_k=8, search_margin=0.5, score_rate=0.20, records=[override])
    _write_run(tmp_path, "2026-05-25T03-00-00_other-run_vs-stockfish", search_k=3, search_margin=1.0, score_rate=0.99, records=[agree])

    summaries = sweep_summary.collect_summaries(tmp_path, "v2d-grid")

    # Sorted by score_rate desc; the non-matching "other-run" tag is excluded.
    assert [(cell.search_k, cell.search_margin) for cell in summaries] == [(8, 0.5), (3, 1.0)]
    assert summaries[0].score_rate == 0.20
    assert summaries[1].override_rate == 0.5  # 1 of 2 decisions overridden
```

- [ ] **Step 6: Run to verify the new test fails**

Run: `uv run pytest tests/scripts/sweep_summary_test.py::test_collect_summaries_sorts_by_score_and_parses_knobs -v`
Expected: FAIL — `AttributeError: module 'scripts.sweep_summary' has no attribute 'collect_summaries'`.

- [ ] **Step 7: Implement `summarize_run`, `collect_summaries`, formatting, and the CLI**

Append to `scripts/sweep_summary.py`:

```python
def summarize_run(entry: RunEntry) -> CellSummary:
    reader = RunReader(entry.run_dir)
    config = reader.read_config()
    headline = headline_metrics(Evaluator.VS_STOCKFISH, load_or_compute_metrics(entry))
    return CellSummary(
        run_id=entry.run_id,
        search_k=int(config["search_k"]),
        search_margin=float(config["search_margin"]),
        games=int(config["games"]),
        score_rate=float(headline.get("score_rate", 0.0)),
        wins=int(headline.get("wins", 0)),
        draws=int(headline.get("draws", 0)),
        losses=int(headline.get("losses", 0)),
        cpl_median=headline.get("cpl_median"),
        cpl_p95=headline.get("cpl_p95"),
        blunder_rate=headline.get("blunder_rate"),
        legal_rate=headline.get("legal_rate"),
        tactical_oversight=headline.get("tactical_oversight"),
        override_rate=override_rate(reader.read_records()),
    )


def collect_summaries(evals_dir: Path, tag_prefix: str) -> list[CellSummary]:
    summaries = [
        summarize_run(entry)
        for entry in discover_runs(evals_dir)
        if entry.evaluator is Evaluator.VS_STOCKFISH and _run_tag(entry.run_id).startswith(tag_prefix)
    ]
    summaries.sort(key=lambda cell: cell.score_rate, reverse=True)
    return summaries


def _run_tag(run_id: str) -> str:
    """Extract the tag segment from a `<timestamp>_<tag>_<evaluator>` run-id."""
    parts = run_id.split("_")
    return parts[1] if len(parts) >= 3 else ""


def _fmt(value: float | None, *, places: int = 1) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}"


def format_leaderboard(summaries: list[CellSummary]) -> str:
    header = f"{'k':>2} {'τ':>4} {'games':>5} {'score':>6} {'W/D/L':>9} {'cpl_med':>7} {'cpl_p95':>7} {'blndr%':>7} {'ovrd%':>6} {'legal%':>7} {'tact%':>6}"
    lines = [header, "-" * len(header)]
    for cell in summaries:
        wdl = f"{cell.wins}/{cell.draws}/{cell.losses}"
        lines.append(
            f"{cell.search_k:>2} {cell.search_margin:>4.1f} {cell.games:>5} {cell.score_rate:>6.3f} {wdl:>9} "
            f"{_fmt(cell.cpl_median, places=0):>7} {_fmt(cell.cpl_p95, places=0):>7} {_pct(cell.blunder_rate):>7} "
            f"{_pct(cell.override_rate):>6} {_pct(cell.legal_rate):>7} {_pct(cell.tactical_oversight):>6}"
        )
    return "\n".join(lines)


def main(
    tag_prefix: Annotated[str, typer.Option(help="Only summarize runs whose checkpoint-tag starts with this prefix (e.g. 'v2d-grid').")],
    evals_dir: Annotated[Path, typer.Option(help=f"Parent directory of run outputs (default: {DEFAULT_EVALS_DIR}).")] = DEFAULT_EVALS_DIR,
) -> None:
    summaries = collect_summaries(evals_dir, tag_prefix)
    if not summaries:
        typer.echo(f"No vs-stockfish runs under {evals_dir} with tag prefix {tag_prefix!r}.")
        raise typer.Exit(code=1)
    typer.echo(format_leaderboard(summaries))


if __name__ == "__main__":
    typer.run(main)
```

- [ ] **Step 8: Run the full test file to verify all pass**

Run: `uv run pytest tests/scripts/sweep_summary_test.py -v`
Expected: PASS (3 passed).

- [ ] **Step 9: Run lint + typecheck + full suite**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest`
Expected: all pass. (The PostToolUse hook auto-formats on write; if `ruff format --check` flags this file, run `uv run ruff format .` and re-run.)

- [ ] **Step 10: Run the project verification workflow**

Per `CLAUDE.md` Development Workflow: invoke `/test-and-fix`, then `@agent code-simplifier`, `@agent verify-app`, `@agent build-validator`, `@agent code-architect` (must APPROVE). Address any findings before committing.

- [ ] **Step 11: Commit**

```bash
git add scripts/sweep_summary.py tests/scripts/sweep_summary_test.py
git commit -m "feat(eval): add read-only sweep-summary leaderboard reader

Ranks same-checkpoint vs-stockfish runs that differ only in search config
(score_rate + guardrails + derived override-rate), which the two-checkpoint
compare.py dashboard cannot do. Used to read the tau-k grid sweep.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Stage 1 — run the 30g directional sweep (operational, long-running)

**Files:** none (produces 9 run dirs under `evals/`).

- [ ] **Step 1: Pre-flight checks**

Run: `which stockfish && uv run python -c "import scripts.eval"`
Expected: a stockfish path prints and the import succeeds. (`jspaulsen/halluci-mate-v2d` is fetched from HF on first use.)

- [ ] **Step 2: Checkpoint with the user before launching**

This launches 270 analyzed games (~3× wall time of a non-analyzed run). Confirm the user wants Stage 1 to start, and whether to run it in the background.

- [ ] **Step 3: Run the 9-cell sweep loop**

```bash
for k in 3 5 8; do
  for entry in "0.5:05" "1.0:10" "1.5:15"; do
    tau="${entry%%:*}"; tag="${entry##*:}"
    uv run python scripts/eval.py vs-stockfish \
      --checkpoint jspaulsen/halluci-mate-v2d \
      --checkpoint-tag "v2d-gridk${k}t${tag}" \
      --games 30 \
      --stockfish-skill 5 --stockfish-depth 12 --sf-analyze \
      --temperature 0.0 --halluci-color alternate --max-plies 400 \
      --search --search-k "${k}" --search-margin "${tau}" \
      --search-leaf material-king-safety --search-quiescence --search-qdepth 4
  done
done
```

Expected: 9 new dirs `evals/<timestamp>_v2d-gridk{3,5,8}t{05,10,15}_vs-stockfish`, each with `config.json`, `records.jsonl`, `metrics.json`, `games.pgn`, and a printed per-run summary.

- [ ] **Step 4: Sanity-check completion**

Run: `ls -d evals/*v2d-gridk*_vs-stockfish | wc -l`
Expected: `9`.

---

### Task 3: Stage 1 analysis — rank cells and choose confirm picks (operational)

**Files:** none (read-only).

- [ ] **Step 1: Print the sweep leaderboard**

Run: `uv run python scripts/sweep_summary.py --tag-prefix v2d-grid`
Expected: a 9-row table sorted by `score_rate`, with `k`, `τ`, W/D/L, cpl_med/p95, blunder%, override%, legal%, tactical%.

- [ ] **Step 2: Apply the decision rule and present to the user**

Per the spec: do NOT pick the bare argmax cell (one 30g game ≈ 0.033 score). Read the grid as a surface — rank by `score_rate`, break near-ties with the steadier CPL-median / blunder columns, and identify the best-performing *region*. The `k=3, τ=1.0` anchor cell calibrates the noise floor. Present the leaderboard + a recommended **top 1–2 cells** to the user and get sign-off before spending the 100g budget.

---

### Task 4: Stage 2 — 100g confirm of the chosen cell(s) (operational, long-running)

**Files:** none (produces 1–2 run dirs under `evals/`).

- [ ] **Step 1: Run the confirm cell(s) at 100g**

For each chosen `(K, τ)` (with `TT` = the two-digit margin code, e.g. τ=0.5→`05`), run:

```bash
uv run python scripts/eval.py vs-stockfish \
  --checkpoint jspaulsen/halluci-mate-v2d \
  --checkpoint-tag "v2d-confk${K}t${TT}" \
  --games 100 \
  --stockfish-skill 5 --stockfish-depth 12 --sf-analyze \
  --temperature 0.0 --halluci-color alternate --max-plies 400 \
  --search --search-k "${K}" --search-margin "${τ}" \
  --search-leaf material-king-safety --search-quiescence --search-qdepth 4
```

Expected: one dir `evals/<timestamp>_v2d-confk${K}t${TT}_vs-stockfish` per cell.

- [ ] **Step 2: Read the confirmed numbers**

Run: `uv run python scripts/sweep_summary.py --tag-prefix v2d-conf`
Expected: a 1–2 row table with the confirmed `score_rate` + guardrails.

- [ ] **Step 3: Apply the win condition**

A cell replaces the default only if its 100g `score_rate` beats **0.175** AND it holds the guardrails (CPL median ≤ 14, blunder ≤ 4.34%, legal_rate ~flat) vs the existing baseline run `evals/2026-05-25T03-25-36_v2d-ab-ks-100_vs-stockfish`. If nothing clears it, the documented outcome is "v2 defaults hold" (skip Task 6).

---

### Task 5: Record the result in memory (recording)

**Files:** Modify `/home/jpaulsen/.claude/projects/-home-jpaulsen-repos-halluci-mate/memory/project_inference_search_eval.md`

- [ ] **Step 1: Append a "grid follow-up" section**

Add the observed result: the leaderboard winner (or "defaults hold"), the winning cell's 100g `score_rate` + guardrails, and the new eval dir names (sweep dirs `v2d-gridk*`, confirm dirs `v2d-conf*`). Keep the existing v1/v2 sections intact. (The MEMORY.md pointer already exists; update its hook line if the headline number changes.) These are observed values recorded after the runs — fill them from the actual Task 3/4 output.

---

### Task 6 (CONDITIONAL — only if a cell won in Task 4): bump shipped defaults

**Files:** Modify `scripts/eval.py:104` (`search_k` default) and `scripts/eval.py:106-108` (`search_margin` default); update `docs/superpowers/specs/2026-05-23-search-strength-v2-design.md` default note.

- [ ] **Step 1: Change the CLI defaults to the winning cell**

In `scripts/eval.py`, set the `search_k` default (currently `= 3`) and the `search_margin` default (currently `= 1.0`) to the winning `(K, τ)`. Update the help-text default mentions to match.

- [ ] **Step 2: Update the config-recording test**

`tests/scripts/eval_test.py` asserts the recorded search config. Update any pinned default values there to the new `(K, τ)`.

- [ ] **Step 3: Verify and run the project workflow**

Run: `uv run ruff check . && uv run ty check && uv run pytest`. Then `/test-and-fix` and the `@agent` review chain per `CLAUDE.md`.

- [ ] **Step 4: Commit**

```bash
git add scripts/eval.py tests/scripts/eval_test.py docs/superpowers/specs/2026-05-23-search-strength-v2-design.md
git commit -m "feat(eval): bump default search-k/margin to the grid winner

<one line with the confirmed 100g score_rate vs the prior 0.175 default>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Goal / "no improvement is valid" → Task 4 Step 3 win condition + Task 5 (records "defaults hold").
- Primary metric score_rate + guardrails → CellSummary fields, `format_leaderboard`, Task 4 Step 3.
- Fixed protocol → Task 2 Step 3 / Task 4 Step 1 flags match the spec table exactly.
- 30g grid k∈{3,5,8}×τ∈{0.5,1.0,1.5} + in-sweep anchor → Task 2 loop (includes k3/τ1.0).
- Surface-not-argmax decision rule → Task 3 Step 2.
- 100g confirm + reuse existing baseline → Task 4.
- Read-only summary script reusing eval.compare loaders + derived override-rate → Task 1.
- Tag scheme (no underscores) → `v2d-gridk{K}t{TT}` / `v2d-confk{K}t{TT}`, hyphens only.
- Outputs: memory update (Task 5) + conditional default-bump (Task 6).
- Out of scope (other leaves/quiescence/v1) → not present in any task.

**Placeholder scan:** No "TBD/TODO". The only deferred values are future eval *results* (Task 3/4/5/6), which cannot exist before the runs — flagged as observed-value recording steps, not unwritten code.

**Type consistency:** `override_rate`, `collect_summaries`, `summarize_run`, `format_leaderboard`, `CellSummary`, `_run_tag`, `_fmt`, `_pct` are named identically in the impl and tests. `RunEntry`/`RunReader`/`discover_runs`/`load_or_compute_metrics`/`headline_metrics`/`PerMoveRecord`/`Record`/`Evaluator` match their real signatures in `halluci_mate.eval.*`. Config keys (`search_k`, `search_margin`, `games`) and record fields (`model_move`, `model_top_k[].move`) match the on-disk schema.
