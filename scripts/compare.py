"""Streamlit dashboard for comparing two models' eval runs side by side.

Launch with::

    uv run streamlit run scripts/compare.py

The app scans ``evals/``, lists every checkpoint that has at least one run, and
lets the user pick two of them. For each of vs-stockfish, legal-rate, and
perplexity the user picks one run per model (defaulting to the most recent),
then the section renders a small metrics table + a headline bar chart. The
picker matters when the same checkpoint has been evaluated against multiple
input sets (e.g. perplexity on general blitz vs. a high-elo split).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from halluci_mate.eval.compare import (
    COMPARED_EVALUATORS,
    RunEntry,
    discover_runs,
    list_checkpoints,
    load_or_compute_metrics,
    runs_for,
)
from halluci_mate.eval.records import Evaluator

DEFAULT_EVALS_DIR = Path("evals")


def main() -> None:
    st.set_page_config(page_title="halluci-mate eval compare", layout="wide")
    st.title("halluci-mate · eval comparison")

    evals_dir = Path(st.sidebar.text_input("Evals directory", value=str(DEFAULT_EVALS_DIR)))
    runs = discover_runs(evals_dir)
    if not runs:
        st.warning(f"No readable runs under `{evals_dir}`. Run an evaluator first.")
        return

    checkpoints = list_checkpoints(runs)
    if len(checkpoints) < 2:
        st.warning(f"Need at least two distinct checkpoints; found {len(checkpoints)} under `{evals_dir}`.")
        return

    model_a = st.sidebar.selectbox("Model A", checkpoints, index=0, key="model_a")
    default_b_index = 1 if checkpoints[1] != model_a else 0
    model_b = st.sidebar.selectbox("Model B", checkpoints, index=default_b_index, key="model_b")
    if model_a == model_b:
        st.info("Pick two different checkpoints to see a comparison.")
        return

    for evaluator in COMPARED_EVALUATORS:
        _render_evaluator_section(runs, evaluator, model_a, model_b)


def _render_evaluator_section(runs: list[RunEntry], evaluator: Evaluator, model_a: str, model_b: str) -> None:
    st.header(evaluator.value)
    runs_a = runs_for(runs, model_a, evaluator)
    runs_b = runs_for(runs, model_b, evaluator)
    if not runs_a and not runs_b:
        st.caption("No runs for either model.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        entry_a = _run_picker(runs_a, label="Run for Model A", key=f"run_a_{evaluator.value}")
    with col_b:
        entry_b = _run_picker(runs_b, label="Run for Model B", key=f"run_b_{evaluator.value}")

    metrics_a = load_or_compute_metrics(entry_a) if entry_a else {}
    metrics_b = load_or_compute_metrics(entry_b) if entry_b else {}
    flat_a = _headline_metrics(evaluator, metrics_a)
    flat_b = _headline_metrics(evaluator, metrics_b)

    table = _metrics_table(flat_a, flat_b, model_a, model_b)
    st.dataframe(table, use_container_width=True)

    chart = _evaluator_chart(evaluator, flat_a, flat_b, model_a, model_b)
    if chart is not None:
        st.bar_chart(chart)


def _run_picker(runs: list[RunEntry], *, label: str, key: str) -> RunEntry | None:
    """Render a selectbox of run-ids for one (model, evaluator) pair.

    Single-run cases render as a caption rather than a one-option selectbox —
    the dropdown affordance would imply choice where there is none, and the
    caption keeps the layout consistent with the no-run branch above.
    """
    if not runs:
        st.caption("(no run)")
        return None
    if len(runs) == 1:
        st.caption(runs[0].run_id)
        return runs[0]
    indices = list(range(len(runs)))
    chosen = st.selectbox(label, indices, index=0, format_func=lambda i: runs[i].run_id, key=key)
    return runs[chosen]


def _headline_metrics(evaluator: Evaluator, metrics: dict[str, Any]) -> dict[str, float]:
    """Flatten the evaluator's nested ``metrics.json`` payload to scalar headlines.

    Anything missing in ``metrics`` (e.g. CPL when ``--sf-analyze`` was off) is
    omitted rather than zero-filled — leaving the row blank is more honest than
    implying "the model scored 0".
    """
    if not metrics:
        return {}
    if evaluator is Evaluator.VS_STOCKFISH:
        return _vs_stockfish_headlines(metrics)
    if evaluator is Evaluator.LEGAL_RATE:
        overall = metrics.get("legal_rate", {}).get("overall", {})
        return _filter_numeric({"legal_rate": overall.get("rate"), "n": overall.get("n"), "legal": overall.get("legal")})
    if evaluator is Evaluator.PERPLEXITY:
        return _filter_numeric(
            {
                "perplexity": metrics.get("perplexity"),
                "mean_nll": metrics.get("mean_nll"),
                "bits_per_token": metrics.get("bits_per_token"),
                "num_tokens": metrics.get("num_tokens"),
                "num_sequences": metrics.get("num_sequences"),
            }
        )
    return {}


def _vs_stockfish_headlines(metrics: dict[str, Any]) -> dict[str, float]:
    win_overall = metrics.get("win_rate", {}).get("overall", {})
    legal_overall = metrics.get("legal_rate", {}).get("overall", {})
    cpl_overall = metrics.get("centipawn_loss", {}).get("overall", {}) if "centipawn_loss" in metrics else {}
    blunder_overall = metrics.get("blunder_rate", {}).get("overall", {}) if "blunder_rate" in metrics else {}
    return _filter_numeric(
        {
            "win_rate": win_overall.get("win_rate"),
            "score_rate": win_overall.get("score_rate"),
            "wins": win_overall.get("wins"),
            "draws": win_overall.get("draws"),
            "losses": win_overall.get("losses"),
            "unfinished": win_overall.get("unfinished"),
            "legal_rate": legal_overall.get("rate"),
            "cpl_mean": cpl_overall.get("mean"),
            "cpl_median": cpl_overall.get("median"),
            "cpl_p95": cpl_overall.get("p95"),
            "blunder_rate": blunder_overall.get("rate"),
        }
    )


def _filter_numeric(values: dict[str, Any]) -> dict[str, float]:
    return {k: float(v) for k, v in values.items() if isinstance(v, (int, float))}


def _metrics_table(flat_a: dict[str, float], flat_b: dict[str, float], model_a: str, model_b: str) -> pd.DataFrame:
    keys = list(dict.fromkeys([*flat_a.keys(), *flat_b.keys()]))
    rows = []
    for key in keys:
        a = flat_a.get(key)
        b = flat_b.get(key)
        delta = b - a if (a is not None and b is not None) else None
        rows.append({"metric": key, model_a: a, model_b: b, "delta (B - A)": delta})
    return pd.DataFrame(rows).set_index("metric")


def _evaluator_chart(
    evaluator: Evaluator,
    flat_a: dict[str, float],
    flat_b: dict[str, float],
    model_a: str,
    model_b: str,
) -> pd.DataFrame | None:
    chart_keys = _chart_keys(evaluator, flat_a, flat_b)
    if not chart_keys:
        return None
    return pd.DataFrame(
        {model_a: [flat_a.get(k, 0.0) for k in chart_keys], model_b: [flat_b.get(k, 0.0) for k in chart_keys]},
        index=pd.Index(chart_keys),
    )


def _chart_keys(evaluator: Evaluator, flat_a: dict[str, float], flat_b: dict[str, float]) -> list[str]:
    candidates_by_evaluator = {
        Evaluator.VS_STOCKFISH: ["wins", "draws", "losses", "unfinished"],
        Evaluator.LEGAL_RATE: ["legal_rate"],
        Evaluator.PERPLEXITY: ["perplexity"],
    }
    candidates = candidates_by_evaluator.get(evaluator, [])
    return [k for k in candidates if k in flat_a or k in flat_b]


if __name__ == "__main__":
    main()
