"""Shared scoring logic used by both the Phase 3 validation comparison and
the Phase 4 frozen test run, so the two can't drift apart."""
from __future__ import annotations

import pandas as pd

from fplxp.baselines import last5_mean_baseline, season_to_date_baseline, xp_baseline
from fplxp.metrics import calibration_table, oracle_topk_metrics, point_metrics_by_position, topk_metrics

BASELINE_LABELS = {
    "model": "Model",
    "baseline_last5": "Baseline: last-5 mean",
    "baseline_season": "Baseline: season-to-date mean",
    "baseline_xp": "Baseline: FPL xP",
}


def add_predictions(df: pd.DataFrame, model, fill_value: float) -> pd.DataFrame:
    d = df.copy()
    d["model"] = model.predict(d)
    d["baseline_last5"] = last5_mean_baseline(d, fill_value)
    d["baseline_season"] = season_to_date_baseline(d, fill_value)
    d["baseline_xp"] = xp_baseline(d)
    return d


def full_report(df: pd.DataFrame, pred_cols: list[str] = None) -> dict:
    """Every metric in docs/evaluation-plan.md for the given predictions frame."""
    pred_cols = pred_cols or list(BASELINE_LABELS.keys())
    point = []
    topk = []
    for col in pred_cols:
        if col not in df.columns:
            continue
        pm = point_metrics_by_position(df, col)
        pm.insert(0, "method", BASELINE_LABELS.get(col, col))
        point.append(pm)

        tk = topk_metrics(df, col)
        tk.insert(0, "method", BASELINE_LABELS.get(col, col))
        topk.append(tk)

    oracle = oracle_topk_metrics(df)
    oracle.insert(0, "method", "Oracle (perfect hindsight)")
    topk.append(oracle)

    calibration = calibration_table(df, "model") if "model" in df.columns else pd.DataFrame()

    return {
        "point_metrics": pd.concat(point, ignore_index=True) if point else pd.DataFrame(),
        "topk_metrics": pd.concat(topk, ignore_index=True) if topk else pd.DataFrame(),
        "calibration": calibration,
    }
