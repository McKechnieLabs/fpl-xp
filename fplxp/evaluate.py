"""Shared scoring logic used by both the Phase 3 validation comparison and
the Phase 4 frozen test run, so the two can't drift apart."""
from __future__ import annotations

import pandas as pd

from fplxp.baselines import last5_mean_baseline, season_to_date_baseline, xp_baseline
from fplxp.metrics import (
    CONSTRAINED_FORMATION,
    calibration_table,
    constrained_squad_report,
    oracle_topk_metrics,
    point_metrics_by_position,
    topk_metrics,
)

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
    """Every metric in docs/evaluation-plan.md for the given predictions frame.

    Spearman across the returned pred_cols is aligned to a common
    gameweek set per docs/judgment-calls.md (a predictor with a
    degenerate slate -- e.g. baseline_xp on a gameweek where xP is a
    single constant value -- would otherwise silently average over a
    different set of gameweeks than the others).
    """
    pred_cols = pred_cols or list(BASELINE_LABELS.keys())
    pred_cols = [c for c in pred_cols if c in df.columns]

    point = point_metrics_by_position(df, pred_cols)
    point["method"] = point["pred_col"].map(BASELINE_LABELS).fillna(point["pred_col"])
    point = point.drop(columns=["pred_col"])

    topk = []
    for col in pred_cols:
        tk = topk_metrics(df, col)
        tk.insert(0, "method", BASELINE_LABELS.get(col, col))
        topk.append(tk)

    oracle = oracle_topk_metrics(df)
    oracle.insert(0, "method", "Oracle (perfect hindsight)")
    topk.append(oracle)

    calibration = calibration_table(df, "model") if "model" in df.columns else pd.DataFrame()

    squad = constrained_squad_report(df, pred_cols)
    squad["method"] = squad["pred_col"].replace({"__oracle__": "Oracle (perfect hindsight)"})
    squad["method"] = squad["method"].map(BASELINE_LABELS).fillna(squad["method"])
    squad = squad.drop(columns=["pred_col"])

    formation_str = "-".join(f"{n}{pos}" for pos, n in CONSTRAINED_FORMATION.items())

    return {
        "point_metrics": point,
        "topk_metrics": pd.concat(topk, ignore_index=True) if topk else pd.DataFrame(),
        "calibration": calibration,
        "constrained_squad": squad,
        "constrained_squad_formation": formation_str,
    }
