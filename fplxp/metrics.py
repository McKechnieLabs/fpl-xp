"""RMSE, MAE, and per-gameweek Spearman rank correlation, overall and by position.

Spearman is computed within each (season, round) slate separately and then
averaged, because what matters for team selection is whether a model ranks
players correctly *within a gameweek*, not whether its errors correlate
across the whole pooled sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def mean_gw_spearman(df: pd.DataFrame, pred_col: str, target_col: str = "total_points") -> float:
    """Average Spearman correlation of pred_col vs target_col within each (season, round)."""
    corrs = []
    for _, group in df.groupby(["season", "round"]):
        if group[pred_col].nunique() < 2 or group[target_col].nunique() < 2:
            continue
        corr, _ = spearmanr(group[pred_col], group[target_col])
        if not np.isnan(corr):
            corrs.append(corr)
    return float(np.mean(corrs)) if corrs else float("nan")


def evaluate(df: pd.DataFrame, pred_col: str, target_col: str = "total_points") -> dict:
    return {
        "rmse": rmse(df[target_col], df[pred_col]),
        "mae": mae(df[target_col], df[pred_col]),
        "spearman": mean_gw_spearman(df, pred_col, target_col),
        "n": len(df),
    }


def evaluate_by_position(df: pd.DataFrame, pred_col: str, target_col: str = "total_points") -> pd.DataFrame:
    rows = []
    for pos, group in df.groupby("position"):
        row = {"position": pos}
        row.update(evaluate(group, pred_col, target_col))
        rows.append(row)
    overall = {"position": "ALL"}
    overall.update(evaluate(df, pred_col, target_col))
    rows.append(overall)
    return pd.DataFrame(rows)
