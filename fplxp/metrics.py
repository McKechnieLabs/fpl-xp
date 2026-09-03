"""The frozen metric set from docs/evaluation-plan.md.

RMSE, MAE, and mean per-gameweek Spearman rank correlation, each computed
twice per position (over all rows, and restricted to minutes > 0), plus
within-gameweek precision@k / realised-points-of-top-k for k in
config.TOP_K_VALUES, plus a calibration table (actual points by predicted
decile). Spearman uses DataFrame.corr(method="spearman") -- no scipy, per
the standard-stack constraint.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fplxp.config import TOP_K_VALUES


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation via DataFrame.corr (standard-stack only, no scipy)."""
    return pd.DataFrame({"a": a, "b": b}).corr(method="spearman").iloc[0, 1]


def mean_gw_spearman(df: pd.DataFrame, pred_col: str, target_col: str = "total_points") -> float:
    """Average Spearman correlation of pred_col vs target_col within each (season, round)."""
    corrs = []
    for _, group in df.groupby(["season", "round"]):
        if group[pred_col].nunique() < 2 or group[target_col].nunique() < 2:
            continue
        corr = _spearman(group[pred_col], group[target_col])
        if not np.isnan(corr):
            corrs.append(corr)
    return float(np.mean(corrs)) if corrs else float("nan")


def point_metrics(df: pd.DataFrame, pred_col: str, target_col: str = "total_points") -> dict:
    """RMSE/MAE/Spearman, both over all rows and restricted to minutes > 0."""
    played = df[df["minutes"] > 0]
    return {
        "rmse": rmse(df[target_col], df[pred_col]),
        "mae": mae(df[target_col], df[pred_col]),
        "spearman": mean_gw_spearman(df, pred_col, target_col),
        "rmse_played": rmse(played[target_col], played[pred_col]) if len(played) else float("nan"),
        "mae_played": mae(played[target_col], played[pred_col]) if len(played) else float("nan"),
        "spearman_played": mean_gw_spearman(played, pred_col, target_col) if len(played) else float("nan"),
        "n": len(df),
        "n_played": len(played),
    }


def point_metrics_by_position(df: pd.DataFrame, pred_col: str, target_col: str = "total_points") -> pd.DataFrame:
    rows = []
    for pos, group in df.groupby("position"):
        rows.append({"position": pos, **point_metrics(group, pred_col, target_col)})
    rows.append({"position": "ALL", **point_metrics(df, pred_col, target_col)})
    return pd.DataFrame(rows)


def _gw_topk(group: pd.DataFrame, pred_col: str, target_col: str, k: int) -> tuple[float, float]:
    """(precision@k, realised points of top-k) for one gameweek slate."""
    k = min(k, len(group))
    if k == 0:
        return float("nan"), float("nan")
    picked = group.nlargest(k, pred_col)
    actual_top = set(group.nlargest(k, target_col).index)
    precision = len(set(picked.index) & actual_top) / k
    realised = picked[target_col].sum()
    return precision, realised


def topk_metrics(df: pd.DataFrame, pred_col: str, target_col: str = "total_points") -> pd.DataFrame:
    """Precision@k and realised-points-of-top-k, per k, averaged over gameweeks."""
    rows = []
    for k in TOP_K_VALUES:
        precisions, realised = [], []
        for _, group in df.groupby(["season", "round"]):
            p, r = _gw_topk(group, pred_col, target_col, k)
            if not np.isnan(p):
                precisions.append(p)
                realised.append(r)
        rows.append({
            "k": k,
            "precision_at_k": float(np.mean(precisions)) if precisions else float("nan"),
            "realised_points_of_topk": float(np.mean(realised)) if realised else float("nan"),
        })
    return pd.DataFrame(rows)


def oracle_topk_metrics(df: pd.DataFrame, target_col: str = "total_points") -> pd.DataFrame:
    """The theoretical ceiling: top-k picked with perfect hindsight each gameweek."""
    return topk_metrics(df, pred_col=target_col, target_col=target_col).assign(
        precision_at_k=1.0
    )


def calibration_table(df: pd.DataFrame, pred_col: str, target_col: str = "total_points", n_bins: int = 10) -> pd.DataFrame:
    """Mean actual points by predicted-points decile, per position."""
    rows = []
    for pos, group in df.groupby("position"):
        try:
            decile = pd.qcut(group[pred_col], n_bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        g = group.assign(decile=decile).groupby("decile").agg(
            mean_predicted=(pred_col, "mean"),
            mean_actual=(target_col, "mean"),
            n=(target_col, "size"),
        ).reset_index()
        g.insert(0, "position", pos)
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
