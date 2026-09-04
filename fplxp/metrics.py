"""The frozen metric set from docs/evaluation-plan.md.

RMSE, MAE, and mean per-gameweek Spearman rank correlation, each computed
three ways per position (over all rows; excluding synthetic is_blank
reindexed rows; and restricted to minutes > 0), plus within-gameweek
precision@k / realised-points-of-top-k for k in config.TOP_K_VALUES, plus
a calibration table (actual points by predicted decile). Spearman uses
DataFrame.corr(method="spearman") -- no scipy, per the standard-stack
constraint.

Spearman is averaged only over gameweeks where EVERY predictor being
compared has a defined (non-degenerate) correlation -- see
`_common_valid_gameweeks`. Without this, one predictor (e.g. baseline_xp,
which is a constant value across an entire slate in a couple of
gameweeks -- a real data-quality gap in the source, see
docs/judgment-calls.md) silently drops different gameweeks than the
others, and the averaged numbers are no longer comparing the same thing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fplxp.config import RANDOM_STATE, TOP_K_VALUES

# A single fixed formation for the constrained-squad metric (docs/judgment-calls.md):
# not formation-optimized (that would mean trying every valid (DEF,MID,FWD) split
# each gameweek), just one common, valid FPL shape, so the metric stays cheap and
# auditable while still being budget/formation/club-constrained.
CONSTRAINED_FORMATION = {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}
CONSTRAINED_BUDGET = 1000  # raw `value` units = GBP100.0m
CONSTRAINED_MAX_PER_CLUB = 3

CUTS = {
    "": lambda df: df,
    "_nonblank": lambda df: df[df["is_blank"] == 0],
    "_played": lambda df: df[df["minutes"] > 0],
}


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation via DataFrame.corr (standard-stack only, no scipy)."""
    return pd.DataFrame({"a": a, "b": b}).corr(method="spearman").iloc[0, 1]


def _common_valid_gameweeks(df: pd.DataFrame, pred_cols: list[str], target_col: str) -> set:
    """(season, round) slates where the target AND every pred_col in
    pred_cols has >= 2 distinct values -- i.e. a defined Spearman
    correlation for all of them, so an averaged comparison is over the
    same gameweeks for every predictor."""
    valid = set()
    for key, group in df.groupby(["season", "round"]):
        if group[target_col].nunique() < 2:
            continue
        if all(group[c].nunique() >= 2 for c in pred_cols if c in group.columns):
            valid.add(key)
    return valid


def mean_gw_spearman(df: pd.DataFrame, pred_col: str, target_col: str = "total_points", valid_gameweeks: set | None = None) -> float:
    """Average Spearman correlation of pred_col vs target_col within each (season, round).

    If valid_gameweeks is given, only those slates are used (see
    _common_valid_gameweeks); otherwise every slate where pred_col itself
    is non-degenerate is used (the old, unaligned behaviour -- fine for a
    single predictor in isolation, but not for comparing several).
    """
    corrs = []
    for key, group in df.groupby(["season", "round"]):
        if valid_gameweeks is not None:
            if key not in valid_gameweeks:
                continue
        elif group[pred_col].nunique() < 2 or group[target_col].nunique() < 2:
            continue
        corr = _spearman(group[pred_col], group[target_col])
        if not np.isnan(corr):
            corrs.append(corr)
    return float(np.mean(corrs)) if corrs else float("nan")


def point_metrics_multi(df: pd.DataFrame, pred_cols: list[str], target_col: str = "total_points") -> pd.DataFrame:
    """RMSE/MAE/Spearman for every pred_col, over all rows / non-blank /
    minutes>0, with Spearman aligned to a common gameweek set per cut so
    the predictors are compared on the same slates. One row per pred_col.
    """
    pred_cols = [c for c in pred_cols if c in df.columns]
    rows = {c: {"n_gw_total": len(df.groupby(["season", "round"]))} for c in pred_cols}

    for suffix, cut_fn in CUTS.items():
        cut_df = cut_fn(df)
        valid_gw = _common_valid_gameweeks(cut_df, pred_cols, target_col)
        n_gw_in_cut = cut_df.groupby(["season", "round"]).ngroups
        for c in pred_cols:
            rows[c][f"rmse{suffix}"] = rmse(cut_df[target_col], cut_df[c])
            rows[c][f"mae{suffix}"] = mae(cut_df[target_col], cut_df[c])
            rows[c][f"spearman{suffix}"] = mean_gw_spearman(cut_df, c, target_col, valid_gameweeks=valid_gw)
            rows[c][f"n{suffix}"] = len(cut_df)
            rows[c][f"n_gw_used{suffix}"] = len(valid_gw)
            rows[c][f"n_gw_dropped{suffix}"] = n_gw_in_cut - len(valid_gw)

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "pred_col"
    return out.reset_index()


def point_metrics_by_position(df: pd.DataFrame, pred_cols: list[str], target_col: str = "total_points") -> pd.DataFrame:
    frames = []
    for pos, group in df.groupby("position"):
        m = point_metrics_multi(group, pred_cols, target_col)
        m.insert(0, "position", pos)
        frames.append(m)
    m_all = point_metrics_multi(df, pred_cols, target_col)
    m_all.insert(0, "position", "ALL")
    frames.append(m_all)
    return pd.concat(frames, ignore_index=True)


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


# --------------------------------------------------------------------------
# Constrained top-11: budget + club cap + a fixed valid formation.
#
# The unconstrained top-k metrics above answer "if you could buy any 11
# players with no budget or club limit, how would this predictor's picks
# do" -- which is close to the real decision only for k=1 (captaincy: you
# already own your squad, so picking who to captain really is an
# unconstrained "best predicted score today" choice). For an actual XI,
# FPL enforces a total squad budget, a 3-players-per-club cap, and
# formation quotas, so an unconstrained top-11 systematically overstates
# what's achievable -- it can and does pick five Manchester City players.
# This greedily builds one FEASIBLE XI per gameweek respecting those
# constraints (fixed formation, not formation-optimized -- see
# CONSTRAINED_FORMATION -- and a per-position greedy-by-predicted-value
# fill, not a global optimum -- see docs/judgment-calls.md).
# --------------------------------------------------------------------------

def _greedy_constrained_squad(
    group: pd.DataFrame,
    pred_col: str,
    formation: dict = None,
    budget: int = CONSTRAINED_BUDGET,
    max_per_club: int = CONSTRAINED_MAX_PER_CLUB,
):
    formation = formation or CONSTRAINED_FORMATION
    picked_idx = []
    club_count: dict = {}
    spend = 0
    for pos, n in formation.items():
        pool = group[group["position"] == pos].sort_values(pred_col, ascending=False)
        count = 0
        for row in pool.itertuples():
            if count >= n:
                break
            team = row.team
            value = row.value
            if club_count.get(team, 0) >= max_per_club:
                continue
            if spend + value > budget:
                continue
            picked_idx.append(row.Index)
            club_count[team] = club_count.get(team, 0) + 1
            spend += value
            count += 1
        if count < n:
            return None  # infeasible this gameweek (too few affordable eligible players)
    return picked_idx


def constrained_squad_metrics(df: pd.DataFrame, pred_col: str, target_col: str = "total_points") -> dict:
    """Mean predicted/realised points of one greedily-built, budget+formation
    +club-constrained XI per gameweek, picked by pred_col, scored by target_col."""
    rows = []
    n_total = 0
    for _, group in df.groupby(["season", "round"]):
        n_total += 1
        picked_idx = _greedy_constrained_squad(group, pred_col)
        if picked_idx is None:
            continue
        squad = group.loc[picked_idx]
        rows.append({
            "predicted_points": squad[pred_col].sum(),
            "realised_points": squad[target_col].sum(),
            "spend": squad["value"].sum(),
        })
    per_gw = pd.DataFrame(rows)
    return {
        "mean_realised_points": float(per_gw["realised_points"].mean()) if len(per_gw) else float("nan"),
        "mean_predicted_points": float(per_gw["predicted_points"].mean()) if len(per_gw) else float("nan"),
        "mean_spend": float(per_gw["spend"].mean()) if len(per_gw) else float("nan"),
        "n_gameweeks_feasible": len(per_gw),
        "n_gameweeks_total": n_total,
    }


def constrained_squad_report(df: pd.DataFrame, pred_cols: list[str], target_col: str = "total_points") -> pd.DataFrame:
    rows = []
    for col in pred_cols:
        if col not in df.columns:
            continue
        rows.append({"pred_col": col, **constrained_squad_metrics(df, col, target_col)})
    rows.append({
        "pred_col": "__oracle__",
        **constrained_squad_metrics(df, target_col, target_col),
    })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Bootstrap CI on a paired per-gameweek metric difference (model vs. a
# baseline). A single frozen test-set point estimate (e.g. "RMSE 2.926 vs
# 3.068") is uninterpretable on its own -- resampling gameweeks gives a
# standard error on the *difference*, so a report can say whether a gap is
# distinguishable from noise rather than just stating the raw numbers.
# --------------------------------------------------------------------------

def per_gameweek_metric(df: pd.DataFrame, pred_col: str, target_col: str, metric_fn) -> pd.DataFrame:
    """metric_fn(y_true, y_pred) -> float, applied within each (season, round) slate."""
    rows = []
    for (season, rnd), group in df.groupby(["season", "round"]):
        if len(group) < 2:
            continue
        rows.append({"season": season, "round": rnd, "value": metric_fn(group[target_col], group[pred_col])})
    return pd.DataFrame(rows)


def bootstrap_paired_diff(
    df: pd.DataFrame,
    pred_col_a: str,
    pred_col_b: str,
    target_col: str,
    metric_fn,
    n_boot: int = 2000,
    seed: int = RANDOM_STATE,
) -> dict:
    """Bootstrap SE/CI (over gameweeks) for mean(metric(a) - metric(b)),
    paired by gameweek. Positive mean_diff means a scored higher on
    metric_fn than b (e.g. for RMSE, a is WORSE; for Spearman, a is BETTER
    -- the caller interprets sign, this just resamples gameweeks)."""
    a = per_gameweek_metric(df, pred_col_a, target_col, metric_fn).rename(columns={"value": "a"})
    b = per_gameweek_metric(df, pred_col_b, target_col, metric_fn).rename(columns={"value": "b"})
    paired = a.merge(b, on=["season", "round"], how="inner").dropna()

    diffs = (paired["a"] - paired["b"]).to_numpy()
    n = len(diffs)
    if n == 0:
        return {"mean_diff": float("nan"), "se": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_gameweeks": 0}

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[boot_idx].mean(axis=1)
    return {
        "mean_diff": float(diffs.mean()),
        "se": float(boot_means.std(ddof=1)),
        "ci_lo": float(np.percentile(boot_means, 2.5)),
        "ci_hi": float(np.percentile(boot_means, 97.5)),
        "n_gameweeks": n,
    }
