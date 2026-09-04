import numpy as np
import pandas as pd
import pytest

from fplxp.metrics import (
    _common_valid_gameweeks,
    _greedy_constrained_squad,
    bootstrap_paired_diff,
    calibration_table,
    constrained_squad_metrics,
    mae,
    mean_gw_spearman,
    oracle_topk_metrics,
    per_gameweek_metric,
    point_metrics_by_position,
    point_metrics_multi,
    rmse,
    topk_metrics,
)


def test_rmse_mae_known_values():
    y_true = [1, 2, 3, 4]
    y_pred = [1, 2, 3, 6]  # one error of 2
    assert mae(y_true, y_pred) == pytest.approx(0.5)
    assert rmse(y_true, y_pred) == pytest.approx((4 / 4) ** 0.5)


def test_mean_gw_spearman_perfect_rank_within_gameweek():
    df = pd.DataFrame(
        {
            "season": ["s1"] * 8,
            "round": [1, 1, 1, 1, 2, 2, 2, 2],
            "pred": [1, 2, 3, 4, 4, 3, 2, 1],
            "total_points": [1, 2, 3, 4, 4, 3, 2, 1],
        }
    )
    assert mean_gw_spearman(df, "pred") == pytest.approx(1.0)


def test_mean_gw_spearman_ignores_degenerate_slates():
    df = pd.DataFrame(
        {
            "season": ["s1", "s1", "s2", "s2"],
            "round": [1, 1, 1, 1],
            "pred": [5, 5, 1, 2],  # s1: constant prediction -> undefined, skipped
            "total_points": [1, 2, 1, 2],
        }
    )
    assert mean_gw_spearman(df, "pred") == pytest.approx(1.0)


def _minutes_df():
    return pd.DataFrame(
        {
            "season": ["s1"] * 4,
            "round": [1, 1, 1, 1],
            "position": ["MID", "MID", "MID", "MID"],
            "is_blank": [0, 0, 1, 0],
            "minutes": [90, 0, 0, 0],
            "pred": [5, 5, 3, 0.5],
            "total_points": [6, 0, 0, 0],
        }
    )


def test_point_metrics_multi_splits_all_vs_nonblank_vs_played():
    df = _minutes_df()
    m = point_metrics_multi(df, ["pred"]).set_index("pred_col").loc["pred"]
    assert m["n"] == 4
    assert m["n_nonblank"] == 3  # one is_blank row excluded
    assert m["n_played"] == 1
    played_only_rmse = rmse(df[df["minutes"] > 0]["total_points"], df[df["minutes"] > 0]["pred"])
    assert m["rmse_played"] == pytest.approx(played_only_rmse)


def test_point_metrics_by_position_includes_all_row():
    out = point_metrics_by_position(_minutes_df(), ["pred"])
    assert set(out["position"]) == {"MID", "ALL"}


def test_common_valid_gameweeks_excludes_slate_where_any_predictor_is_degenerate():
    df = pd.DataFrame(
        {
            "season": ["s1"] * 4 + ["s2"] * 4,
            "round": [1] * 4 + [2] * 4,
            "a": [1, 2, 3, 4, 1, 2, 3, 4],
            "b": [5, 5, 5, 5, 4, 3, 2, 1],  # constant on gw1's slate -> degenerate for b
            "total_points": [1, 2, 3, 4, 4, 3, 2, 1],
        }
    )
    valid = _common_valid_gameweeks(df, ["a", "b"], "total_points")
    assert valid == {("s2", 2)}


def test_point_metrics_multi_aligns_spearman_across_predictors():
    """A predictor degenerate on one slate must not silently change which
    gameweeks another predictor's average is computed over."""
    df = pd.DataFrame(
        {
            "season": ["s1"] * 4 + ["s2"] * 4,
            "round": [1] * 4 + [2] * 4,
            "is_blank": [0] * 8,
            "minutes": [90] * 8,
            "good": [1, 2, 3, 4, 4, 3, 2, 1],  # perfect rank both gameweeks
            "flat": [5, 5, 5, 5, 4, 3, 2, 1],  # degenerate on gw1
            "total_points": [1, 2, 3, 4, 4, 3, 2, 1],
        }
    )
    out = point_metrics_multi(df, ["good", "flat"]).set_index("pred_col")
    # Only gw2 is common to both -> "good"'s average must also drop gw1,
    # even though "good" alone is well-defined on both gameweeks.
    assert out.loc["good", "n_gw_used"] == 1
    assert out.loc["flat", "n_gw_used"] == 1
    assert out.loc["good", "spearman"] == pytest.approx(1.0)
    assert out.loc["good", "n_gw_dropped"] == 1


def test_topk_precision_and_realised_points():
    df = pd.DataFrame(
        {
            "season": ["s1"] * 4,
            "round": [1, 1, 1, 1],
            "pred": [10, 8, 1, 0],
            "total_points": [5, 9, 2, 1],
        }
    )
    out = topk_metrics(df, "pred").set_index("k")
    # top-1 by pred is the first row (pred=10, actual=5); top-1 by actual is row 2 (actual=9)
    assert out.loc[1, "precision_at_k"] == pytest.approx(0.0)
    assert out.loc[1, "realised_points_of_topk"] == pytest.approx(5.0)


def test_oracle_topk_has_perfect_precision():
    df = pd.DataFrame(
        {
            "season": ["s1"] * 4,
            "round": [1, 1, 1, 1],
            "total_points": [5, 9, 2, 1],
        }
    )
    out = oracle_topk_metrics(df).set_index("k")
    assert (out["precision_at_k"] == 1.0).all()
    assert out.loc[1, "realised_points_of_topk"] == pytest.approx(9.0)


def _squad_pool():
    """4 clubs, enough depth per position for a 1-4-4-2 XI, budget-tight."""
    rows = []
    rid = 0
    for pos, n_per_club in [("GK", 2), ("DEF", 3), ("MID", 3), ("FWD", 2)]:
        for club in ["A", "B", "C", "D"]:
            for i in range(n_per_club):
                rows.append({
                    "position": pos, "team": club, "value": 50 + i * 10,
                    "pred": 5.0 - i, "total_points": 4.0 - i, "id": rid,
                })
                rid += 1
    df = pd.DataFrame(rows)
    df.index = range(len(df))
    return df


def test_greedy_constrained_squad_respects_club_cap():
    group = _squad_pool()
    # Make club A's DEF candidates dominate on predicted value.
    is_def_a = (group["position"] == "DEF") & (group["team"] == "A")
    group.loc[is_def_a, "pred"] = 100
    picked_idx = _greedy_constrained_squad(group, "pred", formation={"GK": 1, "DEF": 4, "MID": 4, "FWD": 2})
    assert picked_idx is not None
    picked = group.loc[picked_idx]
    def_counts = picked[picked["position"] == "DEF"]["team"].value_counts()
    assert def_counts.get("A", 0) <= 3  # club cap enforced even though A's DEFs rank highest


def test_greedy_constrained_squad_respects_budget():
    group = _squad_pool()
    group.loc[group["position"] == "GK", "value"] = 2000  # unaffordable
    picked_idx = _greedy_constrained_squad(group, "pred", budget=1000)
    assert picked_idx is None  # infeasible: no affordable GK


def test_constrained_squad_metrics_oracle_matches_pred_when_pred_is_target():
    df = _squad_pool()
    df["season"] = "s1"
    df["round"] = 1
    out = constrained_squad_metrics(df, "total_points", "total_points")
    assert out["mean_predicted_points"] == pytest.approx(out["mean_realised_points"])
    assert out["n_gameweeks_feasible"] == 1


def test_bootstrap_paired_diff_recovers_constant_gap():
    # 20 gameweeks, model's RMSE is always exactly 1.0 better than baseline's.
    rows = []
    for gw in range(20):
        rows.append({"season": "s1", "round": gw, "total_points": 5, "model": 5.0, "baseline": 6.0})
        rows.append({"season": "s1", "round": gw, "total_points": 3, "model": 3.0, "baseline": 4.0})
    df = pd.DataFrame(rows)
    out = bootstrap_paired_diff(df, "model", "baseline", "total_points", rmse, n_boot=500, seed=0)
    assert out["mean_diff"] == pytest.approx(-1.0, abs=1e-6)  # model's RMSE is 1.0 lower (better)
    assert out["ci_lo"] <= -1.0 <= out["ci_hi"]
    assert out["n_gameweeks"] == 20


def test_calibration_table_shape():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "position": ["MID"] * 50 + ["DEF"] * 50,
            "pred": rng.normal(size=100),
            "total_points": rng.normal(size=100),
        }
    )
    out = calibration_table(df, "pred")
    assert set(out["position"]) == {"MID", "DEF"}
    assert {"mean_predicted", "mean_actual", "n"}.issubset(out.columns)
