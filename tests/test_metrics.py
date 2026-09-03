import pandas as pd
import pytest

from fplxp.metrics import (
    calibration_table,
    mae,
    mean_gw_spearman,
    oracle_topk_metrics,
    point_metrics,
    point_metrics_by_position,
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
            "minutes": [90, 0, 90, 0],
            "pred": [5, 5, 3, 0.5],
            "total_points": [6, 0, 2, 0],
        }
    )


def test_point_metrics_splits_all_rows_vs_played_only():
    df = _minutes_df()
    m = point_metrics(df, "pred")
    assert m["n"] == 4
    assert m["n_played"] == 2
    # played-only RMSE computed on just the two minutes>0 rows
    played_only_rmse = rmse(df[df["minutes"] > 0]["total_points"], df[df["minutes"] > 0]["pred"])
    assert m["rmse_played"] == pytest.approx(played_only_rmse)


def test_point_metrics_by_position_includes_all_row():
    out = point_metrics_by_position(_minutes_df(), "pred")
    assert set(out["position"]) == {"MID", "ALL"}


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


def test_calibration_table_shape():
    import numpy as np

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
