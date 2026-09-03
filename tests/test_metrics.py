import pandas as pd
import pytest

from fplxp.metrics import evaluate, evaluate_by_position, mae, mean_gw_spearman, rmse


def test_rmse_mae_known_values():
    y_true = [1, 2, 3, 4]
    y_pred = [1, 2, 3, 6]  # one error of 2
    assert mae(y_true, y_pred) == pytest.approx(0.5)
    assert rmse(y_true, y_pred) == pytest.approx((4 / 4) ** 0.5)


def test_mean_gw_spearman_perfect_rank_within_gameweek():
    df = pd.DataFrame(
        {
            "season": ["s1"] * 4 + ["s1"] * 4,
            "round": [1, 1, 1, 1, 2, 2, 2, 2],
            "pred": [1, 2, 3, 4, 4, 3, 2, 1],
            "total_points": [1, 2, 3, 4, 4, 3, 2, 1],
        }
    )
    corr = mean_gw_spearman(df, "pred")
    assert corr == pytest.approx(1.0)


def test_mean_gw_spearman_ignores_degenerate_slates():
    # Round 1: predictions are constant -> undefined correlation, must be skipped.
    df = pd.DataFrame(
        {
            "season": ["s1", "s1", "s2", "s2"],
            "round": [1, 1, 1, 1],
            "pred": [5, 5, 1, 2],
            "total_points": [1, 2, 1, 2],
        }
    )
    corr = mean_gw_spearman(df, "pred")
    assert corr == pytest.approx(1.0)  # only the s2 slate contributes


def test_evaluate_by_position_includes_overall_row():
    df = pd.DataFrame(
        {
            "season": ["s1"] * 4,
            "round": [1, 1, 1, 1],
            "position": ["GK", "GK", "DEF", "DEF"],
            "pred": [1, 2, 3, 4],
            "total_points": [1, 2, 3, 4],
        }
    )
    out = evaluate_by_position(df, "pred")
    assert set(out["position"]) == {"GK", "DEF", "ALL"}
    assert (out["rmse"] == 0).all()
