"""Phase 3: compare model variants on the validation set ONLY (2023-24).

Runs the four variants from docs/evaluation-plan.md, applies the
pre-committed selection rule (also in that doc) mechanically, writes
reports/validation_comparison.md, and records the winner to
reports/chosen_model.json for fplxp.backtest to pick up.

The test set (2024-25, 2025-26) is never loaded in this module.
"""
from __future__ import annotations

import json

import pandas as pd

from fplxp.config import RANDOM_STATE, REPORTS_DIR, TRAIN_SEASONS, VAL_SEASONS
from fplxp.evaluate import add_predictions, full_report
from fplxp.features import build_feature_table
from fplxp.model import PooledModel, PositionModels, TwoStageModel
from fplxp.report_utils import df_to_markdown

VARIANTS = ["gbr", "hgb_poisson", "two_stage", "pooled"]
SIMPLICITY_ORDER = {"gbr": 0, "hgb_poisson": 1, "two_stage": 2, "pooled": 3}  # tiebreak: simpler wins


def _fit(kind: str, train_df: pd.DataFrame):
    if kind == "two_stage":
        return TwoStageModel().fit(train_df)
    if kind == "pooled":
        return PooledModel().fit(train_df)
    return PositionModels(estimator_kind=kind).fit(train_df)


def compare_variants(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    fill_value = train_df["total_points"].mean()
    reports = {}
    rows = []
    for kind in VARIANTS:
        model = _fit(kind, train_df)
        scored = add_predictions(val_df, model, fill_value)
        report = full_report(scored, pred_cols=["model"])
        reports[kind] = report

        played = scored[scored["minutes"] > 0]
        pm = report["point_metrics"]
        all_row = pm[pm["position"] == "ALL"].iloc[0]
        top11 = report["topk_metrics"]
        top11_row = top11[(top11["method"] == "Model") & (top11["k"] == 11)].iloc[0]
        rows.append({
            "variant": kind,
            "rmse_played": all_row["rmse_played"],
            "mae_played": all_row["mae_played"],
            "spearman_played": all_row["spearman_played"],
            "realised_points_of_top11": top11_row["realised_points_of_topk"],
        })
    summary = pd.DataFrame(rows).set_index("variant")

    ranks = pd.DataFrame(index=summary.index)
    ranks["rmse_played"] = summary["rmse_played"].rank(ascending=True)
    ranks["mae_played"] = summary["mae_played"].rank(ascending=True)
    ranks["spearman_played"] = summary["spearman_played"].rank(ascending=False)
    ranks["realised_points_of_top11"] = summary["realised_points_of_top11"].rank(ascending=False)
    summary["mean_rank"] = ranks.mean(axis=1)

    best_rank = summary["mean_rank"].min()
    tied = summary[summary["mean_rank"] <= best_rank + 0.1].index.tolist()
    winner = min(tied, key=lambda k: SIMPLICITY_ORDER[k])

    return {"reports": reports, "summary": summary, "winner": winner}, summary


def write_validation_report(result: dict) -> None:
    summary = result["summary"].round(4).reset_index()
    lines = [
        "# Phase 3: model variant comparison (validation season 2023-24 only)\n",
        "Selection rule fixed in docs/evaluation-plan.md *before* this was run: "
        "rank each variant on {RMSE, MAE, Spearman, realised-points-of-top-11} "
        "(minutes > 0 rows, all positions pooled), average the ranks, lowest "
        "average wins; ties broken toward the simpler model.\n",
        df_to_markdown(summary),
        "",
        f"**Selected variant: `{result['winner']}`** (see reports/chosen_model.json).",
        "",
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "validation_comparison.md").write_text("\n".join(lines))


def main():
    print(f"Building feature table (train={TRAIN_SEASONS}, val={VAL_SEASONS})...")
    table = build_feature_table(TRAIN_SEASONS + VAL_SEASONS)
    train_df = table[table["season"].isin(TRAIN_SEASONS)].copy()
    val_df = table[table["season"].isin(VAL_SEASONS)].copy()
    print(f"train rows={len(train_df)}, val rows={len(val_df)}")

    result, summary = compare_variants(train_df, val_df)
    print(summary.round(4).to_string())
    print(f"Selected variant: {result['winner']}")

    write_validation_report(result)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "chosen_model.json").write_text(
        json.dumps({"winner": result["winner"], "random_state": RANDOM_STATE}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
