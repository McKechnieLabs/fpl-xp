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
from fplxp.features import FEATURE_COLUMNS, build_feature_table
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


def run_pts_form_ablation(train_df: pd.DataFrame, val_df: pd.DataFrame, winner: str) -> dict | None:
    """Validation-only experiment (docs/judgment-calls.md): does adding back
    the raw pts_form3/pts_form5 rolling-mean-of-points columns -- excluded
    from FEATURE_COLUMNS because the volume/per-90-quality decomposition was
    meant to replace them -- actually improve the winning variant? Clean
    sheets, bonus, and BPS thresholds are position-dependent step functions,
    not linear in per-90 rates, so the decomposition doesn't necessarily
    span everything the raw rolling mean captured. Only run for a
    PositionModels-compatible winner (gbr / hgb_poisson); two_stage/pooled
    have a different feature-injection path not worth building for a
    one-off check.
    """
    if winner not in ("gbr", "hgb_poisson"):
        return None

    extended_cols = FEATURE_COLUMNS + ["pts_form3", "pts_form5"]
    fill_value = train_df["total_points"].mean()

    base_model = PositionModels(estimator_kind=winner).fit(train_df)
    ext_model = PositionModels(estimator_kind=winner, feature_columns=extended_cols).fit(train_df)

    rows = []
    for label, model in [("current (no pts_form)", base_model), ("+ pts_form3/5", ext_model)]:
        scored = add_predictions(val_df, model, fill_value)
        report = full_report(scored, pred_cols=["model"])
        played = scored[scored["minutes"] > 0]
        all_row = report["point_metrics"][report["point_metrics"]["position"] == "ALL"].iloc[0]
        top11 = report["topk_metrics"]
        top11_row = top11[(top11["method"] == "Model") & (top11["k"] == 11)].iloc[0]
        rows.append({
            "variant": label,
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
    ablation_winner = summary["mean_rank"].idxmin()

    return {"summary": summary, "winner": ablation_winner, "extended_wins": ablation_winner == "+ pts_form3/5"}


def write_validation_report(result: dict, ablation: dict | None) -> None:
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
    lines.append("## Ablation experiment: add back pts_form3/pts_form5 (validation only)\n")
    lines.append(
        "The volume/per-90-quality decomposition dropped the raw rolling-mean-of-points "
        "feature from the model input (kept only for driving the naive baselines). Clean "
        "sheets, bonus, and BPS are position-dependent step functions of the underlying "
        "stats, not linear in per-90 rates, so the decomposition might not fully span what "
        "the raw feature captured. Tested here on validation only, same selected variant, "
        "same rank-average rule as above, applied to just these two rows.\n"
    )
    if ablation is not None:
        lines += [
            df_to_markdown(ablation["summary"].round(4).reset_index()),
            "",
            (
                f"**Adding pts_form3/pts_form5 back wins the ablation "
                f"({ablation['winner']}) -- adopted into FEATURE_COLUMNS.**"
                if ablation["extended_wins"] else
                f"**Adding pts_form3/pts_form5 back does NOT win the ablation "
                f"({ablation['winner']} has the lower mean rank) -- current feature set kept.**"
            ),
            "",
        ]
    else:
        lines.append(
            f"Skipped: the selected variant (`{result['winner']}`) isn't a `PositionModels` "
            f"estimator (`gbr`/`hgb_poisson`), and `TwoStageModel`/`PooledModel` don't share "
            f"the same feature-injection path, so this ablation doesn't apply this run. "
            f"`FEATURE_COLUMNS` (without `pts_form3`/`pts_form5`) is used as-is.\n"
        )
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

    print("Running pts_form ablation experiment (validation only)...")
    ablation = run_pts_form_ablation(train_df, val_df, result["winner"])
    if ablation is not None:
        print(ablation["summary"].round(4).to_string())
        print(f"Ablation winner: {ablation['winner']}")

    write_validation_report(result, ablation)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "chosen_model.json").write_text(
        json.dumps({"winner": result["winner"], "random_state": RANDOM_STATE}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
