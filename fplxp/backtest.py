"""Phase 4: the ONE frozen evaluation on the test set (2024-25 + 2025-26).

Run with: python -m fplxp.backtest

Requires reports/chosen_model.json to already exist (python -m
fplxp.model_selection must run first -- `make all` does this in order).
That file records which variant Phase 3's pre-committed rule selected on
validation; this script does not re-decide anything, it only fits that
one variant on train+validation and scores it once on the held-out test
seasons.

Writes reports/backtest.md (the frozen metric set, conditional-on-minutes
decomposition, calibration plots, top-k results) plus
site/data/metrics.json, site/data/predictions.json, and the plot images,
for the static site (Phase 5) to render.
"""
from __future__ import annotations

import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fplxp.config import ALL_SEASONS, POSITIONS, REPORTS_DIR, SITE_DATA_DIR, TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS
from fplxp.evaluate import BASELINE_LABELS, add_predictions, full_report
from fplxp.features import build_feature_table
from fplxp.model import PooledModel, PositionModels, TwoStageModel
from fplxp.report_utils import df_to_markdown

CHOSEN_MODEL_PATH = REPORTS_DIR / "chosen_model.json"


def _fit_chosen(kind: str, train_df: pd.DataFrame):
    if kind == "two_stage":
        return TwoStageModel().fit(train_df)
    if kind == "pooled":
        return PooledModel().fit(train_df)
    return PositionModels(estimator_kind=kind).fit(train_df)


def load_chosen_variant() -> str:
    if not CHOSEN_MODEL_PATH.exists():
        raise RuntimeError(
            "reports/chosen_model.json not found. Run `python -m fplxp.model_selection` "
            "first (Phase 3 must pick a variant on validation before Phase 4 can run the "
            "one frozen test evaluation) -- `make all` does this in order."
        )
    return json.loads(CHOSEN_MODEL_PATH.read_text())["winner"]


def make_plots(test_df: pd.DataFrame, report: dict, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Predicted vs actual, model only, faceted by position.
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True, sharey=True)
    for ax, pos in zip(axes, POSITIONS):
        sub = test_df[test_df["position"] == pos]
        ax.scatter(sub["total_points"], sub["model"], s=6, alpha=0.25)
        lim = max(sub["total_points"].max(), sub["model"].max(), 5)
        ax.plot([0, lim], [0, lim], color="red", linewidth=1, linestyle="--")
        ax.set_title(pos)
        ax.set_xlabel("actual points")
    axes[0].set_ylabel("predicted points")
    fig.suptitle("v0.2 model: predicted vs. actual points (test seasons)")
    fig.tight_layout()
    fig.savefig(out_dir / "predicted_vs_actual.png", dpi=130)
    plt.close(fig)

    # 2. RMSE (minutes>0 only) by position: model vs baselines.
    pm = report["point_metrics"]
    fig, ax = plt.subplots(figsize=(9, 5))
    positions_plot = POSITIONS + ["ALL"]
    methods = pm["method"].unique().tolist()
    width = 0.8 / max(len(methods), 1)
    x = range(len(positions_plot))
    for i, method in enumerate(methods):
        sub = pm[pm["method"] == method].set_index("position").loc[positions_plot]
        ax.bar([xi + i * width for xi in x], sub["rmse_played"], width=width, label=method)
    ax.set_xticks([xi + width * (len(methods) - 1) / 2 for xi in x])
    ax.set_xticklabels(positions_plot)
    ax.set_ylabel("RMSE (points), minutes > 0 rows only")
    ax.set_title("RMSE by position, players who played: model vs. baselines (test seasons)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "rmse_by_position.png", dpi=130)
    plt.close(fig)

    # 3. Spearman rank correlation per gameweek (minutes>0 rows only), over the test window.
    played = test_df[test_df["minutes"] > 0]
    all_slates = played[["season", "round"]].drop_duplicates().sort_values(["season", "round"]).reset_index(drop=True)
    tick_labels = (all_slates["season"].astype(str) + " GW" + all_slates["round"].astype(str)).tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    for col, label in BASELINE_LABELS.items():
        if col not in played.columns:
            continue
        rows = []
        for _, group in played.groupby(["season", "round"]):
            if group[col].nunique() < 2 or group["total_points"].nunique() < 2:
                continue
            corr = pd.DataFrame({"a": group[col], "b": group["total_points"]}).corr(method="spearman").iloc[0, 1]
            if not np.isnan(corr):
                rows.append({"season": group["season"].iloc[0], "round": group["round"].iloc[0], "spearman": corr})
        series = pd.DataFrame(rows)
        aligned = all_slates.merge(series, on=["season", "round"], how="left")
        ax.plot(range(len(aligned)), aligned["spearman"], label=label, marker="o", markersize=2)
    n = len(all_slates)
    step = max(n // 12, 1)
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels([tick_labels[i] for i in range(0, n, step)], rotation=60, ha="right")
    ax.set_ylabel("Spearman rank correlation")
    ax.set_title("Per-gameweek Spearman (minutes > 0 rows only): model vs. baselines")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "spearman_by_gameweek.png", dpi=130)
    plt.close(fig)

    # 4. Calibration: mean actual vs mean predicted, per decile, per position.
    calib = report["calibration"]
    if not calib.empty:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True, sharey=True)
        for ax, pos in zip(axes, POSITIONS):
            sub = calib[calib["position"] == pos].sort_values("mean_predicted")
            if sub.empty:
                continue
            lim = max(sub["mean_predicted"].max(), sub["mean_actual"].max(), 1)
            ax.plot([0, lim], [0, lim], color="red", linewidth=1, linestyle="--", label="perfect calibration")
            ax.plot(sub["mean_predicted"], sub["mean_actual"], marker="o")
            ax.set_title(pos)
            ax.set_xlabel("mean predicted (decile)")
        axes[0].set_ylabel("mean actual points")
        axes[0].legend(fontsize=7)
        fig.suptitle("Calibration: actual vs. predicted points by predicted decile (test seasons)")
        fig.tight_layout()
        fig.savefig(out_dir / "calibration.png", dpi=130)
        plt.close(fig)

    # 5. Top-k realised points: model vs baselines vs oracle.
    tk = report["topk_metrics"]
    fig, ax = plt.subplots(figsize=(9, 5))
    methods = tk["method"].unique().tolist()
    ks = sorted(tk["k"].unique())
    width = 0.8 / max(len(methods), 1)
    x = range(len(ks))
    for i, method in enumerate(methods):
        sub = tk[tk["method"] == method].set_index("k").reindex(ks)
        ax.bar([xi + i * width for xi in x], sub["realised_points_of_topk"], width=width, label=method)
    ax.set_xticks([xi + width * (len(methods) - 1) / 2 for xi in x])
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("mean realised points of top-k picks")
    ax.set_title("Realised points of top-k picks per gameweek: model vs. baselines vs. oracle")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "topk_realised_points.png", dpi=130)
    plt.close(fig)


def write_backtest_report(test_df: pd.DataFrame, report: dict, variant: str) -> None:
    lines = []
    lines.append("# FPL Expected-Points Backtest -- v0.2\n")
    lines.append(
        "**v0.1's original report is preserved unmodified at "
        "[reports/backtest_v0.1.md](backtest_v0.1.md)** -- see "
        "docs/leakage-audit.md for why it wasn't touched (no leakage found, "
        "so no correction was needed).\n"
    )
    lines.append(f"Train seasons: {', '.join(TRAIN_SEASONS)}")
    lines.append(f"Validation season (Phase 3 model selection): {', '.join(VAL_SEASONS)}")
    lines.append(
        f"Test seasons: {', '.join(TEST_SEASONS)} (frozen -- touched exactly once, here, "
        f"for this report)\n"
    )
    lines.append(
        f"Selected model: **`{variant}`**, chosen by the pre-committed rule in "
        f"docs/evaluation-plan.md on validation only -- see "
        f"reports/validation_comparison.md for the comparison. Refit on "
        f"train+validation combined before this one test run.\n"
    )
    lines.append(f"Test rows: {len(test_df):,} ({(test_df['minutes'] > 0).sum():,} with minutes > 0)\n")

    lines.append("## Metrics: all rows vs. minutes > 0 only\n")
    lines.append(
        "Reported both ways per docs/evaluation-plan.md: a model that mostly predicts "
        "\"0 points, didn't play\" correctly is demonstrating a real but different skill "
        "from ranking players who actually featured. This split is a headline finding, "
        "not a footnote.\n"
    )
    pm_cols = ["method", "position", "rmse", "mae", "spearman", "rmse_played", "mae_played", "spearman_played", "n", "n_played"]
    lines.append(df_to_markdown(report["point_metrics"][pm_cols].round(4)))
    lines.append("")

    lines.append("## Decision metric: precision@k and realised points of top-k\n")
    lines.append(
        "Within each gameweek's slate, the model's/baseline's top-k predicted players "
        "vs. the actual top-k by real points, and how many points those top-k picks "
        "actually returned -- compared against an oracle (perfect hindsight) as the "
        "theoretical ceiling.\n"
    )
    lines.append(df_to_markdown(report["topk_metrics"].round(4)))
    lines.append("")

    lines.append("## Calibration\n")
    lines.append(
        "Mean actual points by predicted-points decile, per position. Systematic "
        "under-prediction in the top decile is the failure mode that breaks captaincy "
        "picks.\n"
    )
    lines.append("![Calibration](calibration.png)\n")

    lines.append("## Plots\n")
    lines.append("![Predicted vs actual](predicted_vs_actual.png)\n")
    lines.append("![RMSE by position](rmse_by_position.png)\n")
    lines.append("![Spearman by gameweek](spearman_by_gameweek.png)\n")
    lines.append("![Top-k realised points](topk_realised_points.png)\n")

    lines.append("## Verdict\n")
    model_row = report["point_metrics"]
    model_all = model_row[(model_row["method"] == "Model") & (model_row["position"] == "ALL")].iloc[0]
    others = model_row[(model_row["method"] != "Model") & (model_row["position"] == "ALL")]
    best_rmse_played = others.loc[others["rmse_played"].idxmin()]
    best_spearman_played = others.loc[others["spearman_played"].idxmax()]

    beats_rmse = model_all["rmse_played"] < best_rmse_played["rmse_played"]
    beats_spearman = model_all["spearman_played"] > best_spearman_played["spearman_played"]

    if beats_rmse:
        lines.append(
            f"- **RMSE (minutes > 0):** model beats the best baseline "
            f"({model_all['rmse_played']:.3f} vs {best_rmse_played['rmse_played']:.3f}, "
            f"{best_rmse_played['method']})."
        )
    else:
        lines.append(
            f"- **RMSE (minutes > 0): model does NOT beat the best baseline** "
            f"({model_all['rmse_played']:.3f} vs {best_rmse_played['rmse_played']:.3f}, "
            f"{best_rmse_played['method']}). Reported as-is."
        )
    if beats_spearman:
        lines.append(
            f"- **Spearman (minutes > 0):** model beats the best baseline "
            f"({model_all['spearman_played']:.3f} vs {best_spearman_played['spearman_played']:.3f}, "
            f"{best_spearman_played['method']})."
        )
    else:
        lines.append(
            f"- **Spearman (minutes > 0): model does NOT beat the best baseline** "
            f"({model_all['spearman_played']:.3f} vs {best_spearman_played['spearman_played']:.3f}, "
            f"{best_spearman_played['method']}). Reported as-is rather than tuned away."
        )

    tk = report["topk_metrics"]
    k11 = tk[tk["k"] == 11].set_index("method")
    if "Model" in k11.index:
        model_pts = k11.loc["Model", "realised_points_of_topk"]
        oracle_pts = k11.loc["Oracle (perfect hindsight)", "realised_points_of_topk"]
        best_baseline_pts = k11.drop(index=["Model", "Oracle (perfect hindsight)"], errors="ignore")["realised_points_of_topk"].max()
        lines.append(
            f"- **Top-11 realised points (the decision-relevant number):** model's top-11 "
            f"picks average {model_pts:.1f} points/gameweek, vs. {best_baseline_pts:.1f} "
            f"for the best baseline and {oracle_pts:.1f} for a perfect-hindsight oracle."
        )
    lines.append("")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "backtest.md").write_text("\n".join(lines))


def write_site_data(test_df: pd.DataFrame, report: dict, variant: str) -> None:
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "generated_from": "fplxp.backtest",
        "selected_model": variant,
        "train_seasons": TRAIN_SEASONS,
        "val_season": VAL_SEASONS,
        "test_seasons": TEST_SEASONS,
        "n_test_rows": int(len(test_df)),
        "n_test_rows_played": int((test_df["minutes"] > 0).sum()),
        "point_metrics": json.loads(report["point_metrics"].to_json(orient="records")),
        "topk_metrics": json.loads(report["topk_metrics"].to_json(orient="records")),
        "calibration": json.loads(report["calibration"].to_json(orient="records")),
    }
    (SITE_DATA_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    display_cols = ["season", "round", "name", "position", "team", "total_points", "model", "baseline_last5", "baseline_season", "baseline_xp"]
    sample = test_df.sort_values(["season", "round"], ascending=[False, False]).head(2000)
    predictions = json.loads(sample[display_cols].round(2).to_json(orient="records"))
    (SITE_DATA_DIR / "predictions.json").write_text(json.dumps(predictions))

    # The site is self-contained: copy the plot images generated into
    # reports/ alongside the JSON, under site/plots/, rather than having
    # index.html reach outside site/ at deploy time.
    plots_dir = SITE_DATA_DIR.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for name in ["predicted_vs_actual.png", "rmse_by_position.png", "spearman_by_gameweek.png",
                 "calibration.png", "topk_realised_points.png"]:
        src = REPORTS_DIR / name
        if src.exists():
            shutil.copy(src, plots_dir / name)


def main():
    variant = load_chosen_variant()
    print(f"Phase 4: final frozen evaluation using selected variant `{variant}`")

    print("Building feature table for all seasons (uses cached CSVs under data/raw/)...")
    table = build_feature_table(ALL_SEASONS)
    trainval_df = table[table["season"].isin(TRAIN_SEASONS + VAL_SEASONS)].copy()
    test_df = table[table["season"].isin(TEST_SEASONS)].copy()
    print(f"train+val rows: {len(trainval_df)} | TEST rows: {len(test_df)} (frozen, scored once)")

    print("Fitting selected model on train+validation...")
    model = _fit_chosen(variant, trainval_df)

    print("Scoring test set (once)...")
    fill_value = trainval_df["total_points"].mean()
    test_df = add_predictions(test_df, model, fill_value)

    report = full_report(test_df)
    print(report["point_metrics"].round(3).to_string(index=False))

    print("Writing plots...")
    make_plots(test_df, report, REPORTS_DIR)

    print("Writing reports/backtest.md...")
    write_backtest_report(test_df, report, variant)

    print("Writing site/data/*.json...")
    write_site_data(test_df, report, variant)

    print("Done.")


if __name__ == "__main__":
    main()
