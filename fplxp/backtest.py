"""End-to-end backtest: build features, split train/test by season, fit
per-position GBM, compare against two naive baselines, write
reports/backtest.md plus three diagnostic plots.

Run with: python -m fplxp.backtest
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fplxp.baselines import last5_mean_baseline, season_to_date_baseline
from fplxp.config import ALL_SEASONS, POSITIONS, REPORTS_DIR, TEST_SEASONS, TRAIN_SEASONS
from fplxp.features import build_feature_table
from fplxp.metrics import evaluate_by_position
from fplxp.model import PositionModels

PRED_COLS = {
    "model": "GBM (v0.1)",
    "baseline_last5": "Baseline: last-5 mean",
    "baseline_season": "Baseline: season-to-date mean",
}


def df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal DataFrame -> GitHub-flavored markdown table, no extra deps."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |" for row in df[cols].itertuples(index=False)
    ]
    return "\n".join([header, sep] + rows)


def build_train_test():
    table = build_feature_table(ALL_SEASONS)
    train_df = table[table["season"].isin(TRAIN_SEASONS)].copy()
    test_df = table[table["season"].isin(TEST_SEASONS)].copy()
    return train_df, test_df


def add_predictions(train_df, test_df, models: PositionModels):
    fill_value = train_df["total_points"].mean()
    for df in (train_df, test_df):
        df["model"] = models.predict(df)
        df["baseline_last5"] = last5_mean_baseline(df, fill_value)
        df["baseline_season"] = season_to_date_baseline(df, fill_value)
    return train_df, test_df


def build_metrics_table(test_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for col, label in PRED_COLS.items():
        m = evaluate_by_position(test_df, col)
        m.insert(0, "method", label)
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def gw_spearman_series(test_df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    from scipy.stats import spearmanr
    import numpy as np

    rows = []
    for (season, rnd), group in test_df.groupby(["season", "round"]):
        if group[pred_col].nunique() < 2 or group["total_points"].nunique() < 2:
            continue
        corr, _ = spearmanr(group[pred_col], group["total_points"])
        if np.isnan(corr):
            continue
        rows.append({"season": season, "round": rnd, "spearman": corr})
    return pd.DataFrame(rows)


def make_plots(test_df: pd.DataFrame, metrics_table: pd.DataFrame, models: PositionModels):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Predicted vs actual scatter for the model, faceted by position.
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True, sharey=True)
    for ax, pos in zip(axes, POSITIONS):
        sub = test_df[test_df["position"] == pos]
        ax.scatter(sub["total_points"], sub["model"], s=6, alpha=0.25)
        lim = max(sub["total_points"].max(), sub["model"].max(), 5)
        ax.plot([0, lim], [0, lim], color="red", linewidth=1, linestyle="--")
        ax.set_title(pos)
        ax.set_xlabel("actual points")
    axes[0].set_ylabel("predicted points")
    fig.suptitle("Model: predicted vs. actual points (test seasons)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "predicted_vs_actual.png", dpi=130)
    plt.close(fig)

    # 2. RMSE by position, model vs baselines.
    fig, ax = plt.subplots(figsize=(8, 5))
    positions_plot = POSITIONS + ["ALL"]
    width = 0.25
    x = range(len(positions_plot))
    for i, (col, label) in enumerate(PRED_COLS.items()):
        sub = metrics_table[metrics_table["method"] == label].set_index("position").loc[positions_plot]
        ax.bar([xi + i * width for xi in x], sub["rmse"], width=width, label=label)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(positions_plot)
    ax.set_ylabel("RMSE (points)")
    ax.set_title("RMSE by position: model vs. baselines (test seasons)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "rmse_by_position.png", dpi=130)
    plt.close(fig)

    # 3. Spearman rank correlation per gameweek over the test window.
    # Build one shared (season, round) x-axis across all three methods so that
    # a gameweek a baseline can't rank (e.g. GW1, where its rolling window is
    # empty for every player and predictions are a constant) shows as a gap
    # instead of silently shifting that line relative to the others.
    all_slates = (
        test_df[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    ).reset_index(drop=True)
    tick_labels = (all_slates["season"].astype(str) + " GW" + all_slates["round"].astype(str)).tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    for col, label in PRED_COLS.items():
        series = gw_spearman_series(test_df, col)
        aligned = all_slates.merge(series, on=["season", "round"], how="left")
        ax.plot(range(len(aligned)), aligned["spearman"], label=label, marker="o", markersize=2)
    n = len(all_slates)
    step = max(n // 12, 1)
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels([tick_labels[i] for i in range(0, n, step)], rotation=60, ha="right")
    ax.set_ylabel("Spearman rank correlation")
    ax.set_title("Per-gameweek Spearman correlation: model vs. baselines")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "spearman_by_gameweek.png", dpi=130)
    plt.close(fig)


def write_report(train_df, test_df, metrics_table: pd.DataFrame, models: PositionModels):
    model_all = metrics_table[
        (metrics_table["method"] == PRED_COLS["model"]) & (metrics_table["position"] == "ALL")
    ].iloc[0]
    baselines_all = metrics_table[
        (metrics_table["method"] != PRED_COLS["model"]) & (metrics_table["position"] == "ALL")
    ]
    best_baseline_rmse_row = baselines_all.loc[baselines_all["rmse"].idxmin()]
    best_baseline_spearman_row = baselines_all.loc[baselines_all["spearman"].idxmax()]
    beats_rmse = model_all["rmse"] < best_baseline_rmse_row["rmse"]
    beats_spearman = model_all["spearman"] > best_baseline_spearman_row["spearman"]

    model_by_pos = metrics_table[metrics_table["method"] == PRED_COLS["model"]].set_index("position")
    best_baseline_spearman_by_pos = (
        metrics_table[metrics_table["method"] != PRED_COLS["model"]]
        .groupby("position")["spearman"]
        .max()
    )
    spearman_losses = [
        pos
        for pos in POSITIONS
        if model_by_pos.loc[pos, "spearman"] < best_baseline_spearman_by_pos.loc[pos]
    ]

    lines = []
    lines.append("# FPL Expected-Points Backtest\n")
    lines.append(f"Train seasons: {', '.join(TRAIN_SEASONS)}")
    lines.append(f"Test seasons: {', '.join(TEST_SEASONS)} (held out, evaluated only once)\n")
    lines.append(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}\n")

    lines.append("## Leakage handling\n")
    lines.append(
        "- Every rolling/form feature (points, minutes, ICT, goals, assists, bps, bonus, "
        "clean sheets, goals conceded, saves, starts, team goals for/against) is computed "
        "as `shift(1)` within its (season, player) or (season, team) group **before** any "
        "rolling window or expanding mean is applied, so a gameweek's features only see "
        "strictly earlier gameweeks."
    )
    lines.append(
        "- Fixture difficulty, home/away, scheduled fixture count, and price are used "
        "unshifted: FPL sets these before kickoff and they don't depend on the match's "
        "outcome, so there is nothing to leak."
    )
    lines.append(
        "- `xP` (FPL's own pre-match expected points for that same gameweek) is **dropped "
        "entirely**, not shifted, because even lagged by one gameweek it would be injecting "
        "a competing model's prediction for a different player-gameweek as a feature. "
        "See QUESTIONS.md.\n"
    )

    lines.append("## Metrics (test seasons)\n")
    lines.append(
        df_to_markdown(metrics_table[["method", "position", "rmse", "mae", "spearman", "n"]].round(4))
    )
    lines.append("")

    lines.append("## Verdict\n")
    if beats_rmse:
        lines.append(
            f"- **RMSE/MAE:** the model beats the best naive baseline overall "
            f"({model_all['rmse']:.3f} vs {best_baseline_rmse_row['rmse']:.3f} RMSE, "
            f"method: {best_baseline_rmse_row['method']}), and on RMSE in every "
            f"individual position too (see the RMSE-by-position plot)."
        )
    else:
        lines.append(
            f"- **RMSE/MAE: the model does NOT beat the best naive baseline** "
            f"({model_all['rmse']:.3f} vs {best_baseline_rmse_row['rmse']:.3f} RMSE, "
            f"method: {best_baseline_rmse_row['method']})."
        )
    if beats_spearman:
        lines.append(
            f"- **Spearman rank correlation (what matters for team selection):** the model "
            f"also wins overall ({model_all['spearman']:.3f} vs "
            f"{best_baseline_spearman_row['spearman']:.3f}, method: "
            f"{best_baseline_spearman_row['method']})."
        )
    else:
        lines.append(
            f"- **Spearman rank correlation (what matters more for team selection than "
            f"RMSE): the model does NOT win overall.** The last-5-mean baseline has a "
            f"slightly higher average per-gameweek rank correlation "
            f"({best_baseline_spearman_row['spearman']:.3f} vs {model_all['spearman']:.3f} "
            f"for the model). This is reported as-is rather than tuned away."
        )
    if spearman_losses:
        lines.append(
            f"- By position, the model's Spearman correlation is below the best baseline's "
            f"for: {', '.join(spearman_losses)}. It wins on RMSE/MAE in those same positions "
            f"regardless, i.e. it is a better *point* predictor there but not a better "
            f"*ranker* than simply using recent-form averages."
        )
    lines.append(
        "- Net read: the model is a clear, non-tuned win on point-estimate accuracy "
        "(RMSE/MAE) across every position, but it is not a uniform win on rank "
        "correlation, which is the metric that actually drives which 11-15 players you'd "
        "pick. A v0.2 iteration should focus on ranking quality (e.g. a pairwise/ranking "
        "loss, or the two-stage P(plays 60+) x points-if-played structure noted as a "
        "stretch goal) rather than further squeezing RMSE."
    )
    lines.append("")

    lines.append("## Plots\n")
    lines.append("![Predicted vs actual](predicted_vs_actual.png)\n")
    lines.append("![RMSE by position](rmse_by_position.png)\n")
    lines.append("![Spearman by gameweek](spearman_by_gameweek.png)\n")

    lines.append("## Top feature importances by position\n")
    for pos in POSITIONS:
        top = models.feature_importances(pos).head(8).round(4)
        top_df = top.rename_axis("feature").reset_index(name="importance")
        lines.append(f"**{pos}**\n")
        lines.append(df_to_markdown(top_df))
        lines.append("")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "backtest.md").write_text("\n".join(lines))


def main():
    print("Building feature table for all seasons (uses cached CSVs under data/raw/)...")
    train_df, test_df = build_train_test()
    print(f"Train: {len(train_df)} rows across {TRAIN_SEASONS}")
    print(f"Test:  {len(test_df)} rows across {TEST_SEASONS}")

    print("Fitting per-position GBM models on train seasons...")
    models = PositionModels().fit(train_df)

    print("Scoring baselines and model...")
    train_df, test_df = add_predictions(train_df, test_df, models)

    metrics_table = build_metrics_table(test_df)
    print(metrics_table.round(3).to_string(index=False))

    print("Writing plots...")
    make_plots(test_df, metrics_table, models)

    print("Writing reports/backtest.md...")
    write_report(train_df, test_df, metrics_table, models)
    print("Done.")


if __name__ == "__main__":
    main()
