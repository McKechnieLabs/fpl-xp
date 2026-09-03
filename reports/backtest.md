# FPL Expected-Points Backtest -- v0.2

**v0.1's original report is preserved unmodified at [reports/backtest_v0.1.md](backtest_v0.1.md)** -- see docs/leakage-audit.md for why it wasn't touched (no leakage found, so no correction was needed).

Train seasons: 2020-21, 2021-22, 2022-23
Validation season (Phase 3 model selection): 2023-24
Test seasons: 2024-25, 2025-26 (frozen -- touched exactly once, here, for this report)

Selected model: **`gbr`**, chosen by the pre-committed rule in docs/evaluation-plan.md on validation only -- see reports/validation_comparison.md for the comparison. Refit on train+validation combined before this one test run.

Test rows: 57,026 (22,792 with minutes > 0)

## Metrics: all rows vs. minutes > 0 only

Reported both ways per docs/evaluation-plan.md: a model that mostly predicts "0 points, didn't play" correctly is demonstrating a real but different skill from ranking players who actually featured. This split is a headline finding, not a footnote.

| method | position | rmse | mae | spearman | rmse_played | mae_played | spearman_played | n | n_played |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | DEF | 1.9933 | 1.0502 | 0.6415 | 3.0126 | 2.0219 | 0.3049 | 18879 | 7649 |
| Model | FWD | 2.1735 | 1.1471 | 0.7398 | 3.1827 | 2.1242 | 0.4604 | 6294 | 2670 |
| Model | GK | 1.5338 | 0.6775 | 0.6572 | 2.9319 | 2.1815 | 0.0768 | 6296 | 1517 |
| Model | MID | 1.9314 | 0.9774 | 0.7231 | 2.7947 | 1.7206 | 0.4005 | 25557 | 10956 |
| Model | ALL | 1.9412 | 0.9872 | 0.7008 | 2.9256 | 1.8997 | 0.3666 | 57026 | 22792 |
| Baseline: last-5 mean | DEF | 2.2056 | 1.1633 | 0.615 | 3.2592 | 2.311 | 0.2013 | 18879 | 7649 |
| Baseline: last-5 mean | FWD | 2.3811 | 1.1924 | 0.7517 | 3.4802 | 2.3488 | 0.4055 | 6294 | 2670 |
| Baseline: last-5 mean | GK | 1.6863 | 0.7191 | 0.7821 | 3.2255 | 2.4083 | 0.0708 | 6296 | 1517 |
| Baseline: last-5 mean | MID | 2.1242 | 1.0624 | 0.7162 | 3.0434 | 1.9808 | 0.3391 | 25557 | 10956 |
| Baseline: last-5 mean | ALL | 2.1383 | 1.0722 | 0.6976 | 3.1826 | 2.1632 | 0.2907 | 57026 | 22792 |
| Baseline: season-to-date mean | DEF | 2.1457 | 1.167 | 0.5976 | 3.1515 | 2.1884 | 0.2207 | 18879 | 7649 |
| Baseline: season-to-date mean | FWD | 2.3282 | 1.2223 | 0.7169 | 3.3568 | 2.2643 | 0.4249 | 6294 | 2670 |
| Baseline: season-to-date mean | GK | 1.6544 | 0.7446 | 0.7327 | 3.1375 | 2.3015 | 0.0872 | 6296 | 1517 |
| Baseline: season-to-date mean | MID | 2.0651 | 1.075 | 0.6838 | 2.9225 | 1.8577 | 0.3617 | 25557 | 10956 |
| Baseline: season-to-date mean | ALL | 2.0823 | 1.0852 | 0.6697 | 3.0682 | 2.0459 | 0.3117 | 57026 | 22792 |
| Baseline: FPL xP | DEF | 2.1762 | 1.0341 | 0.6962 | 3.3597 | 2.2857 | 0.5386 | 18879 | 7649 |
| Baseline: FPL xP | FWD | 2.3193 | 1.076 | 0.798 | 3.5052 | 2.3503 | 0.6305 | 6294 | 2670 |
| Baseline: FPL xP | GK | 1.747 | 0.7573 | 0.7119 | 3.4217 | 2.4759 | 0.4524 | 6296 | 1517 |
| Baseline: FPL xP | MID | 2.1388 | 0.9918 | 0.7817 | 3.2143 | 2.1048 | 0.5901 | 25557 | 10956 |
| Baseline: FPL xP | ALL | 2.1327 | 0.9892 | 0.7524 | 3.3125 | 2.2189 | 0.581 | 57026 | 22792 |

## Decision metric: precision@k and realised points of top-k

Within each gameweek's slate, the model's/baseline's top-k predicted players vs. the actual top-k by real points, and how many points those top-k picks actually returned -- compared against an oracle (perfect hindsight) as the theoretical ceiling.

| method | k | precision_at_k | realised_points_of_topk |
| --- | --- | --- | --- |
| Model | 1 | 0.0263 | 5.75 |
| Model | 11 | 0.1483 | 57.0 |
| Model | 15 | 0.1675 | 74.0263 |
| Baseline: last-5 mean | 1 | 0.0395 | 5.3158 |
| Baseline: last-5 mean | 11 | 0.1005 | 46.0921 |
| Baseline: last-5 mean | 15 | 0.1114 | 59.6711 |
| Baseline: season-to-date mean | 1 | 0.0789 | 7.0263 |
| Baseline: season-to-date mean | 11 | 0.1304 | 49.0789 |
| Baseline: season-to-date mean | 15 | 0.1404 | 63.1974 |
| Baseline: FPL xP | 1 | 0.1053 | 8.4737 |
| Baseline: FPL xP | 11 | 0.2045 | 60.1842 |
| Baseline: FPL xP | 15 | 0.2237 | 76.5921 |
| Oracle (perfect hindsight) | 1 | 1.0 | 17.6053 |
| Oracle (perfect hindsight) | 11 | 1.0 | 138.6184 |
| Oracle (perfect hindsight) | 15 | 1.0 | 176.8553 |

## Calibration

Mean actual points by predicted-points decile, per position. Systematic under-prediction in the top decile is the failure mode that breaks captaincy picks.

![Calibration](calibration.png)

## Plots

![Predicted vs actual](predicted_vs_actual.png)

![RMSE by position](rmse_by_position.png)

![Spearman by gameweek](spearman_by_gameweek.png)

![Top-k realised points](topk_realised_points.png)

## Verdict

- **RMSE (minutes > 0):** model beats the best baseline (2.926 vs 3.068, Baseline: season-to-date mean).
- **Spearman (minutes > 0): model does NOT beat the best baseline** (0.367 vs 0.581, Baseline: FPL xP). Reported as-is rather than tuned away.
- **Top-11 realised points (the decision-relevant number):** model's top-11 picks average 57.0 points/gameweek, vs. 60.2 for the best baseline and 138.6 for a perfect-hindsight oracle.
