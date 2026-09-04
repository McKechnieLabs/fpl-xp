# Phase 3: model variant comparison (validation season 2023-24 only)

Selection rule fixed in docs/evaluation-plan.md *before* this was run: rank each variant on {RMSE, MAE, Spearman, realised-points-of-top-11} (minutes > 0 rows, all positions pooled), average the ranks, lowest average wins; ties broken toward the simpler model.

| variant | rmse_played | mae_played | spearman_played | realised_points_of_top11 | mean_rank |
| --- | --- | --- | --- | --- | --- |
| gbr | 2.9974 | 1.9323 | 0.3855 | 55.2895 | 1.75 |
| hgb_poisson | 3.0139 | 1.939 | 0.3735 | 55.5263 | 3.0 |
| two_stage | 3.0896 | 1.9757 | 0.375 | 54.3158 | 3.75 |
| pooled | 2.9928 | 1.934 | 0.3797 | 57.4211 | 1.5 |

**Selected variant: `pooled`** (see reports/chosen_model.json).

## Ablation experiment: add back pts_form3/pts_form5 (validation only)

The volume/per-90-quality decomposition dropped the raw rolling-mean-of-points feature from the model input (kept only for driving the naive baselines). Clean sheets, bonus, and BPS are position-dependent step functions of the underlying stats, not linear in per-90 rates, so the decomposition might not fully span what the raw feature captured. Tested here on validation only, same selected variant, same rank-average rule as above, applied to just these two rows.

Skipped: the selected variant (`pooled`) isn't a `PositionModels` estimator (`gbr`/`hgb_poisson`), and `TwoStageModel`/`PooledModel` don't share the same feature-injection path, so this ablation doesn't apply this run. `FEATURE_COLUMNS` (without `pts_form3`/`pts_form5`) is used as-is.
