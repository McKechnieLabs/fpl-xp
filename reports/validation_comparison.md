# Phase 3: model variant comparison (validation season 2023-24 only)

Selection rule fixed in docs/evaluation-plan.md *before* this was run: rank each variant on {RMSE, MAE, Spearman, realised-points-of-top-11} (minutes > 0 rows, all positions pooled), average the ranks, lowest average wins; ties broken toward the simpler model.

| variant | rmse_played | mae_played | spearman_played | realised_points_of_top11 | mean_rank |
| --- | --- | --- | --- | --- | --- |
| gbr | 2.9924 | 1.9274 | 0.3867 | 57.6842 | 1.0 |
| hgb_poisson | 3.0074 | 1.9345 | 0.3745 | 56.4474 | 3.25 |
| two_stage | 3.0907 | 1.9695 | 0.3758 | 54.3684 | 3.75 |
| pooled | 2.9945 | 1.9336 | 0.382 | 56.8947 | 2.0 |

**Selected variant: `gbr`** (see reports/chosen_model.json).
