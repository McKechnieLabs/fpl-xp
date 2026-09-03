# Evaluation plan (frozen before Phase 3 results exist)

This document is written before any v0.2 model variant is trained or
scored. Its purpose is to fix the metric set in advance so that later
model choices can't be justified by picking whichever metric happens to
look best after the fact ("metric shopping"). Every metric listed here is
reported for every model/baseline in `reports/backtest.md`; none are
added or dropped after seeing results.

## Data discipline

- All Phase 3 comparisons run on **train (2020-21..2022-23) fit, validation
  (2023-24) scored**. The test set (2024-25, 2025-26) is not touched until
  Phase 4, exactly once.
- Any hyperparameter selection uses forward-chaining CV by season (fit on
  season *t*, score on season *t+1*, within the training seasons only) or
  `GroupKFold` grouped by player -- never random `KFold`, since adjacent
  gameweeks of the same player are correlated and would leak across folds.

## Metric set

For every model/baseline (`model`, `baseline_last5`, `baseline_season`,
`baseline_xp` where available), report all of the following, each broken
down **by position** and **ALL**:

1. **RMSE** and **MAE** -- standard point-accuracy metrics, kept from
   v0.1 as a continuity baseline.
2. **The same RMSE/MAE/Spearman, computed twice**: once over all rows,
   and once restricted to rows with `minutes > 0`. This decomposition is
   frozen as a headline result, not a diagnostic aside: a model that is
   mostly predicting "0 points, didn't play" correctly is demonstrating a
   real but different skill from ranking players who actually featured.
   Reporting only the pooled number lets the first skill silently stand
   in for the second.
3. **Mean per-gameweek Spearman rank correlation** -- computed within
   each (season, round) slate and averaged, as in v0.1, via
   `DataFrame.corr(method="spearman")` (no scipy).
4. **Precision@k** for k in {1, 11, 15} -- within each gameweek's slate,
   take the model's top-k predicted players; precision@k is the fraction
   of those k who were also in the *actual* top-k by real points that
   gameweek. Averaged across gameweeks.
5. **Realised points of top-k** for k in {1, 11, 15} -- the sum of real
   `total_points` earned by the model's top-k picks that gameweek,
   averaged across gameweeks, reported alongside the same quantity for:
   - each baseline,
   - an **oracle** (the actual top-k by real points that gameweek --
     the theoretical ceiling).
   This is the decision-relevant number: RMSE is a proxy for it, this
   *is* the thing a squad-selection decision actually cares about.
6. **Calibration**: mean actual points by predicted-points decile, per
   position, plotted. Systematic under-prediction in the top decile is
   the specific failure mode that breaks captaincy picks (the model
   telling you your best player is a safe, boring floor pick when they're
   actually a explosive ceiling pick) -- this is checked explicitly
   rather than hoped RMSE would catch it.

## Baselines compared

- `baseline_last5`: player's mean points over their last 5 gameweeks.
- `baseline_season`: player's season-to-date mean points.
- `baseline_xp`: FPL's own pre-match `xP` for that gameweek, for seasons
  where the column exists (2020-21 onward, i.e. all seasons in this
  project). `xP` is never used as a *feature* (docs/judgment-calls.md
  v0.1 #3) but it is the natural benchmark everyone actually compares FPL
  models against, and the column is already in the data, so withholding
  it from the report table would be an unforced omission.

## Model variants compared (validation only, per docs/judgment-calls.md)

In this order, each kept only if it earns its place:

1. **v0.1-style per-position `GradientBoostingRegressor`** (squared-error
   loss) -- carried forward as the continuity baseline for the model
   itself, re-fit on the new split/features.
2. **Per-position `HistGradientBoostingRegressor(loss="poisson")`** --
   tried first among the new variants: `total_points` is a non-negative,
   right-skewed count-like target, and Poisson deviance is the
   textbook-correct loss for that shape, unlike squared error which
   implicitly assumes symmetric, homoscedastic noise.
3. **Two-stage**: a classifier for `P(played_60)` times a regressor for
   `E[points | played_60]`, multiplied. Compared against the single-stage
   model on the same metric set.
4. **Pooled model**: one `HistGradientBoostingRegressor` over all
   positions with `position` as a categorical feature, as a fourth
   baseline -- checks whether per-position splits are starving the
   smallest class (GK) of training data rather than actually helping.

Whichever variant wins on the metric set **as a whole** (not
cherry-picked per metric) is the one carried into Phase 4. If no variant
clearly wins across the set, the simplest one is kept and that is stated
plainly, per docs/judgment-calls.md v0.1 #9's precedent of reporting a
mixed result honestly rather than picking a favorite post hoc.

## Pre-committed selection rule

To avoid picking a "winner" by eyeballing whichever number looks best
after the fact, the variant carried into Phase 4 is chosen by a rule
fixed here, before any variant is trained:

> On the validation set, ALL positions pooled, restricted to
> `minutes > 0` rows (see "split the two skills" above -- this is the
> ranking-relevant regime): rank the four variants on each of
> {RMSE, MAE, Spearman, realised-points-of-top-11}, lower-is-better
> reversed for the latter two. Average each variant's four ranks. Lowest
> average rank wins. A tie (average ranks within 0.1 of each other) is
> broken toward the simpler model, in this order:
> `gbr` (v0.1-style) > `hgb_poisson` > two-stage > pooled.

This rule is mechanical and reported alongside the raw numbers in
`reports/validation_comparison.md`, so the choice is auditable even
though it isn't hand-picked.

## What "winning" means here

No single metric is nominated as the tiebreaker in advance, deliberately:
the whole point of Phase 3 is that RMSE, Spearman, and top-k realised
points can and do disagree (v0.1 already showed RMSE and Spearman
diverging). Phase 4's report presents the full table and states, in
words, which variant is best on which metric and why that might be -- not
a single collapsed leaderboard rank.
