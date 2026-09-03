# Phase 0 leakage audit

**Conclusion: no leakage found, in either the v0.1 pipeline or the v0.2
rebuild.** This document records the audit that reached that conclusion,
per the v0.2 brief's requirement that the audit be done and shown, not
assumed.

## Why an audit was warranted

The v0.1 report claimed the model beat both naive baselines decisively on
RMSE and MAE, in every position, with zero tuning. That is a real result,
but it is *also* the exact fingerprint target leakage produces, so it
can't be taken at face value without checking. The most common concrete
bug that produces it: `groupby(...).rolling(...)` without a `.shift(1)`
first, which lets a gameweek's own outcome leak into its own "rolling"
feature.

## 1. Every rolling/expanding expression, read directly from the code

`fplxp/features.py`, `_add_player_rolling` (player-level features):

```python
shifted = grp[col].shift(1)          # <- shift BEFORE window, every time
... .transform(lambda x, w=w: x.rolling(w, min_periods=1).mean())
```

and for per-90 rates:

```python
mins_shifted = shifted("minutes")
sum_mins_w = mins_shifted.groupby(key).transform(lambda x, w=w: x.rolling(w, min_periods=1).sum())
sum_stat_w = shifted(src).groupby(key).transform(lambda x, w=w: x.rolling(w, min_periods=1).sum())
```

`_add_team_rolling` (team-level goals for/against):

```python
shifted = team_gw.groupby(["season", "team_id"])[col].shift(1)
team_gw[f"{out}{TEAM_ROLLING_WINDOW}"] = shifted.groupby(key).transform(
    lambda s: s.rolling(TEAM_ROLLING_WINDOW, min_periods=1).mean()
)
```

`_add_prior_season_carryover` only ever reads a strictly **earlier**
season's full-season aggregate to backfill a gameweek-1 row -- by
construction that season already finished before the current one starts,
so there is no shift to get wrong.

In every case, `.shift(1)` (or an equivalent one-row lag via `groupby(...).shift(1)`
before the window) is applied to the raw per-gameweek value **before**
`.rolling()`/`.expanding()` ever sees it. There is no `groupby(...).rolling(...)`
call anywhere in the codebase without a preceding shift -- confirmed by
`grep -n "\.rolling(\|\.expanding(" fplxp/features.py` and reading every
hit against its preceding `shift(1)`.

Same-round (unshifted) features -- `num_fixtures`, `home_fraction`,
`fixture_difficulty_{min,max,mean}`, `own/opp_{attack,defence}_strength`,
`value`, `selected_log` (+ deltas), `is_blank`, `gw_number`,
`is_first_gw_of_season`, `has_prior_season` -- are all set by FPL, or
derivable from the fixture list, *before* kickoff and don't depend on the
match's outcome, so there's nothing to shift. `tests/test_no_leakage.py::test_same_round_columns_are_the_only_legitimate_ones`
enforces that this is the *complete* list: any new same-round-ish column
added to the feature set without also being added to that test's
independently-written whitelist fails the suite.

## 2. Synthetic spike test

`tests/test_no_leakage.py::test_spike_gameweek_features_do_not_reflect_the_spike`
builds a full synthetic season through the *real* pipeline (`fplxp.data.load_all`
monkeypatched, no network) in which one player scores 0 every gameweek
except a single 100-point gameweek-6 spike. It asserts every rolling
feature at gameweek 6 for the spiking player is identical to a player who
never spikes -- i.e. the 100 points is invisible to that gameweek's own
features -- and separately that gameweek 7's features *do* reflect the
spike (proving the test isn't vacuously passing because the pipeline
ignores history altogether). A companion test checks gameweeks 1-5 are
unaffected by the spike too (a later outcome can't reach backward).

## 3. Correlation sanity check on real train data

`tests/test_no_leakage.py::test_no_feature_is_suspiciously_correlated_with_target`
computes `Spearman(feature, total_points)` for every column in
`FEATURE_COLUMNS` over the training seasons (2020-21..2022-23) and fails
if any exceeds 0.9 in magnitude -- the threshold a genuinely leaked
feature (e.g. a same-gameweek raw stat that slipped through) would blow
past immediately. The actual maximum found is 0.64
(`mins_form3`, rolling minutes), with the next few all also
minutes/availability features (`starts_rate_form3` 0.64,
`played60_rate_form3` 0.60). This is exactly the expected shape: minutes
are the single biggest driver of FPL points (a player who doesn't play
can't score), so a moderately strong, *not suspiciously perfect*,
correlation on the minutes-availability features is the signature of a
clean pipeline, not a leaked one. No output-quality feature (goals/assists/bps
per-90 rates) exceeds 0.46.

## 4. What this says about the v0.1 RMSE/MAE win

The v0.1 report is corroborating evidence, not just consistent with a
clean pipeline: it did *not* also show the model crushing the baselines
on Spearman rank correlation (it lost narrowly on Spearman overall and in
3 of 4 positions -- see the old `reports/backtest.md` v0.1 section). A
model with real, undocumented lookahead into `total_points` would be
expected to dominate on *every* metric, including rank correlation,
because it would effectively already know the answer. Winning decisively
on RMSE/MAE (driven substantially by the same minutes-availability
signal identified above -- a genuinely strong, legitimate predictor) while
only roughly matching a simple average on ranking quality is the profile
of an honest, moderately-informative feature set, not a leaked one.

## Verdict

No leakage found. No fix was necessary, so v0.1's original numbers stand
unmodified in `reports/backtest.md`'s v0.1 section; v0.2's rebuilt
pipeline (new split, new features) is evaluated fresh in the same file's
v0.2 section, using the same audit methodology re-applied after the
rebuild (all 9 tests in `tests/test_no_leakage.py` pass against the final
v0.2 feature set, not just the pre-rebuild one).
