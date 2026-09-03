# Judgment calls made without asking

Per instructions, this project was built end-to-end without pausing for
clarification. Every non-obvious decision is recorded here, along with the
option picked and why.

## 1. Season coverage: 2020-21 through 2025-26 only

`merged_gw.csv` only carries `position` and `team` columns directly from
the 2020-21 season onward; 2016-17 through 2019-20 only have a numeric
`element` id and would require joining a separate per-season id-list file
just to know a player's position. `xP` is also absent before 2020-21.
Given the per-position modeling requirement and the decision to drop `xP`
as a feature (see #3) but still want the option to inspect it, restricting
to 2020-21+ keeps the schema uniform across all seasons with one
straightforward download path. **Picked:** seasons =
`["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]`.

## 2. Train/test split

**Picked:** train = 2020-21..2023-24 (4 seasons), test = 2024-25, 2025-26
(2 seasons), a strictly chronological split. As of the date this was
built, 2025-26 fixture data in the source repo runs through gameweek 38
with kickoff timestamps into May 2026, i.e. the season is complete in the
cache, so it's a legitimate held-out test set rather than partial/live
data.

## 3. `xP` column: dropped entirely, not shifted

The task allowed either `.shift(1)` or dropping `xP` outright, as long as
the choice was stated. **Picked: drop entirely.** Reasoning: `xP` is
FPL's own model's prediction. Even lagged by one gameweek, it would mean
one row's feature set contains *a different, more sophisticated model's
opinion* about a different player-gameweek — that's a confounding
shortcut for a v0.1 project whose whole point is to build features from
first principles and see if they beat naive baselines. Dropping it keeps
the comparison clean: this model's features are all directly observable
match/fixture history, nothing derived from a competing predictive model.

## 4. Double gameweeks (DGWs)

A player can have two fixtures in one gameweek. `merged_gw.csv` has one
row per fixture. **Picked:** aggregate to one row per
(season, element, round) by *summing* counting stats (points, minutes,
goals, assists, bonus, bps, clean sheets, goals conceded, saves, cards,
ICT sub-components) — matching how FPL itself totals a double gameweek's
score — and averaging `was_home` into a `home_fraction` feature and
fixture difficulty across the gameweek's fixtures. A `num_fixtures`
feature flags DGWs/blank gameweeks to the model.

## 5. Position label cleanup

Two anomalies found in the raw data:
- `"GKP"` appears instead of `"GK"` for some 2021-22 rows (same real
  goalkeepers, just an inconsistent label). **Picked:** normalize
  `GKP -> GK`.
- `"AM"` appears in 2024-25+ for FPL's newer "Manager" fantasy element
  type (real football managers like Guardiola/Arteta scoring FPL points
  as a distinct picks category, always 0 minutes). **Picked:** drop these
  rows entirely — the task specifies GK/DEF/MID/FWD only, and managers
  aren't outfield/keeper players with minutes-driven scoring anyway.

## 6. Missing `starts` column (2020-21, 2021-22)

FPL only added a `starts` column to the data from 2022-23 onward.
**Picked:** for seasons without it, proxy `starts = (minutes > 0)`. It's
an imperfect stand-in for "named in the starting XI" (a very late sub
would count as a "start" here), but it's only used as a rolling *feature*
input (a minutes-availability signal), not the target, so the
approximation is acceptable for v0.1.

## 7. "Played 60+" threshold

Used on the gameweek-aggregated (i.e. post-DGW-summing) minutes figure:
`played_60 = (minutes_this_gameweek >= 60)`. In a double gameweek this can
be true even if the player didn't hit 60 in either individual match, which
somewhat overstates rotation-proof availability in DGWs, but matches how
FPL's own bonus-point-style thresholds are generally discussed
per-gameweek rather than per-fixture, and keeps the feature simple.

## 8. Missing feature values (mostly gameweek 1 of a season)

A player's first gameweek in a season has no history, so all rolling/
expanding features are `NaN` by construction. **Picked:** median-impute
per position using **train-set-only** medians (`fplxp/model.py`), applied
identically to train and test. Baseline predictions for the same rows are
filled with the train set's global mean `total_points`
(`fplxp/backtest.py::add_predictions`). Rows are kept rather than dropped,
so the reported metrics include the (harder, less-informed) start of every
season rather than only the easy in-season rows.

## 9. Model tuning

The v0.1 model beats both baselines decisively on RMSE and MAE, in every
position, with zero hyperparameter tuning. It does **not** uniformly beat
the last-5-mean baseline on Spearman rank correlation (loses narrowly
overall and in 3 of 4 positions — see `reports/backtest.md`). Per
instructions, this is reported as-is rather than tuned away: tuning to
chase Spearman risked either (a) hand-fitting to the two test seasons, or
(b) masking the more informative finding that a plain regression loss
doesn't automatically optimize for within-gameweek ranking, which is the
actually-useful quantity for squad selection. **Picked:** ship v0.1
untuned, state the RMSE-win-but-not-uniform-Spearman-win result plainly,
and note ranking-aware loss functions / the two-stage structure as the
concrete next step.

## 10. Rolling window sizes

**Picked:** player-level windows of 3 and 5 gameweeks (`ROLLING_WINDOWS`
in `fplxp/config.py`), team-level window of 5 gameweeks
(`TEAM_ROLLING_WINDOW`). These are standard, unremarkable choices for
short-memory form signals in this domain, not tuned.

## 11. No `requests` dependency

`requests` is available in this environment, but the task specifies
"standard stack only: pandas, numpy, scikit-learn, matplotlib." **Picked:**
use stdlib `urllib.request` for the handful of file downloads in
`fplxp/data.py` instead, plus `scipy.stats.spearmanr` (a
scikit-learn/pandas-ecosystem dependency, not an extra data source) for
rank correlation.
