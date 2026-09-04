# Judgment calls made without asking

## v0.2 log

## 0. File renamed: QUESTIONS.md -> docs/judgment-calls.md

The v0.2 brief refers to this log as `docs/judgment-calls.md` and assumes
it already exists at that path. v0.1 actually wrote it as `QUESTIONS.md`
at the repo root. **Picked:** `git mv QUESTIONS.md docs/judgment-calls.md`,
keep all v0.1 entries below unchanged, and continue the log at this new
path from here on. `CLAUDE.md` and `README.md` references are updated to
match.

## 1. Blank-gameweek reindex: filled rows are always `is_blank=True, num_fixtures=0`

A gap in a player's (season, round) span can mean two different things:
their team genuinely had no fixture that round, or the source data simply
has no row for them even though their team played (e.g. an unregistered
player, or an idiosyncratic gap in the vaastav dump). Distinguishing these
precisely would require cross-referencing every gap against `fixtures.csv`
per team. **Picked:** collapse both cases to the same treatment --
`is_blank=True`, `num_fixtures=0`, zero counting stats -- since from the
player's own gameweek-scoring perspective they're equivalent (0 points,
no evidence of fixture participation), and the brief's own wording
("blanks are absent rows... become explicit rows with num_fixtures=0")
describes exactly this simpler rule. Noted as a scope-bounding
simplification, not treated as free of edge cases.

## 2. `starts` proxy: `minutes > 0`, uniformly, dropping the real column

The real `starts` column (from 2022-23 onward) is more precise than
`minutes > 0` (it's specifically "named in the XI", so a very late
sub off the bench wouldn't count, but under the `minutes > 0` proxy they
would). **Picked:** use `minutes > 0` for every season regardless of
whether the real column is available, per the brief's explicit
instruction, accepting the precision loss on 2022-23+ in exchange for a
feature that means the same thing in every training/validation/test row
-- v0.1's season-dependent proxy meant train and test rows weren't
measuring quite the same thing, which is a worse problem than losing a
little precision on the late-sub edge case.

## 3. `played_60` from max per-fixture minutes, not summed gameweek minutes

Fixed directly per the brief: two 45-minute double-gameweek appearances
now correctly do NOT count as `played_60`, whereas v0.1's
summed-minutes version incorrectly would have. `n_fixtures_60plus` is
computed and kept in the aggregated table (useful for inspection / could
seed a future two-stage target) but is **not** a model feature -- it's
derived from the current gameweek's own match events, so it's leaky by
the same rule as `minutes` itself.

## 4. Fixture difficulty: min/max/mean instead of a single mean

Implemented directly per the brief. A double gameweek with one trivial
fixture and one brutal one now shows up as
`fixture_difficulty_min != fixture_difficulty_max` rather than being
smoothed into one misleading "medium" number.

## 5. `data/manifest.json`: verify-or-fail, not verify-and-overwrite

On a warm cache, nothing is re-downloaded or re-hashed (reproducibility
without network access). On a *cold* cache re-download, if a manifest
entry already exists for that path and the freshly-downloaded content's
SHA256 doesn't match, `HistoricalCSVDataSource` raises rather than
silently overwriting -- **picked** fail-loud over fail-silent, since a
silently-changed historical CSV upstream (the vaastav repo does
occasionally get corrections pushed) is exactly the kind of thing that
should stop a `make all` run and get a human's attention, not quietly
change what every downstream number means.

## 6. Per-90 rate decomposition: `pts_form{w}` kept out of `FEATURE_COLUMNS`

Splitting "volume" (minutes, played-60 rate, starts rate) from "quality"
(rolling per-90 output rates: goals/assists/bps/bonus/ict/saves/goals-conceded/points)
meant the old plain rolling-mean-of-points feature (`pts_form{w}`, which
conflates both) had to go somewhere. **Picked:** keep computing it, since
`fplxp.baselines`'s "mean of last 5 gameweeks" baseline needs exactly
that number, but exclude it from `FEATURE_COLUMNS` -- the model instead
sees `pts_per90_form{w}` (quality) and `mins_form{w}` /
`played60_rate_form{w}` (volume) as the decomposed inputs. This satisfies
the decomposition requirement without breaking the two required naive
baselines.

## 7. Opponent strength: side-matched via a fixtures+teams join, not per-player

`opponent_team` exists per-fixture in `merged_gw.csv`, but double
gameweeks make a single "the opponent" ill-defined per player-gameweek in
the same way fixture difficulty was. **Picked:** derive
own/opponent attack/defence strength from the same fixtures.csv-based
team-gameweek table already used for goals-for/against and difficulty
(home team gets `strength_*_home` for itself and the away team's
`strength_*_away` as its opponent's, and vice versa), averaged across a
gameweek's fixtures the same way difficulty is. Reuses one join instead
of adding a second, parallel per-player join.

## 8. Market features: `log1p(selected)` + rolling deltas

`selected` (ownership) ranges from 0 to ~9.5 million and is heavily
right-skewed (a handful of Haaland/Salah-tier players dominate the raw
scale). **Picked:** feed the model `log1p(selected)` rather than the raw
count, plus rolling 3/5-gameweek deltas of both price and
log-ownership, all labeled as **market features** in
`reports/backtest.md` (crowd/FPL-pricing-algorithm wisdom, not football
signal computed from match events).

## 9. GW1 carryover matched by player *name*, not `element` id

FPL's `element` id is **not stable across seasons** -- confirmed directly
(e.g. Harry Kane is element 357 in 2021-22, 427 in 2022-23, 500 in
2023-24). Matching prior-season data therefore has to go by name, which
is itself inconsistent between seasons (`"Ben White"` vs
`"Benjamin White"`, `"Eddie Nketiah"` vs `"Edward Nketiah"`,
`"Gabriel Magalhães"` vs `"Gabriel dos Santos Magalhães"`). **Picked:**
exact-string name matching, accepting the resulting gap. Measured
`has_prior_season` coverage at each season's gameweek 1: **75.6%
(2021-22), 67.2% (2022-23), 71.9% (2023-24)**. The remaining ~25-33%
(genuine debutants, plus name-format mismatches for returning players)
fall back to ordinary train-median imputation with `has_prior_season=0`,
same as v0.1. A real fix would need a stable cross-season player-id
crosswalk that this data source doesn't provide; out of scope for v0.2.

## 10. Poisson-loss models: clip the *training* target at 0, evaluate unclipped

`total_points` can be slightly negative (own goals, red cards -- min
observed: -4). Poisson deviance requires `y >= 0`, so
`HistGradientBoostingRegressor(loss="poisson")` would raise on the raw
target. **Picked:** `y.clip(lower=0)` only for fitting the poisson-loss
variants (affects ~0.5% of training rows, e.g. -4 -> 0, -1 -> 0); every
evaluation everywhere in the project still scores predictions against the
true, unclipped `total_points`. This only touches training labels for one
specific loss function, never the evaluation target.

## 11. Model selection rule fixed before Phase 3 ran

See `docs/evaluation-plan.md`'s "Pre-committed selection rule" section,
written and committed before any of the four variants were trained. The
rule (average rank across RMSE/MAE/Spearman/realised-points-of-top-11 on
validation, `minutes > 0` rows, ties toward simplicity) is mechanical
specifically so the model choice can't be justified after the fact by
whichever number happened to look best.

## 12. Final model refit on train+validation before the one frozen test run

Phase 3 selects the model *type* using train-fit/validation-scored
comparisons only. **Picked:** for the single Phase 4 test evaluation, the
selected variant is refit on train+validation combined (2020-21..2023-24)
before scoring once on 2024-25+2025-26 -- standard practice once a model
type is chosen, so the validation season's data isn't simply thrown away
for the number that actually matters. The validation *comparison itself*
never touches test data either way.

## 13. Omitted signal: set-piece / penalty-taker status

None of `merged_gw.csv`, `fixtures.csv`, or `teams.csv` records who takes
a team's penalties, corners, or free kicks. This is a large real signal
for goal/assist involvement (a nominal penalty taker's expected points
are structurally higher than a teammate with similar general output) that
this data source simply doesn't carry. Not approximated in v0.2; noted
here as a known gap rather than silently absent.

## 14. `baseline_xp`'s per-gameweek Spearman line has real gaps (not a bug) --
**UPDATE (review round): the gap is far bigger than first measured**

`reports/spearman_by_gameweek.png` shows `Baseline: FPL xP` missing
gameweeks in the test seasons. Originally checked against only 2024-25 and
found 2 affected gameweeks (GW32, GW34), each with `xP` a single constant
value across every row (`nunique=1`) -- a source data-quality gap, not a
plotting bug. **After building the cross-predictor gameweek alignment
(#22 below) and re-measuring across both full test seasons, the real
scope is much larger: 30 of 76 test gameweeks (39%) have degenerate
`xP`**, overwhelmingly concentrated in 2025-26 gameweeks 7 through 37 --
essentially the entire back half of that season, presumably because the
upstream dump caught `xP` before it had been backfilled for those rounds
(match results, minutes, and points for the same gameweeks are populated
normally -- it's specifically the `xP` column that's stale/missing). This
means `baseline_xp`'s reported Spearman is only ever averaged over the
44 gameweeks common to every predictor (`n_gw_used` /
`n_gw_dropped` in `reports/backtest.md`), which happen to skew toward
the earlier, more-established part of each season. No fix applied to
the source data (can't backfill what FPL/vaastav never published); the
mitigation is #22's alignment, so at least every predictor's average is
over the same 44 gameweeks rather than each silently choosing its own
easier subset.

## 15. GitHub Pages served from a `gh-pages` branch, not `docs/`

The brief allows either. **Picked: `gh-pages` branch.** This repo's
`docs/` already holds project decision records
(`judgment-calls.md`, `evaluation-plan.md`, `leakage-audit.md`) --
pointing GitHub Pages at `docs/` would either publish those alongside the
app (confusing: a phone visitor doesn't want the leakage audit) or force
renaming the site's own directory away from the `site/` path the brief
names explicitly. A `gh-pages` branch keeps the published site
(`site/index.html` at its root) completely separate from the source
tree's own docs, and is regenerated wholesale on every run (see #16) --
nothing there is ever hand-edited, so treating it as fully disposable
generated output is safe.

## 16. `gh-pages` branch is force-pushed wholesale, not diffed/merged

`.github/workflows/weekly-refresh.yml`'s publish step replaces the entire
`gh-pages` branch content with the current `site/` directory and
force-pushes. **Picked** over incrementally updating it because the
branch's *only* content is generated output (`site/` verbatim) with no
history worth preserving commit-by-commit; force-pushing a fully
regenerated branch is simpler and can't drift from `site/`'s actual
current state. This is the one place in the project a force-push is
used, and it's scoped to a branch that never carries hand-written commits
(see the Git Safety Protocol note in CLAUDE.md as it applies to the
`gh-pages` branch specifically).

## 17. Live-season predictions are out of scope; `data.py` is swappable regardless

Per the brief: predicting the *current, in-progress* season would need
FPL's own `bootstrap-static` (current prices/ownership/teams) and
`fixtures` endpoints, not the historical vaastav CSV dumps this project
uses, and those live endpoints aren't reachable from this project's usual
sandboxed dev environment (only the weekly GitHub Actions runner has the
egress for it). **Picked:** don't implement it for v0.2, but do restructure
`fplxp/data.py` around a `DataSource` interface first
(`HistoricalCSVDataSource` / `LiveFPLDataSource`) specifically so that
gap has one obvious, already-proven-swappable place to fill in later,
rather than leaving live-data support as a hypothetical refactor.

## Review round (external review of the v0.2 draft)

A review of the v0.2 draft raised nine issues, addressed here.

## 18. Blank vs. data-gap distinction: correct in principle, empirically a no-op

#1's original text declined to distinguish "team had no fixture" from
"player row missing from the dump while the team played" because it
would need a per-team fixture cross-reference. The review pointed out
`_team_fixture_features` already builds exactly that table (#7), so
there was no reason not to reuse it. **Fixed:** `build_feature_table` now
merges `team_id` before reindexing, builds a
`{(season, team_id, round): num_fixtures}` schedule lookup from the real
fixture list, and `_reindex_blank_gameweeks` uses it: a filled gap where
the team's schedule shows zero fixtures gets `is_blank=True,
num_fixtures=0` (a true blank); a filled gap where the team DID play
gets `is_blank=False` and the team's real fixture count (a data gap --
still zero counting stats, since there's no evidence of what an omitted
player actually did, but `num_fixtures=0` would have been factually
wrong). Checked empirically after the fix, across all 6 seasons: **zero
data-gap rows found** -- every gap in this data source's per-player round
sequence corresponds to a genuine team-wide blank. The distinction is
real and tested (`tests/test_features.py::test_data_gap_is_not_treated_as_a_blank_when_team_actually_played`),
kept for correctness and in case future seasons' data behaves
differently, but it didn't change any actual v0.2 numbers.

## 19. Third metric cut: excluding `is_blank` rows

The reindex introduced in Phase 1 gives every blank gameweek an explicit
row with `total_points=0` and a feature (`num_fixtures=0`) that trivially
predicts it -- free RMSE/MAE improvement that v0.1 never had access to,
since it never had blank rows at all. **Fixed:** `fplxp/metrics.py` now
reports every metric three ways -- all rows, excluding `is_blank` rows,
and `minutes > 0` only -- and `reports/backtest.md` states plainly which
cut is (and isn't) comparable to v0.1: the "excluding is_blank" column
is, the "all rows" column is not.

## 20. Cross-season player matching: switched from `name` to `code`

Prompted by the review to check whether `players_raw.csv` exposes a
stable id. It does: `code` (verified directly -- Harry Kane is `code`
78830 in 2021-22, 2022-23, *and* 2023-24, while his `element`/`id` is
357/427/500 across those same three seasons). `code` also resolves every
name-format mismatch found earlier (`"Ben White"` vs `"Benjamin White"`,
`"Eddie Nketiah"` vs `"Edward Nketiah"`, `"Gabriel Magalhães"` vs
`"Gabriel dos Santos Magalhães"`, `"Mohamed Elneny"` vs `"Mohamed Naser
El Sayed Elneny"` -- all four verified to share one `code` across
seasons despite the differing name strings). **Fixed:** `players_raw.csv`
added to `HistoricalCSVDataSource`/`DataSource`, `code` merged onto every
row, GW1 carryover matching switched from `name` to `code`. Measured
`has_prior_season` coverage at each season's gameweek 1 after the
switch: **76.7% (2021-22), 75.6% (2022-23), 73.4% (2023-24), 77.9%
(2024-25), 74.2% (2025-26)** -- up from name-matching's 75.6% / 67.2% /
71.9% for the first three (2022-23 gained +8.4pp; 2021-22 and 2023-24
were already close, since most of that season's transfers happened not
to hit a name-format mismatch). The remaining ~23-27% is not a matching
failure: it's genuine debutants (promoted-club players, first PL season)
who have no prior season to carry over, which is the correct outcome for
them, not a gap to close further.

## 21. Per-90 rate: explicit zero-fill for genuine zero-minute windows

`goals_per90_form{w}` etc. were computed as `rolling_sum(stat) /
rolling_sum(minutes) * 90`, `NaN` whenever the denominator was 0 --
correct for "no history yet" (first-ever gameweek), but also incorrectly
covering "real history exists, all of it 0 minutes" (an unused sub, or a
long-term injury). Both landed as `NaN` and were then train-median-imputed
at model-fit time, which describes an out-injured player as
*league-average quality* rather than "no evidence of output" -- a
materially different and misleading claim, one the volume-side features
(`mins_form`, `played60_rate_form`) don't make (they correctly show ~0
for the same player, since they're plain rolling means, not ratios).
**Fixed:** the three cases are now handled separately in
`_add_player_rolling`: a real, positive rolling-minutes sum gives a real
per-90 rate; a rolling-minutes sum of exactly 0 (real zero-minute
history) gives `0.0`, not `NaN`; only a `NaN` rolling-minutes sum (no
history at all yet) stays `NaN` for prior-season-carryover/median
imputation. Confirmed via `tests/test_no_leakage.py` still passing (the
per-90 leakage properties are unaffected) and by the NaN count on a
single season dropping from 12,778 to 778 (exactly the "true first
gameweek" count) after the fix.

## 22. Aligned Spearman: same gameweek set for every predictor being compared

`mean_gw_spearman` computed each predictor's average independently,
silently skipping whichever gameweeks were degenerate *for that
predictor* (e.g. `baseline_xp` on the two gameweeks where `xP` is a
constant value across the whole slate -- #14). Averaging two predictors
over different gameweek sets isn't a fair comparison. **Fixed:**
`fplxp/metrics.py::_common_valid_gameweeks` computes the intersection of
gameweeks where the target AND every predictor being compared has >= 2
distinct values, and every Spearman average in `point_metrics_multi` uses
that common set. `n_gw_dropped` is reported alongside each metric so the
count is visible, not silently absorbed.

## 23. `pts_form` ablation: validation-only experiment, not a silent revert

The volume/per-90-quality decomposition (v0.2 #6 above) dropped the raw
`pts_form{w}` rolling-mean-of-points feature from the model input on the
theory that per-90 quality + volume span the same information. The
review pointed out this isn't obviously true: clean sheets, bonus, and
BPS are position-dependent step-function thresholds of the underlying
counting stats, not linear in a per-90 rate, so the decomposition could
lose signal a plain rolling mean captured. **Fixed:**
`fplxp/model_selection.py::run_pts_form_ablation` fits the selected
variant with and without `pts_form3`/`pts_form5` added back, on
validation only, using the same rank-average rule as the main comparison
restricted to those two rows, and `reports/validation_comparison.md`
states which won. The feature is adopted into `FEATURE_COLUMNS` only if
it wins that ablation -- see the report for the actual outcome on this
run.

## 24. Constrained top-11: fixed formation, greedy fill, real squad rules

The unconstrained top-k metric has no budget, no per-club cap, and no
formation quotas, so it can (and does) pick five players from the same
club -- overstating what real squad selection achieves, and only
actually representing the true decision at k=1 (captaincy, where you
already own your squad and it really is an unconstrained "best predicted
score today" choice). **Picked:** add a second, constrained metric
(`fplxp/metrics.py::constrained_squad_metrics`) rather than only
softening the framing (though the framing was softened too, in
`docs/evaluation-plan.md` and `reports/backtest.md`): one XI per
gameweek, built greedily (by predicted value, per position, in a fixed
processing order) subject to a GBP100m budget, a max of 3 players per
club, and **one fixed formation** (1 GK / 4 DEF / 4 MID / 2 FWD) rather
than searching all valid formations each gameweek. Both simplifications
are deliberate scope bounds, not oversights: a full 36-formation search
plus true global-optimum selection within each is an ILP problem, and a
greedy heuristic within one common, valid formation is enough to show
whether budget/club constraints change the picture without building a
solver. Documented as a heuristic, not an exact solve, everywhere it's
reported.

## 25. Bootstrap CI on the frozen test-set comparison

A single point estimate from the one frozen Phase 4 run ("RMSE 2.926 vs.
3.068") doesn't say whether that's a real gap or gameweek-to-gameweek
noise. **Fixed:** `fplxp/metrics.py::bootstrap_paired_diff` computes the
metric (RMSE, or Spearman) per gameweek for the model and the relevant
baseline, pairs them by gameweek, and bootstraps (2000 resamples,
gameweeks resampled with replacement, seeded from `config.RANDOM_STATE`)
the mean paired difference to get a standard error and 95% CI, reported
in `reports/backtest.md`'s verdict alongside the raw numbers.

## 26. Leakage-audit reasoning correction

A prior version of `docs/leakage-audit.md` §4 argued that v0.1 losing
narrowly on Spearman while winning on RMSE/MAE was itself evidence of a
clean pipeline. The review correctly identified this as invalid: partial
leakage confined to the minutes/availability signal specifically would
produce the identical pattern (big RMSE gains from correctly zeroing
non-players, no Spearman gain among players who did play), so the
pattern can't distinguish the two explanations. §4 has been rewritten to
retract that inference and rely only on the actual evidence (the
line-by-line shift-before-rolling code audit in §1, and the synthetic
spike test in §2) -- see that file for the corrected text.

---

# v0.1 log (original entries, unchanged)

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
