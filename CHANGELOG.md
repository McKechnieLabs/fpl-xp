# Changelog

## v0.2

**Phase 0 leakage audit: no leakage found**, in either v0.1 or the v0.2
rebuild. Every rolling/expanding feature expression was read directly
against the code, a synthetic 100-point-spike test and a
`|Spearman| > 0.9`-with-target check were added, and a hand-curated,
independently-written whitelist of legitimately pre-deadline "same-round"
columns was locked in as a test. Full writeup: `docs/leakage-audit.md`.
No corrective fix was needed, so v0.1's original report is preserved
unmodified at `reports/backtest_v0.1.md`.

### Changed

- **Split**: train/validation/test replaces train/test --
  train = 2020-21..2022-23, validation = 2023-24 (Phase 3 model
  selection only), test = 2024-25+2025-26 (frozen, touched exactly once,
  at the end).
- **Data layer**: gap gameweeks (blanks) are now explicit reindexed rows
  (`is_blank`, `num_fixtures=0`) instead of absent rows, so rolling
  windows span calendar gameweeks. `starts` proxy is now `minutes > 0`
  uniformly across every season. `played_60` now derives from max
  per-fixture minutes, not summed gameweek minutes. Fixture difficulty is
  now min/max/mean across a gameweek's fixtures. Every download is
  recorded in `data/manifest.json` (URL/date/SHA256); the data source is
  now a swappable `DataSource` interface.
- **Features**: rolling stats split into volume (minutes/availability)
  and per-90 output-rate (quality) features. Added opponent
  attack/defence strength (side-matched to home/away), market features
  (price, log-ownership, rolling deltas -- labelled distinctly), explicit
  missingness flags (`gws_of_history`, `is_first_gw_of_season`), and
  prior-season GW1 carryover (matched by FPL's persistent `code`, ~73-78%
  coverage -- see review-round fixes below for why `code` and not `name`).
- **Metrics**: dropped `scipy` (Spearman now via
  `DataFrame.corr(method="spearman")`). Every metric now reported three
  ways (all rows / excluding blanks / `minutes > 0`). Added unconstrained
  precision@k / realised-points-of-top-k (k in {1, 11, 15}) against an
  oracle, a constrained top-11 (budget + formation + 3-per-club),
  predicted-decile calibration, and a paired bootstrap CI on the
  model-vs-best-baseline gap.
- **Model**: added `HistGradientBoostingRegressor(loss="poisson")`, a
  two-stage `P(played_60) x E[points|played]` model, and a pooled
  cross-position model, compared against the v0.1-style
  `GradientBoostingRegressor` on validation only via a rule fixed in
  `docs/evaluation-plan.md` *before* any variant was run. On the real
  data, **the pooled model won** (mean rank 1.5 vs. 1.75/3.0/3.75).
- **Baselines**: added `baseline_xp` (FPL's own published `xP`) as a
  third, non-naive benchmark, prominently reported.
- Pinned exact dependency versions in `requirements.txt`; added
  `Makefile` (`make all` reproduces every number in `reports/` from a
  cold checkout: test -> model-selection -> backtest, in order).
- Renamed `QUESTIONS.md` -> `docs/judgment-calls.md` (continuing the same
  log) to match this phase's expected path.

### Added

- `docs/evaluation-plan.md` -- the frozen metric set and pre-committed
  model-selection rule, written before any Phase 3 result existed.
- `docs/leakage-audit.md` -- the Phase 0 audit.
- Static site (`site/`): a single dependency-free `index.html` rendering
  the headline table, minutes>0 split, per-position breakdown, top-k
  results, calibration/RMSE/Spearman plots, and a plain-English
  "what this model does and doesn't do" section. Mobile-first, PWA
  manifest + icons.
- `.github/workflows/weekly-refresh.yml` -- weekly cron that reruns the
  pipeline, commits regenerated reports/site data to `main`, and
  publishes `site/` to `gh-pages`.

### Review round (fixes made in response to an external review of the v0.2 draft)

Before the frozen test run, a review of the v0.2 draft raised nine
issues, all addressed -- see `docs/judgment-calls.md`'s "Review round"
section (#18-26) for the full writeup of each:

1. Blank-gameweek reindexing now distinguishes a true team-wide blank
   from a data gap where the team played but the row was simply missing
   (reusing the team-fixture table already built for #7) -- empirically a
   no-op on this data source (0 data-gap rows found), but now correct in
   principle and tested either way.
2. Every metric gained a third cut excluding synthetic blank rows, so the
   headline numbers are comparable to v0.1 (which never had blank rows).
3. `docs/leakage-audit.md` §4's "losing on Spearman corroborates a clean
   pipeline" argument was retracted as invalid (a partial minutes-only
   leak would produce the identical pattern) and rewritten to rely only
   on the actual evidence (the code audit and the spike test).
4. GW1 carryover switched from name-matching to `players_raw.csv`'s
   `code` (FPL's real persistent player id) after confirming `element`
   resets every season but `code` doesn't.
5. Per-90 rate features now correctly zero-fill for real zero-minute
   history (an unused sub, a long-term injury) instead of leaving `NaN`
   to be median-imputed as league-average quality.
6. A validation-only ablation now checks whether adding the raw
   `pts_form3`/`pts_form5` columns back improves the winning model
   (only applicable to `PositionModels`-based winners; not run this
   round since `pooled` won).
7. Added a constrained top-11 metric (fixed formation, budget, 3-per-club
   cap) alongside the unconstrained top-k, and softened the "decision-
   relevant" framing of the unconstrained k=11/15 numbers.
8. Added a paired bootstrap CI (2000 resamples over gameweeks) on the
   model-vs-best-baseline RMSE and Spearman gaps.
9. Spearman is now averaged only over gameweeks where every predictor
   being compared has a defined correlation -- which surfaced a much
   larger `xP` data-quality gap than previously known (30 of 76 test
   gameweeks, concentrated in the back half of 2025-26, not the 2
   gameweeks first noted).

### Result

On the frozen test seasons, the v0.2 model (the `pooled` variant) beats
every baseline (including FPL's own `xP`) on RMSE/MAE in every position,
with no hyperparameter tuning -- a paired bootstrap puts that RMSE gap at
-0.150 points (95% CI [-0.194, -0.110], real, not noise). It does **not**
beat FPL's `xP` on Spearman rank correlation among players who played
(bootstrap mean diff -0.204, 95% CI [-0.226, -0.180], also real) or on a
realistic constrained-squad's realised points (53.6 vs. 61.0 pts/gameweek
for `xP`, vs. 132.7 for a perfect-hindsight oracle) -- reported plainly
rather than tuned away. Full table: `reports/backtest.md` and the
[live site](https://mckechnielabs.github.io/fpl-xp/).

## v0.1

Initial expected-points model: per-position `GradientBoostingRegressor`
on rolling-window form, minutes-availability, home/away, fixture
difficulty, and team-level rolling goals for/against. Chronological
train (2020-21..2023-24) / test (2024-25+2025-26) split. Beat both naive
baselines (last-5 mean, season-to-date mean) on RMSE/MAE in every
position; did not uniformly beat the last-5-mean baseline on Spearman
rank correlation. See `reports/backtest_v0.1.md` and `docs/judgment-calls.md`'s
v0.1 log section.
