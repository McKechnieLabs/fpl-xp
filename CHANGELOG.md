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
  prior-season GW1 carryover (matched by player name, ~67-76% coverage).
- **Metrics**: dropped `scipy` (Spearman now via
  `DataFrame.corr(method="spearman")`). Every metric now reported twice
  (all rows vs. `minutes > 0`). Added precision@k / realised-points-of-
  top-k (k in {1, 11, 15}) against an oracle, and predicted-decile
  calibration.
- **Model**: added `HistGradientBoostingRegressor(loss="poisson")`, a
  two-stage `P(played_60) x E[points|played]` model, and a pooled
  cross-position model, compared against the v0.1-style
  `GradientBoostingRegressor` on validation only via a rule fixed in
  `docs/evaluation-plan.md` *before* any variant was run. On the real
  data, the v0.1-style model won.
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

### Result

On the frozen test seasons, the v0.2 model beats every baseline
(including FPL's own `xP`) on RMSE/MAE in every position, with no
hyperparameter tuning. It does **not** beat FPL's `xP` on Spearman rank
correlation or realised points of a top-11 pick among players who
actually played -- reported plainly rather than tuned away. Full table:
`reports/backtest.md` and the [live site](https://mckechnielabs.github.io/fpl-xp/).

## v0.1

Initial expected-points model: per-position `GradientBoostingRegressor`
on rolling-window form, minutes-availability, home/away, fixture
difficulty, and team-level rolling goals for/against. Chronological
train (2020-21..2023-24) / test (2024-25+2025-26) split. Beat both naive
baselines (last-5 mean, season-to-date mean) on RMSE/MAE in every
position; did not uniformly beat the last-5-mean baseline on Spearman
rank correlation. See `reports/backtest_v0.1.md` and `docs/judgment-calls.md`'s
v0.1 log section.
