# fpl-xp

An expected-points (xP) model for Fantasy Premier League: predicts a
player's `total_points` for a single upcoming gameweek from data available
before that gameweek's deadline (a **nowcast** for that one gameweek, not a
season-long projection), and checks the prediction against naive baselines
*and* FPL's own published `xP` rather than assuming a from-scratch model is
automatically better.

**Live results:** [mckechnielabs.github.io/fpl-xp](https://mckechnielabs.github.io/fpl-xp/)
(mobile-friendly; add to your phone's home screen for a full-screen app).
Published from the `gh-pages` branch -- if that link 404s, GitHub Pages
hasn't been switched on for this repo yet: an admin needs to visit
**Settings -> Pages -> Build and deployment -> Source: Deploy from a
branch -> `gh-pages` / `/(root)`** once. The `gh-pages` branch itself
already has the current site pushed either way.

## Headline result (test seasons: 2024-25 + 2025-26, frozen, scored once)

All positions pooled. Full breakdown, per-position numbers, calibration
plots, and top-k decision metrics in [`reports/backtest.md`](reports/backtest.md)
and on the [live site](https://mckechnielabs.github.io/fpl-xp/).

| Method | RMSE (all) | RMSE (played) | Spearman (all) | Spearman (played) |
|---|---|---|---|---|
| **Model (v0.2, `gbr`)** | **1.941** | **2.926** | 0.701 | 0.367 |
| Baseline: FPL's own xP | 2.133 | 3.313 | 0.752 | **0.581** |
| Baseline: last-5 mean | 2.138 | 3.183 | 0.698 | 0.291 |
| Baseline: season-to-date mean | 2.082 | 3.068 | 0.670 | 0.312 |

**"played" = restricted to `minutes > 0` rows** -- see "the two skills"
below for why that split, not the pooled column, is the one that matters
for squad selection.

**Read plainly:** the model beats every baseline, including FPL's own xP,
on point-estimate accuracy (RMSE/MAE), in every position, with no
hyperparameter tuning. It does **not** beat FPL's xP on rank correlation
among players who actually played, or on realised points of a top-11 pick
(57.0 pts/gameweek for the model vs. 60.2 for xP vs. 138.6 for a
perfect-hindsight oracle). That gap is the honest v0.2 headline, reported
as-is rather than tuned away -- see `docs/evaluation-plan.md`'s
pre-committed rules for why no metric was cherry-picked after the fact.

## Quick start

Requires Python 3.10+.

```bash
git clone <this repo>
cd fpl-xp
pip install -r requirements.txt

make all
# equivalent to:
#   python -m pytest                  # unit tests, including the leakage guarantee
#   python -m fplxp.model_selection   # Phase 3: picks a model variant on validation only
#   python -m fplxp.backtest          # Phase 4: the one frozen test-set run
```

The first run downloads and caches ~30MB of historical CSVs from
`github.com/vaastav/Fantasy-Premier-League` into `data/raw/` (plain HTTPS
GETs, no API key or live FPL API involved -- see `data/manifest.json` for
exactly what was pulled from where and when), then re-uses that cache.
`make all` writes:

- `reports/backtest.md` -- the frozen metric table, verdict, and links to
  the plots below (v0.1's original report is preserved at
  `reports/backtest_v0.1.md`, unmodified)
- `reports/validation_comparison.md` -- the Phase 3 model-variant
  comparison that picked the model used in the final report
- `reports/*.png` -- predicted-vs-actual, RMSE by position, Spearman by
  gameweek, calibration, and top-k realised points
- `site/data/*.json` + `site/plots/*.png` -- what the static site reads

## How it works

### Data (`fplxp/data.py`)

Three files per season, fetched through a `DataSource` interface
(`HistoricalCSVDataSource` is the only implementation used; a
`LiveFPLDataSource` stub documents where a future live-endpoint source
would plug in -- see `docs/judgment-calls.md` #17) from
`raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/<season>/`:

| file | what it gives us |
|---|---|
| `gws/merged_gw.csv` | one row per player per fixture: minutes, points, goals, assists, bps, ICT, xP, price, ownership, etc. |
| `fixtures.csv` | one row per match: teams, scores, FPL's pre-match difficulty rating for each side |
| `teams.csv` | team id -> name and attack/defence strength ratings, for that season |

Seasons used: 2020-21 through 2025-26 (see `docs/judgment-calls.md` for
why earlier seasons are excluded). Files are cached under
`data/raw/<season>/` (git-ignored) and recorded in `data/manifest.json`
(URL, download date, SHA256) -- a cold re-download that doesn't match a
prior manifest entry raises rather than silently training on changed
upstream data.

### Features (`fplxp/features.py`)

1. Per-fixture rows are aggregated to one row per (season, player,
   gameweek), summing counting stats across double-gameweek fixtures the
   way FPL itself totals a DGW score. `played_60` is derived from the
   *max* per-fixture minutes, not the summed total (two 45-minute DGW
   appearances don't count as playing 60+).
2. Each player is reindexed to a complete (season, round) grid over their
   active span -- a gap (blank gameweek) becomes an explicit
   `is_blank=True` row with zero stats, so rolling windows span calendar
   gameweeks rather than silently skipping past blanks.
3. Team-level goals-for/against, fixture difficulty (min/max/mean across
   a gameweek's fixtures), and opponent attack/defence strength
   (side-matched to home/away) are joined from `fixtures.csv`/`teams.csv`.
4. Player-level rolling features are split into **volume** (minutes,
   played-60 rate, starts rate -- how much they play) and **per-90 output
   rate** (goals/assists/bps/bonus/ICT/saves/goals-conceded/points per 90
   minutes -- how good they are when they do play), over trailing 3- and
   5-gameweek windows, plus a season-to-date expanding average.
5. Market features (price, log-ownership, and their rolling deltas) are
   included and labelled distinctly -- crowd/FPL-pricing wisdom, not
   football signal computed from match events.
6. A player's gameweek-1-of-season row is backfilled from their
   *prior season's* per-90 rates and average minutes (matched by name --
   FPL's `element` id is not stable across seasons) where a prior season
   exists, rather than left to positional-median imputation.

**Every rolling/expanding computation is `.shift(1)`'d within its
(season, player) or (season, team) group *before* the window is
applied**, so a gameweek's features are provably built only from
strictly earlier gameweeks. `xP` (FPL's own same-gameweek expected
points) is dropped entirely rather than used as a feature, but kept as
the `baseline_xp` benchmark. `tests/test_no_leakage.py` checks this
structurally (no leaky raw column, and no same-round column outside a
hand-curated whitelist, can appear in the feature list) and mechanically
(a synthetic 100-point spike gameweek must not appear in that gameweek's
own features; no feature may exceed `|Spearman| = 0.9` with the target on
train). See `docs/leakage-audit.md` for the full Phase 0 audit --
**conclusion: no leakage found**, in either v0.1 or the v0.2 rebuild.

### Baselines (`fplxp/baselines.py`)

Three predictors the model is compared against: the player's mean points
over their last 5 gameweeks, their season-to-date mean, and FPL's own
pre-match `xP` for that gameweek. The first two reuse the same shifted
rolling columns the feature pipeline computes, so they're held to the
identical no-lookahead rule as the model.

### Model (`fplxp/model.py`, `fplxp/model_selection.py`)

Four variants are compared **on the validation season (2023-24) only**,
per the pre-committed rule in `docs/evaluation-plan.md` (average rank
across RMSE/MAE/Spearman/realised-points-of-top-11 on `minutes > 0` rows,
ties toward simplicity): a v0.1-style per-position
`GradientBoostingRegressor`, a per-position
`HistGradientBoostingRegressor(loss="poisson")`, a two-stage
`P(plays 60+) x points | played` model, and a pooled cross-position model.
On the real data, the **v0.1-style `gbr` variant won** -- not because it
was assumed to, but because the mechanical rule picked it; see
`reports/validation_comparison.md`.

### Evaluation (`fplxp/metrics.py`, `fplxp/backtest.py`)

Chronological split (`fplxp/config.py`): train 2020-21..2022-23,
validate 2023-24, **test 2024-25+2025-26 -- touched exactly once**, for
the numbers in `reports/backtest.md`. No shuffled cross-validation is
used anywhere. Every metric is reported twice: over all rows, and
restricted to `minutes > 0` -- see "headline result" above for why. Also
reported: precision@k / realised-points-of-top-k for k in {1, 11, 15}
against an oracle, and predicted-decile calibration per position.

## Static site (`site/`)

`site/index.html` is a single, dependency-free page (no build step, no
framework) that fetches `site/data/metrics.json` and
`site/data/predictions.json` client-side and renders the headline table,
the minutes>0 split, a per-position breakdown, top-k results, the plots,
and a plain-English "what this model does and doesn't do" section.
Mobile-first (single column, tables collapse to stacked cards under
600px, tap targets >=44px), with a PWA manifest + icons so it can be
added to a phone home screen. `.github/workflows/weekly-refresh.yml` runs
`make all` on a weekly cron, commits the regenerated `reports/`/`site/data/`
back to `main`, and force-publishes `site/` to the `gh-pages` branch
(that branch is 100% generated output -- see `docs/judgment-calls.md`
#15-16 for why `gh-pages` was chosen over serving from `docs/`).

## Repo layout

```
fplxp/
  config.py          seasons, paths, train/val/test split, seeds
  data.py            DataSource interface + historical-CSV download/cache + manifest
  features.py        leakage-free feature table (blank-GW reindex, per-90 rates,
                      opponent strength, market features, GW1 carryover)
  baselines.py        last-5 mean, season-to-date mean, FPL xP
  model.py            4 model variants (gbr, hgb_poisson, two-stage, pooled)
  metrics.py           frozen metric set: RMSE/MAE/Spearman x2, top-k, calibration
  evaluate.py          shared scoring logic (validation comparison + final test run)
  model_selection.py   Phase 3 entry point (`python -m fplxp.model_selection`)
  backtest.py           Phase 4 entry point (`python -m fplxp.backtest`)
  report_utils.py       tiny markdown-table helper (no tabulate dependency)
tests/
  test_no_leakage.py    the load-bearing guarantee -- keep this passing
  test_features.py, test_baselines.py, test_metrics.py, test_model.py
data/
  manifest.json        URL/date/SHA256 of every cached CSV (committed)
  raw/                 cached CSVs (git-ignored)
reports/
  backtest.md, backtest_v0.1.md, *.png, validation_comparison.md, chosen_model.json
site/
  index.html, manifest.json, icon-*.png, apple-touch-icon.png
  data/metrics.json, data/predictions.json, plots/*.png
docs/
  judgment-calls.md    every non-obvious decision, what was picked, and why
  evaluation-plan.md   the frozen metric set + model-selection rule (written pre-Phase-3)
  leakage-audit.md     the Phase 0 audit writeup
.github/workflows/
  weekly-refresh.yml   cron: rerun pipeline, commit reports/site, publish gh-pages
CLAUDE.md              constraints for future agent sessions in this repo
CHANGELOG.md           what changed between v0.1 and v0.2
```
