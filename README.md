# fpl-xp

An expected-points (xP) model for Fantasy Premier League: predicts a
player's `total_points` for a single upcoming gameweek from data available
before that gameweek's deadline, and checks the prediction against two
naive baselines rather than assuming a fancier model is automatically
better.

## Quick start

Requires Python 3.10+.

```bash
git clone <this repo>
cd fpl-xp
pip install -r requirements.txt

python -m pytest              # unit tests, including the leakage guarantee
python -m fplxp.backtest      # builds features, trains, evaluates, writes reports/
```

`python -m fplxp.backtest` downloads and caches ~30MB of historical CSVs
from `github.com/vaastav/Fantasy-Premier-League` into `data/raw/` the
first time it runs (plain HTTPS GETs, no API key or live FPL API
involved), then re-uses that cache on subsequent runs. It prints a metrics
table to stdout and writes:

- `reports/backtest.md` — the metrics table, verdict, and links to the
  plots below
- `reports/predicted_vs_actual.png` — model calibration, one panel per
  position
- `reports/rmse_by_position.png` — model vs. both baselines, RMSE, by
  position
- `reports/spearman_by_gameweek.png` — model vs. both baselines, rank
  correlation, over the whole test window

## How it works

### Data (`fplxp/data.py`)

Three files per season, fetched from
`raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/<season>/`:

| file | what it gives us |
|---|---|
| `gws/merged_gw.csv` | one row per player per fixture: minutes, points, goals, assists, bps, ICT, etc. |
| `fixtures.csv` | one row per match: teams, scores, FPL's pre-match difficulty rating for each side |
| `teams.csv` | team id -> name for that season (ids are not stable across seasons) |

Seasons used: 2020-21 through 2025-26 (see `QUESTIONS.md` #1 for why
earlier seasons are excluded). Files are cached under `data/raw/<season>/`
and git-ignored; delete that directory to force a re-download.

### Features (`fplxp/features.py`)

1. Per-fixture rows are aggregated to one row per (season, player,
   gameweek), summing counting stats across double-gameweek fixtures the
   way FPL itself totals a DGW score.
2. Team-level goals-for/goals-against are computed per (season, team,
   gameweek) from `fixtures.csv`, then rolled into a trailing 5-gameweek
   average.
3. Player-level rolling form (points, minutes, played-60 rate, ICT,
   goals, assists, bps, bonus, clean sheets, goals conceded, saves,
   starts) is computed over trailing 3- and 5-gameweek windows, plus a
   season-to-date expanding average.
4. Fixture difficulty (FPL's pre-set rating), home/away, scheduled
   fixture count, and price are joined in directly — these are known
   before kickoff, so there's nothing to shift.

**Every rolling/expanding computation is `.shift(1)`'d within its
(season, player) or (season, team) group *before* the window is applied**,
so a gameweek's features are provably built only from strictly earlier
gameweeks. `xP` (FPL's own same-gameweek expected points) is dropped
entirely rather than used as a feature — see `QUESTIONS.md` #3.
`tests/test_no_leakage.py` checks this both structurally (no leaky raw
column can appear in the feature list) and mechanically (a synthetic
player's gameweek-4 rolling feature is asserted to be unchanged when
gameweek 4's own points are mutated to an outlier value).

### Baselines (`fplxp/baselines.py`)

Two naive predictors the model has to beat:
(a) the player's mean points over their last 5 gameweeks,
(b) the player's season-to-date mean points.
Both reuse the same shifted rolling columns the feature pipeline already
computes, so they're held to the identical no-lookahead rule as the model.

### Model (`fplxp/model.py`)

One `GradientBoostingRegressor` per position (GK/DEF/MID/FWD), fixed
hyperparameters (`n_estimators=200, max_depth=3, learning_rate=0.05,
subsample=0.8`), no tuning. Missing features (mostly a player's first
gameweek of a season, before any rolling window has data) are
median-imputed using train-set-only medians. A two-stage
`P(plays 60+) x points | played` model is a stretch goal, not implemented
in v0.1 — the played-60 rolling rate feature is this version's
minutes-availability signal instead.

### Evaluation (`fplxp/metrics.py`, `fplxp/backtest.py`)

Chronological split: train on 2020-21..2023-24, evaluate once on
2024-25..2025-26 (`fplxp/config.py`). No shuffled cross-validation is used
anywhere in the headline numbers. Reported per position and overall: RMSE,
MAE, and mean Spearman rank correlation computed *within each gameweek's
slate of players* and then averaged — this is the metric that actually
matters for picking a squad, since it asks "did the model rank this
gameweek's players correctly," not "were the model's numeric errors small
on average across gameweeks."

## Current result (see `reports/backtest.md` for the live numbers)

The v0.1 model beats both naive baselines on RMSE and MAE, in every
position, with no tuning. It does **not** uniformly beat the last-5-mean
baseline on Spearman rank correlation. That's reported plainly rather than
tuned away — see `QUESTIONS.md` #9 for why, and the report's Verdict
section for the full breakdown.

## Repo layout

```
fplxp/
  config.py     seasons, paths, train/test split, rolling window sizes
  data.py       download + cache season CSVs
  features.py   leakage-free feature table
  baselines.py  the two naive baselines
  model.py      per-position GBM
  metrics.py    RMSE / MAE / per-gameweek Spearman
  backtest.py   orchestrates everything, writes reports/
tests/
  test_no_leakage.py   the load-bearing guarantee -- keep this passing
  test_features.py
  test_baselines.py
  test_metrics.py
  test_model.py
data/raw/       cached CSVs (git-ignored)
reports/        backtest.md + 3 PNGs (committed)
CLAUDE.md       constraints for future agent sessions in this repo
QUESTIONS.md    judgment calls made without stopping to ask
```
