# CLAUDE.md

Instructions for Claude Code (or any future agent) working in this repo.

## What this repo is

An expected-points (xP) model for Fantasy Premier League: predict a
player's `total_points` in a single upcoming gameweek from data available
before that gameweek's deadline (a nowcast for that one gameweek, not a
season projection). Python only, plus a static site published from
`site/`.

## Data

- Source: `github.com/vaastav/Fantasy-Premier-League`, specifically
  `data/<season>/gws/merged_gw.csv`, `data/<season>/fixtures.csv`, and
  `data/<season>/teams.csv`, fetched via direct HTTPS GET
  (`raw.githubusercontent.com`) through `fplxp/data.py`'s `DataSource`
  interface (`HistoricalCSVDataSource` is the only implementation used;
  `LiveFPLDataSource` is a documented no-op stub -- see
  `docs/judgment-calls.md` #17).
- **Never call the live FPL API from the sandboxed dev environment.** It
  is not reachable there. All data comes from the historical CSV dumps
  above. (The weekly GitHub Actions runner has unrestricted egress and is
  the intended place for any future live-data work.)
- Downloaded files are cached under `data/raw/<season>/` and are
  git-ignored. Every download is recorded in `data/manifest.json`
  (URL, first-download date, SHA256); a cold re-download that doesn't
  match a prior manifest entry raises rather than silently training on
  changed upstream data. Re-running the pipeline re-uses the cache;
  delete `data/raw/` to force a re-download (the manifest lets it verify
  or fail instead of trusting a fresh pull blindly).
- Season coverage is 2020-21 through 2025-26 -- see
  `docs/judgment-calls.md` for why earlier seasons are excluded.

## Split (frozen test set -- do not touch casually)

`fplxp/config.py`: train = 2020-21..2022-23, validation = 2023-24,
test = 2024-25 + 2025-26. **The test set is touched exactly once**, in
`fplxp/backtest.py` (Phase 4 in `docs/evaluation-plan.md`'s terms). Any
new experiment, comparison, or sanity check uses train/validation only --
add a validation metric instead of peeking at test. If tuning anything,
use forward-chaining CV by season or `GroupKFold` grouped by player,
never random `KFold` (adjacent gameweeks of the same player leak across
folds).

## Leakage rules (non-negotiable)

Every feature must be computable strictly before the gameweek it predicts.

- `xP` is FPL's own expected-points figure for that same gameweek. It is
  **dropped entirely** from this project's features (not shifted), but
  kept as `baseline_xp` -- the natural benchmark, reported prominently.
  See `docs/judgment-calls.md`.
- Any feature derived from match events (points, minutes, goals, assists,
  bps, bonus, ICT, clean sheets, saves, team goals for/against, played-60,
  ...) must be `.shift(1)`'d within its (season, player) or (season, team)
  group *before* any rolling/expanding window is computed. Fixture
  difficulty, opponent strength, home/away, scheduled fixture count,
  price, and ownership are set by FPL before kickoff and may be used
  unshifted.
- Rolling windows span **calendar gameweeks**, including blanks: each
  player is reindexed to a complete (season, round) grid over their
  active span, and a gap becomes an explicit `is_blank=True` row with
  zero stats, not a silently-skipped round.
- `tests/test_no_leakage.py` enforces all of this: structurally (leaky
  raw columns can never appear in `FEATURE_COLUMNS`; every non-rolling
  feature must be on a hand-curated, independently-written whitelist of
  legitimately pre-deadline information), and mechanically (a synthetic
  100-point spike gameweek must not appear in that gameweek's own
  features; no feature may have `|Spearman| > 0.9` with the target on
  train). Keep this test passing -- it is the load-bearing guarantee of
  the whole project, re-run it after any feature change. See
  `docs/leakage-audit.md` for the full audit writeup.

## Model

- `fplxp/model.py` has four variants: `PositionModels("gbr")` (v0.1-style
  per-position `GradientBoostingRegressor`), `PositionModels("hgb_poisson")`
  (`HistGradientBoostingRegressor(loss="poisson")`), `TwoStageModel`
  (P(played_60) x E[points|played]), and `PooledModel` (one model across
  positions, position one-hot). Poisson-loss variants fit on
  `y.clip(lower=0)` (points can be slightly negative) but are evaluated
  against the true target everywhere.
- Which variant is used is decided **once**, by `fplxp/model_selection.py`,
  on validation only, via the pre-committed rule in
  `docs/evaluation-plan.md` -- never hand-picked after seeing test
  numbers. `python -m fplxp.model_selection` writes
  `reports/chosen_model.json`; `fplxp.backtest` reads it rather than
  re-deciding.
- **Do not tune hyperparameters, and do not add a new model variant,
  without updating `docs/evaluation-plan.md` first.** The whole point of
  freezing the metric set and selection rule before running anything is
  that it can't be gamed after the fact. An honest loss (the v0.2 model
  does not beat FPL's own `xP` on rank correlation -- see
  `reports/backtest.md`) is the correct output to report, not a signal to
  keep tweaking until it goes away.

## Metrics

The frozen set, defined in `docs/evaluation-plan.md` and implemented in
`fplxp/metrics.py`: RMSE/MAE/Spearman, each computed twice (all rows, and
restricted to `minutes > 0`), precision@k and realised-points-of-top-k for
k in {1, 11, 15} against baselines and an oracle, and a calibration table
(actual vs. predicted by predicted decile, per position). Compared against
three baselines: last-5-gameweek mean, season-to-date mean, and FPL's own
`xP`. Spearman uses `DataFrame.corr(method="spearman")` -- no scipy.

## Constraints

- Python only. Standard stack: pandas, numpy, scikit-learn, matplotlib,
  stdlib (`urllib` for downloads, `hashlib`/`json` for the manifest -- no
  `requests`, no `scipy`, to keep the dependency surface exactly as
  specified). `requirements.txt` pins exact versions.
- One branch. Never touch `main`. Commit at each working stage.
- Don't ask the user questions mid-run. Write anything genuinely
  ambiguous to `docs/judgment-calls.md`, pick the most defensible option,
  note which one was picked, and keep going.
- `gh-pages` branch is fully generated output (published `site/` content,
  force-pushed wholesale by `.github/workflows/weekly-refresh.yml`) --
  never hand-edit it, and force-pushing *that specific branch* is
  expected, unlike everywhere else in this project.

## Running things

```
pip install -r requirements.txt
make all          # test -> model-selection (validation only) -> backtest (frozen test, once)
```

or step by step:

```
python -m pytest                    # includes the leakage tests
python -m fplxp.model_selection     # Phase 3: picks a model on validation, writes reports/chosen_model.json
python -m fplxp.backtest            # Phase 4: the one frozen test run, writes reports/backtest.md + site/data
```

See `README.md` for full reproduction details and the published site URL.
