# CLAUDE.md

Instructions for Claude Code (or any future agent) working in this repo.

## What this repo is

An expected-points (xP) model for Fantasy Premier League: predict a
player's `total_points` in a single upcoming gameweek from data available
before that gameweek's deadline. Python only.

## Data

- Source: `github.com/vaastav/Fantasy-Premier-League`, specifically
  `data/<season>/gws/merged_gw.csv`, `data/<season>/fixtures.csv`, and
  `data/<season>/teams.csv`, fetched via direct HTTPS GET
  (`raw.githubusercontent.com`) in `fplxp/data.py`.
- **Never call the live FPL API.** It is not reachable from this
  environment and will fail. All data comes from the historical CSV dumps
  above.
- Downloaded files are cached under `data/raw/<season>/` and are
  git-ignored. Re-running the pipeline re-uses the cache; delete
  `data/raw/` to force a re-download.
- Season coverage is 2020-21 through 2025-26 — see QUESTIONS.md for why
  earlier seasons are excluded.

## Leakage rules (non-negotiable)

Every feature must be computable strictly before the gameweek it predicts.

- `xP` is FPL's own expected-points figure for that same gameweek. It is
  **dropped entirely** from this project's features (not shifted) — see
  QUESTIONS.md for the reasoning. If a future version reintroduces it, it
  must be `.shift(1)`'d within each player group first, and that must be
  stated explicitly in the report.
- Any feature derived from match events (points, minutes, goals, assists,
  bps, bonus, ICT, clean sheets, saves, team goals for/against, ...) must
  be `.shift(1)`'d within its (season, player) or (season, team) group
  *before* any rolling/expanding window is computed. Fixture difficulty,
  home/away, scheduled fixture count, and price are set by FPL before
  kickoff and may be used unshifted.
- `tests/test_no_leakage.py` enforces this both structurally (leaky raw
  columns can never appear in `FEATURE_COLUMNS`) and mechanically (a
  gameweek's rolling feature must be provably unaffected by that same
  gameweek's own outcome). Keep this test passing — it is the load-bearing
  guarantee of the whole project. Extend it if you add new features.
- Train/test splits must always be chronological (train on earlier
  seasons, test on later ones). Never use shuffled/random cross-validation
  for the headline evaluation.

## Model

- v0.1 is deliberately simple: one `GradientBoostingRegressor` per
  position (GK/DEF/MID/FWD) in `fplxp/model.py`, trained on rolling-window
  form, minutes-availability, home/away, fixture difficulty, and
  team-level rolling goals for/against (`fplxp/features.py`).
- A two-stage `P(plays 60+) x points | played` structure is a stretch
  goal, not part of v0.1. The single-stage model instead includes a
  played-60-rate rolling feature as its minutes-availability signal.
- **Do not tune hyperparameters until the model beats both baselines.** If
  it doesn't, report that plainly in `reports/backtest.md` with
  diagnostics — an honest loss is the correct output. A win that turns out
  to come from a leaked feature is worthless; re-check
  `test_no_leakage.py` before trusting any improvement.

## Baselines

Every backtest run must report, alongside the model:
(a) mean of the player's last 5 gameweeks, (b) season-to-date mean.
Report RMSE, MAE, and mean per-gameweek Spearman rank correlation
(more important than RMSE for actual squad selection), broken down by
position, in `reports/backtest.md`.

## Constraints

- Python only. Standard stack: pandas, numpy, scikit-learn, matplotlib
  (plus scipy for Spearman, and stdlib `urllib` for downloads — no
  `requests`, to keep the dependency surface exactly as specified).
- One branch. Never touch `main`. Commit at each working stage.
- Don't ask the user questions mid-run. Write anything genuinely
  ambiguous to `QUESTIONS.md`, pick the most defensible option, note which
  one was picked, and keep going.

## Running things

```
pip install -r requirements.txt
python -m pytest                 # includes the leakage test
python -m fplxp.backtest         # downloads/caches data if needed, writes reports/backtest.md + 3 plots
```

See `README.md` for full reproduction details.
