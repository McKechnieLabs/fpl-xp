"""Naive baselines the model must beat before any tuning happens.

last5_mean_baseline and season_to_date_baseline reuse columns already
computed by fplxp.features.build_feature_table:
  - pts_form5      = mean total_points over the player's previous 5 GWs
                      (shift(1) then rolling(5), so GW g only sees g-5..g-1)
  - pts_season_avg  = expanding mean of total_points over the player's
                      previous GWs *this season* (shift(1) then expanding)

Reusing them (rather than recomputing) guarantees the baselines are held to
exactly the same no-lookahead rule as the model's own features.

xp_baseline is FPL's own pre-match xP for that same gameweek -- the
natural benchmark, not a naive one, included per docs/evaluation-plan.md.
It is never used as a model feature (docs/judgment-calls.md v0.1 #3).
"""
from __future__ import annotations

import pandas as pd


def last5_mean_baseline(df: pd.DataFrame, fill_value: float) -> pd.Series:
    return df["pts_form5"].fillna(fill_value)


def season_to_date_baseline(df: pd.DataFrame, fill_value: float) -> pd.Series:
    return df["pts_season_avg"].fillna(fill_value)


def xp_baseline(df: pd.DataFrame) -> pd.Series:
    return df["xP"]
