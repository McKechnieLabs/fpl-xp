"""Two naive baselines the model must beat before any tuning happens.

Both reuse columns already computed by fplxp.features.build_feature_table:
  - pts_form5      = mean total_points over the player's previous 5 GWs
                      (shift(1) then rolling(5), so GW g only sees g-5..g-1)
  - pts_season_avg  = expanding mean of total_points over the player's
                      previous GWs *this season* (shift(1) then expanding)

Reusing them (rather than recomputing) guarantees the baselines are held to
exactly the same no-lookahead rule as the model's own features.
"""
from __future__ import annotations

import pandas as pd


def last5_mean_baseline(df: pd.DataFrame, fill_value: float) -> pd.Series:
    return df["pts_form5"].fillna(fill_value)


def season_to_date_baseline(df: pd.DataFrame, fill_value: float) -> pd.Series:
    return df["pts_season_avg"].fillna(fill_value)
