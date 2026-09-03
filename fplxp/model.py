"""Per-position gradient boosting model (v0.1 — deliberately simple).

One GradientBoostingRegressor per position (GK/DEF/MID/FWD), trained on the
rolling-form / minutes-availability / fixture / team-strength features from
fplxp.features. No hyperparameter tuning: default-ish, modest-depth trees,
chosen once up front and left alone. Missing feature values (mostly a
player's first gameweek in a season, before any rolling window has data)
are median-imputed per position using train-set medians only.

A two-stage P(plays 60+) x points-if-played model is noted in the README as
a stretch goal, not implemented here.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from fplxp.config import POSITIONS, RANDOM_STATE
from fplxp.features import FEATURE_COLUMNS, TARGET_COLUMN


def _model_params() -> dict:
    return dict(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )


class PositionModels:
    """Holds one fitted GradientBoostingRegressor + imputation medians per position."""

    def __init__(self):
        self.models: dict[str, GradientBoostingRegressor] = {}
        self.medians: dict[str, pd.Series] = {}

    def fit(self, train_df: pd.DataFrame) -> "PositionModels":
        for pos in POSITIONS:
            sub = train_df[train_df["position"] == pos]
            X = sub[FEATURE_COLUMNS]
            medians = X.median()
            X = X.fillna(medians)
            y = sub[TARGET_COLUMN]

            model = GradientBoostingRegressor(**_model_params())
            model.fit(X, y)

            self.models[pos] = model
            self.medians[pos] = medians
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        preds = pd.Series(index=df.index, dtype=float)
        for pos in POSITIONS:
            mask = df["position"] == pos
            if not mask.any():
                continue
            X = df.loc[mask, FEATURE_COLUMNS].fillna(self.medians[pos])
            preds.loc[mask] = self.models[pos].predict(X)
        return preds

    def feature_importances(self, pos: str) -> pd.Series:
        model = self.models[pos]
        return pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(
            ascending=False
        )
