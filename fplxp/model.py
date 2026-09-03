"""Model variants compared per docs/evaluation-plan.md.

  - PositionModels("gbr"):          v0.1-style per-position GradientBoostingRegressor
                                     (squared error), carried forward as the
                                     model-side continuity baseline.
  - PositionModels("hgb_poisson"):  per-position HistGradientBoostingRegressor,
                                     loss="poisson" -- the textbook-correct loss
                                     for a non-negative, right-skewed count-like
                                     target. Tried first among the new variants.
  - TwoStageModel:                  P(played_60) classifier x E[points | played_60]
                                     regressor, per position, multiplied.
  - PooledModel:                    one HistGradientBoostingRegressor(poisson) over
                                     ALL positions, with position one-hot encoded as
                                     a feature -- checks whether per-position splits
                                     starve the smallest class (GK) of data.

`total_points` can be slightly negative (own goals / red cards). Poisson
deviance requires y >= 0, so poisson-loss variants fit on
`y.clip(lower=0)` -- evaluation everywhere else still uses the true,
unclipped total_points. See docs/judgment-calls.md.

Missing feature values (mostly a player's first gameweek before any prior
season or rolling window has data) are median-imputed using train-set-only
medians (per position, where the model is per-position; over the whole
train set for the pooled model).
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from fplxp.config import PLAYED_60_THRESHOLD, POSITIONS, RANDOM_STATE
from fplxp.features import FEATURE_COLUMNS, TARGET_COLUMN

ESTIMATORS = {
    "gbr": lambda: GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE
    ),
    "hgb_poisson": lambda: HistGradientBoostingRegressor(
        loss="poisson", max_depth=6, learning_rate=0.05, max_iter=300, random_state=RANDOM_STATE
    ),
    "hgb_squared": lambda: HistGradientBoostingRegressor(
        loss="squared_error", max_depth=6, learning_rate=0.05, max_iter=300, random_state=RANDOM_STATE
    ),
}


class PositionModels:
    """One fitted regressor + imputation medians per position."""

    def __init__(self, estimator_kind: str = "gbr"):
        self.estimator_kind = estimator_kind
        self.models: dict[str, object] = {}
        self.medians: dict[str, pd.Series] = {}

    def fit(self, train_df: pd.DataFrame) -> "PositionModels":
        for pos in POSITIONS:
            sub = train_df[train_df["position"] == pos]
            X = sub[FEATURE_COLUMNS]
            medians = X.median()
            X = X.fillna(medians)
            y = sub[TARGET_COLUMN]
            if "poisson" in self.estimator_kind:
                y = y.clip(lower=0)

            model = ESTIMATORS[self.estimator_kind]()
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
        if hasattr(model, "feature_importances_"):
            return pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
        return pd.Series(dtype=float)  # HistGradientBoosting* has no native importances


class TwoStageModel:
    """P(played_60) classifier x E[points | played_60] regressor, per position."""

    def __init__(self):
        self.classifiers: dict[str, HistGradientBoostingClassifier] = {}
        self.regressors: dict[str, HistGradientBoostingRegressor] = {}
        self.medians: dict[str, pd.Series] = {}

    def fit(self, train_df: pd.DataFrame) -> "TwoStageModel":
        for pos in POSITIONS:
            sub = train_df[train_df["position"] == pos]
            X = sub[FEATURE_COLUMNS]
            medians = X.median()
            X = X.fillna(medians)
            played_60 = (sub["minutes"] >= PLAYED_60_THRESHOLD).astype(int)

            clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=300, random_state=RANDOM_STATE)
            clf.fit(X, played_60)

            played_mask = played_60 == 1
            reg = HistGradientBoostingRegressor(
                loss="poisson", max_depth=6, learning_rate=0.05, max_iter=300, random_state=RANDOM_STATE
            )
            reg.fit(X[played_mask], sub.loc[played_mask, TARGET_COLUMN].clip(lower=0))

            self.classifiers[pos] = clf
            self.regressors[pos] = reg
            self.medians[pos] = medians
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        preds = pd.Series(index=df.index, dtype=float)
        for pos in POSITIONS:
            mask = df["position"] == pos
            if not mask.any():
                continue
            X = df.loc[mask, FEATURE_COLUMNS].fillna(self.medians[pos])
            p_played = self.classifiers[pos].predict_proba(X)[:, 1]
            e_points = self.regressors[pos].predict(X)
            preds.loc[mask] = p_played * e_points
        return preds


class PooledModel:
    """One HistGradientBoostingRegressor(poisson) over all positions, position one-hot."""

    def __init__(self):
        self.model: HistGradientBoostingRegressor | None = None
        self.median: pd.Series | None = None
        self.columns: list[str] | None = None

    def _design_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURE_COLUMNS].copy()
        dummies = pd.get_dummies(df["position"], prefix="pos")
        for pos in POSITIONS:
            X[f"pos_{pos}"] = dummies.get(f"pos_{pos}", 0)
        return X

    def fit(self, train_df: pd.DataFrame) -> "PooledModel":
        X = self._design_matrix(train_df)
        self.columns = X.columns.tolist()
        self.median = X.median()
        X = X.fillna(self.median)
        y = train_df[TARGET_COLUMN].clip(lower=0)

        self.model = HistGradientBoostingRegressor(
            loss="poisson", max_depth=6, learning_rate=0.05, max_iter=300, random_state=RANDOM_STATE
        )
        self.model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X = self._design_matrix(df)[self.columns].fillna(self.median)
        return pd.Series(self.model.predict(X), index=df.index)
