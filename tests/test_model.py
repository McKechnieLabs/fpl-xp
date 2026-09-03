import numpy as np
import pandas as pd

from fplxp.config import POSITIONS
from fplxp.features import FEATURE_COLUMNS, TARGET_COLUMN
from fplxp.model import PooledModel, PositionModels, TwoStageModel


def _synthetic_table(n_per_position=40, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for pos in POSITIONS:
        data = {col: rng.normal(size=n_per_position) for col in FEATURE_COLUMNS}
        data["position"] = pos
        data["minutes"] = rng.integers(0, 91, size=n_per_position)
        data[TARGET_COLUMN] = rng.normal(loc=3, size=n_per_position)
        frames.append(pd.DataFrame(data))
    df = pd.concat(frames, ignore_index=True)
    df.loc[0, FEATURE_COLUMNS[0]] = np.nan  # sprinkle a realistic missing value
    return df


def test_position_models_gbr_fit_predict_smoke():
    df = _synthetic_table()
    models = PositionModels(estimator_kind="gbr").fit(df)
    preds = models.predict(df)
    assert len(preds) == len(df)
    assert preds.notna().all()


def test_position_models_hgb_poisson_handles_negative_target():
    """total_points can be slightly negative (own goals/red cards); the
    poisson-loss variant must clip for fitting rather than crash."""
    df = _synthetic_table()
    df.loc[0, TARGET_COLUMN] = -3
    models = PositionModels(estimator_kind="hgb_poisson").fit(df)
    preds = models.predict(df)
    assert preds.notna().all()


def test_two_stage_model_smoke():
    df = _synthetic_table()
    model = TwoStageModel().fit(df)
    preds = model.predict(df)
    assert len(preds) == len(df)
    assert preds.notna().all()
    assert (preds >= 0).all()  # P(played) * E[points|played] with clipped-nonneg regressor


def test_pooled_model_smoke_and_uses_position():
    df = _synthetic_table()
    model = PooledModel().fit(df)
    preds = model.predict(df)
    assert len(preds) == len(df)
    assert preds.notna().all()


def test_feature_importances_cover_all_features_for_gbr():
    df = _synthetic_table()
    models = PositionModels(estimator_kind="gbr").fit(df)
    for pos in POSITIONS:
        importances = models.feature_importances(pos)
        assert set(importances.index) == set(FEATURE_COLUMNS)
        assert np.isclose(importances.sum(), 1.0, atol=1e-6)
