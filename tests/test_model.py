import numpy as np
import pandas as pd

from fplxp.config import POSITIONS
from fplxp.features import FEATURE_COLUMNS, TARGET_COLUMN
from fplxp.model import PositionModels


def _synthetic_table(n_per_position=30, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for pos in POSITIONS:
        data = {col: rng.normal(size=n_per_position) for col in FEATURE_COLUMNS}
        data["position"] = pos
        data[TARGET_COLUMN] = rng.normal(loc=3, size=n_per_position)
        frames.append(pd.DataFrame(data))
    df = pd.concat(frames, ignore_index=True)
    # Sprinkle some NaNs like a real first-gameweek-of-season row would have.
    df.loc[0, FEATURE_COLUMNS[0]] = np.nan
    return df


def test_fit_predict_smoke():
    df = _synthetic_table()
    models = PositionModels().fit(df)
    preds = models.predict(df)
    assert len(preds) == len(df)
    assert preds.notna().all()


def test_feature_importances_cover_all_features():
    df = _synthetic_table()
    models = PositionModels().fit(df)
    for pos in POSITIONS:
        importances = models.feature_importances(pos)
        assert set(importances.index) == set(FEATURE_COLUMNS)
        assert np.isclose(importances.sum(), 1.0, atol=1e-6)
