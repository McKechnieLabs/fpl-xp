import numpy as np
import pandas as pd

from fplxp.baselines import last5_mean_baseline, season_to_date_baseline, xp_baseline


def test_baselines_pass_through_and_fill_nan():
    df = pd.DataFrame({"pts_form5": [np.nan, 4.0, 6.0], "pts_season_avg": [np.nan, 3.0, 5.0]})
    last5 = last5_mean_baseline(df, fill_value=2.0)
    season = season_to_date_baseline(df, fill_value=2.0)
    assert last5.tolist() == [2.0, 4.0, 6.0]
    assert season.tolist() == [2.0, 3.0, 5.0]


def test_xp_baseline_passes_through_xp_column():
    df = pd.DataFrame({"xP": [1.5, 2.5, 0.0]})
    assert xp_baseline(df).tolist() == [1.5, 2.5, 0.0]
