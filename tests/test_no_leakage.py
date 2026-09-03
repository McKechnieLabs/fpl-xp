"""Leakage tests.

The whole point of this project is that a gameweek's features must be
computable *before* that gameweek's deadline. These tests fail loudly if
that guarantee is ever broken, either structurally (a raw same-gameweek
column sneaking into FEATURE_COLUMNS) or mechanically (a rolling/expanding
computation that isn't actually shifted before the window is applied).

All tests here use small synthetic, in-memory DataFrames rather than the
real downloaded seasons, so they run offline and don't depend on
data/raw/ being warm.
"""
import numpy as np
import pandas as pd
import pytest

from fplxp.features import (
    FEATURE_COLUMNS,
    ID_COLUMNS,
    LEAK_COLUMNS,
    TARGET_COLUMN,
    _add_player_rolling,
    _add_team_rolling,
    _aggregate_player_gw,
    _team_gw_results,
)


def test_no_leak_columns_in_feature_list():
    """Raw same-gameweek match stats must never appear as model features."""
    overlap = set(FEATURE_COLUMNS) & LEAK_COLUMNS
    assert overlap == set(), f"leaky columns found in FEATURE_COLUMNS: {overlap}"


def test_xP_is_not_a_feature():
    """FPL's own same-gameweek xP must be excluded (dropped, not shifted)."""
    assert "xP" not in FEATURE_COLUMNS
    assert "xP" not in ID_COLUMNS


def test_target_not_in_features():
    assert TARGET_COLUMN not in FEATURE_COLUMNS


def _synthetic_player_gw(points, minutes, rounds=None):
    """One player, one season, with the given per-gameweek total_points/minutes."""
    n = len(points)
    rounds = rounds or list(range(1, n + 1))
    return pd.DataFrame(
        {
            "season": ["2099-00"] * n,
            "element": [1] * n,
            "round": rounds,
            "total_points": points,
            "minutes": minutes,
            "ict_index": [1.0] * n,
            "goals_scored": [0] * n,
            "assists": [0] * n,
            "bps": [0] * n,
            "bonus": [0] * n,
            "clean_sheets": [0] * n,
            "goals_conceded": [0] * n,
            "saves": [0] * n,
            "starts": [1] * n,
        }
    )


def test_player_rolling_form_excludes_current_gameweek():
    df = _synthetic_player_gw(points=[10, 20, 30, 40], minutes=[90, 90, 90, 90])
    out = _add_player_rolling(df).set_index("round")

    # First gameweek: no prior data at all -> NaN, not e.g. 0 or the row's own value.
    assert np.isnan(out.loc[1, "pts_form3"])
    assert np.isnan(out.loc[1, "pts_form5"])

    # Second gameweek's form is exactly gameweek 1's raw value (10), never gw2's own 20.
    assert out.loc[2, "pts_form3"] == pytest.approx(10.0)

    # Fourth gameweek's 3-window form uses gw1-3 only (10,20,30 -> 20), excluding gw4's 40.
    assert out.loc[4, "pts_form3"] == pytest.approx(20.0)
    # 5-window form with only 3 prior points available averages exactly those 3.
    assert out.loc[4, "pts_form5"] == pytest.approx(20.0)


def test_player_rolling_is_unaffected_by_current_gameweek_value():
    """Changing gw4's own total_points must not change gw4's rolling feature."""
    base = _synthetic_player_gw(points=[10, 20, 30, 40], minutes=[90, 90, 90, 90])
    mutated = _synthetic_player_gw(points=[10, 20, 30, 9999], minutes=[90, 90, 90, 90])

    out_base = _add_player_rolling(base).set_index("round")
    out_mut = _add_player_rolling(mutated).set_index("round")

    for col in ["pts_form3", "pts_form5", "mins_form3", "mins_form5"]:
        assert out_base.loc[4, col] == pytest.approx(out_mut.loc[4, col]), (
            f"{col} at gw4 changed when only gw4's own value changed -- leakage"
        )
    # Sanity: earlier gameweeks are identical too (mutation is strictly downstream in time).
    for r in [1, 2, 3]:
        for col in ["pts_form3", "pts_form5"]:
            left = out_base.loc[r, col]
            right = out_mut.loc[r, col]
            if np.isnan(left):
                assert np.isnan(right)
            else:
                assert left == pytest.approx(right)


def test_played_60_uses_only_past_minutes():
    # Player starts slow (below 60) then plays big minutes from gw3 onward.
    df = _synthetic_player_gw(points=[2, 2, 8, 8, 8], minutes=[10, 20, 90, 90, 90])
    out = _add_player_rolling(df).set_index("round")
    # At gw3, the played_60 rate should reflect gw1-2 only (both < 60 -> rate 0),
    # not gw3's own 90-minute appearance.
    assert out.loc[3, "played60_rate_form3"] == pytest.approx(0.0)
    # By gw5, the 3-window rate reflects gw2-4 (0, 1, 1 -> 2/3), excluding gw5 itself.
    assert out.loc[5, "played60_rate_form3"] == pytest.approx(2 / 3)


def _synthetic_team_gw(goals_for, goals_against, difficulty=None):
    n = len(goals_for)
    difficulty = difficulty or [3] * n
    return pd.DataFrame(
        {
            "season": ["2099-00"] * n,
            "team_id": [1] * n,
            "event": list(range(1, n + 1)),
            "goals_for": goals_for,
            "goals_against": goals_against,
            "difficulty": difficulty,
        }
    )


def test_team_rolling_excludes_current_gameweek():
    df = _synthetic_team_gw(goals_for=[1, 2, 3, 4], goals_against=[0, 1, 2, 3])
    out = _add_team_rolling(df).set_index("event")

    assert np.isnan(out.loc[1, "team_gf_form5"])
    assert out.loc[2, "team_gf_form5"] == pytest.approx(1.0)
    # gw4's rolling GF uses gw1-3 (1,2,3 -> 2.0), excluding gw4's own 4 goals.
    assert out.loc[4, "team_gf_form5"] == pytest.approx(2.0)
    assert out.loc[4, "team_ga_form5"] == pytest.approx(1.0)


def test_double_gameweek_stats_are_summed_not_leaked():
    """Two fixtures in one gameweek must be summed into one row, and the
    resulting aggregate must still only be usable as history for *later*
    gameweeks, never for itself."""
    rows = []
    for fixture_points, fixture_minutes in [(4, 90), (6, 90)]:
        rows.append(
            {
                "season": "2099-00",
                "element": 1,
                "round": 2,
                "name": "Test Player",
                "position": "MID",
                "team": "Test FC",
                "was_home": True,
                "value": 50,
                "minutes": fixture_minutes,
                "total_points": fixture_points,
                "goals_scored": 0,
                "assists": 0,
                "clean_sheets": 0,
                "goals_conceded": 0,
                "own_goals": 0,
                "penalties_missed": 0,
                "penalties_saved": 0,
                "saves": 0,
                "bonus": 0,
                "bps": 0,
                "yellow_cards": 0,
                "red_cards": 0,
                "ict_index": 1.0,
                "influence": 1.0,
                "creativity": 1.0,
                "threat": 1.0,
                "starts": 1,
            }
        )
    # Round 1: a normal single fixture, used only as history for round 2.
    rows.append(
        {
            **{k: 0 for k in rows[0] if k not in ("season", "element", "round", "name", "position", "team", "was_home", "value")},
            "season": "2099-00",
            "element": 1,
            "round": 1,
            "name": "Test Player",
            "position": "MID",
            "team": "Test FC",
            "was_home": False,
            "value": 50,
            "minutes": 90,
            "total_points": 5,
            "starts": 1,
        }
    )
    raw = pd.DataFrame(rows)
    agg = _aggregate_player_gw(raw)
    agg = agg.set_index("round")

    assert agg.loc[2, "total_points"] == 10  # 4 + 6, summed across both fixtures
    assert agg.loc[2, "minutes"] == 180
    assert agg.loc[2, "num_fixtures"] == 2

    rolled = _add_player_rolling(agg.reset_index()).set_index("round")
    # Round 2's own double-gameweek haul (10 points) must not appear in round 2's
    # own rolling feature -- only round 1's 5 points should.
    assert rolled.loc[2, "pts_form3"] == pytest.approx(5.0)
