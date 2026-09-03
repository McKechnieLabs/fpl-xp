import numpy as np
import pandas as pd

from fplxp.features import _aggregate_player_gw, _reindex_blank_gameweeks


def _row(**overrides):
    base = {
        "season": "2099-00", "element": 1, "round": 1, "name": "Player", "position": "MID",
        "team": "A", "was_home": True, "value": 50, "selected": 100, "minutes": 90,
        "total_points": 2, "goals_scored": 0, "assists": 0, "clean_sheets": 0,
        "goals_conceded": 0, "own_goals": 0, "penalties_missed": 0, "penalties_saved": 0,
        "saves": 0, "bonus": 0, "bps": 0, "yellow_cards": 0, "red_cards": 0,
        "ict_index": 0.0, "influence": 0.0, "creativity": 0.0, "threat": 0.0, "xP": 1.0,
    }
    base.update(overrides)
    return base


def test_gkp_normalised_to_gk_and_manager_position_dropped():
    rows = [
        _row(element=1, position="GKP"),
        _row(element=2, position="AM"),
        _row(element=3, position="FWD"),
    ]
    out = _aggregate_player_gw(pd.DataFrame(rows))
    positions = set(out["position"])
    assert positions == {"GK", "FWD"}


def test_double_gameweek_sums_stats_and_played60_uses_max_not_sum():
    """Two 45-minute DGW appearances must NOT count as played_60, but one
    90-minute + one 10-minute appearance (same total) also shouldn't --
    played_60 is about the max single fixture, not the summed total."""
    rows = [
        _row(round=2, minutes=45, total_points=2),
        _row(round=2, minutes=45, total_points=3),
    ]
    out = _aggregate_player_gw(pd.DataFrame(rows)).set_index("round")
    assert out.loc[2, "minutes"] == 90        # summed total, for per-90 rates etc.
    assert out.loc[2, "total_points"] == 5    # summed, matches FPL's own DGW scoring
    assert out.loc[2, "max_fixture_minutes"] == 45
    assert out.loc[2, "played_60"] == 0       # neither fixture individually hit 60
    assert out.loc[2, "n_fixtures_60plus"] == 0
    assert out.loc[2, "num_fixtures"] == 2


def test_played_60_true_when_one_fixture_alone_hits_threshold():
    rows = [
        _row(round=2, minutes=70, total_points=6),
        _row(round=2, minutes=20, total_points=1),
    ]
    out = _aggregate_player_gw(pd.DataFrame(rows)).set_index("round")
    assert out.loc[2, "minutes"] == 90
    assert out.loc[2, "max_fixture_minutes"] == 70
    assert out.loc[2, "played_60"] == 1
    assert out.loc[2, "n_fixtures_60plus"] == 1


def test_home_fraction_averages_across_double_gameweek():
    rows = [_row(round=1, was_home=True), _row(round=1, was_home=False)]
    out = _aggregate_player_gw(pd.DataFrame(rows))
    assert out.loc[0, "home_fraction"] == 0.5
    assert out.loc[0, "num_fixtures"] == 2


def test_xp_is_summed_across_double_gameweek():
    rows = [_row(round=1, xP=2.0), _row(round=1, xP=3.5)]
    out = _aggregate_player_gw(pd.DataFrame(rows))
    assert out.loc[0, "xP"] == 5.5


# ---------------------------------------------------------------------------
# Blank gameweek reindexing
# ---------------------------------------------------------------------------


def test_blank_gameweek_reindex_fills_gap_with_zero_stats():
    agg = _aggregate_player_gw(pd.DataFrame([
        _row(round=1, total_points=5, minutes=90),
        # round 2 missing entirely -- e.g. team had no fixture (blank GW)
        _row(round=3, total_points=8, minutes=90),
    ]))
    out = _reindex_blank_gameweeks(agg).set_index("round")

    assert list(out.index) == [1, 2, 3]
    assert out.loc[2, "is_blank"] == True  # noqa: E712
    assert out.loc[2, "total_points"] == 0
    assert out.loc[2, "minutes"] == 0
    assert out.loc[2, "num_fixtures"] == 0
    assert out.loc[1, "is_blank"] == False  # noqa: E712
    assert out.loc[3, "is_blank"] == False  # noqa: E712


def test_blank_gameweek_carries_forward_identity_and_market_fields():
    agg = _aggregate_player_gw(pd.DataFrame([
        _row(round=1, value=55, selected=2000, team="Arsenal"),
        _row(round=3, value=58, selected=2500, team="Arsenal"),
    ]))
    out = _reindex_blank_gameweeks(agg).set_index("round")
    # Blank-week price/ownership carried forward from the last known value,
    # not zeroed (a player's price doesn't reset to 0 during a blank week).
    assert out.loc[2, "value"] == 55
    assert out.loc[2, "selected"] == 2000
    assert out.loc[2, "team"] == "Arsenal"
    assert out.loc[2, "name"] == "Player"


def test_reindex_does_not_extend_beyond_players_active_span():
    """A player who only appears rounds 3-5 should not get rows for 1-2 or 6+."""
    agg = _aggregate_player_gw(pd.DataFrame([
        _row(round=3), _row(round=4), _row(round=5),
    ]))
    out = _reindex_blank_gameweeks(agg)
    assert sorted(out["round"]) == [3, 4, 5]
