import pandas as pd

from fplxp.features import _aggregate_player_gw


def test_gkp_normalised_to_gk_and_manager_position_dropped():
    rows = []
    common = {
        "season": "2099-00",
        "was_home": True,
        "value": 45,
        "minutes": 0,
        "total_points": 0,
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
        "ict_index": 0.0,
        "influence": 0.0,
        "creativity": 0.0,
        "threat": 0.0,
        "starts": 0,
    }
    rows.append({**common, "element": 1, "round": 1, "name": "Keeper", "position": "GKP", "team": "A"})
    rows.append({**common, "element": 2, "round": 1, "name": "Manager", "position": "AM", "team": "A"})
    rows.append({**common, "element": 3, "round": 1, "name": "Striker", "position": "FWD", "team": "A"})

    out = _aggregate_player_gw(pd.DataFrame(rows))
    positions = set(out["position"])
    assert positions == {"GK", "FWD"}
    assert "AM" not in positions
    assert "GKP" not in positions


def test_home_fraction_averages_across_double_gameweek():
    common = {
        "season": "2099-00",
        "element": 1,
        "round": 1,
        "name": "Player",
        "position": "MID",
        "team": "A",
        "value": 50,
        "minutes": 90,
        "total_points": 2,
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
        "ict_index": 0.0,
        "influence": 0.0,
        "creativity": 0.0,
        "threat": 0.0,
        "starts": 1,
    }
    rows = [{**common, "was_home": True}, {**common, "was_home": False}]
    out = _aggregate_player_gw(pd.DataFrame(rows))
    assert out.loc[0, "home_fraction"] == 0.5
    assert out.loc[0, "num_fixtures"] == 2
    assert out.loc[0, "total_points"] == 4
    assert out.loc[0, "minutes"] == 180
