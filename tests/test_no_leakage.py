"""Leakage tests (Phase 0, docs/leakage-audit.md).

The whole point of this project is that a gameweek's features must be
computable *before* that gameweek's deadline. These tests fail loudly if
that guarantee is ever broken, either structurally (a raw same-gameweek
column sneaking into FEATURE_COLUMNS) or mechanically (a rolling/expanding
computation that isn't actually shifted before the window is applied, or a
single huge outlier gameweek leaking into its own feature row).
"""
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from fplxp.config import TRAIN_SEASONS
from fplxp.features import (
    FEATURE_COLUMNS,
    LEAK_COLUMNS,
    MARKET_FEATURE_COLUMNS,
    ROLLING_FEATURE_COLUMNS,
    SAME_ROUND_FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_feature_table,
)

# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------


def test_no_leak_columns_in_feature_list():
    """Raw same-gameweek match stats must never appear as model features."""
    overlap = set(FEATURE_COLUMNS) & LEAK_COLUMNS
    assert overlap == set(), f"leaky columns found in FEATURE_COLUMNS: {overlap}"


def test_xP_is_not_a_feature():
    """FPL's own same-gameweek xP must be excluded (dropped, not shifted)."""
    assert "xP" not in FEATURE_COLUMNS


def test_target_not_in_features():
    assert TARGET_COLUMN not in FEATURE_COLUMNS


def test_minutes_not_a_feature():
    """minutes is target-adjacent (you can't know it before kickoff) -- it's
    kept in the output table for evaluation (minutes > 0 splits) but must
    never be a model input."""
    assert "minutes" not in FEATURE_COLUMNS


def test_same_round_columns_are_the_only_legitimate_ones():
    """Every FEATURE_COLUMNS entry that ISN'T a shift(1)-then-window rolling
    feature must be on a hand-curated whitelist of information that is
    genuinely knowable before the deadline. This whitelist is written
    independently of fplxp.features's own SAME_ROUND_FEATURE_COLUMNS /
    MARKET_FEATURE_COLUMNS constants, precisely so a future same-round-ish
    column can't sneak onto the "legitimate" list just by being added to
    both places at once.
    """
    expected_whitelist = {
        # Schedule / fixture info FPL sets before kickoff, independent of outcome.
        "num_fixtures", "home_fraction",
        "fixture_difficulty_min", "fixture_difficulty_max", "fixture_difficulty_mean",
        "own_attack_strength", "own_defence_strength", "opp_attack_strength", "opp_defence_strength",
        "is_blank", "gw_number",
        # Missingness bookkeeping -- structural, not match-outcome-derived.
        "is_first_gw_of_season", "has_prior_season",
        # Market info: price/ownership are published by FPL before the deadline.
        "value", "selected_log", "value_delta_form3", "value_delta_form5",
        "selected_delta_form3", "selected_delta_form5",
    }
    same_round_in_use = set(SAME_ROUND_FEATURE_COLUMNS) | set(MARKET_FEATURE_COLUMNS)
    assert same_round_in_use == expected_whitelist, (
        f"same-round feature set changed without updating this test's independent "
        f"whitelist. In code but not whitelisted: {same_round_in_use - expected_whitelist}. "
        f"Whitelisted but not in code: {expected_whitelist - same_round_in_use}."
    )
    # And every non-rolling feature column must be in that same-round set.
    non_rolling = set(FEATURE_COLUMNS) - set(ROLLING_FEATURE_COLUMNS)
    assert non_rolling == same_round_in_use


def test_rolling_and_same_round_partition_all_features():
    assert set(FEATURE_COLUMNS) == set(ROLLING_FEATURE_COLUMNS) | set(SAME_ROUND_FEATURE_COLUMNS) | set(MARKET_FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Synthetic spike test: build a full season through the real pipeline (no
# network -- fplxp.data.load_all is monkeypatched) where one player scores
# 0 every gameweek except a single 100-point spike, and prove no feature at
# the spike gameweek reflects it.
# ---------------------------------------------------------------------------


def _synthetic_league(n_rounds=10, spike_round=6, spike_points=100):
    """Two teams, three players each, one full synthetic season."""
    season = "2099-00"
    teams_rows = [
        {"season": season, "id": 1, "name": "Test FC", "strength_attack_home": 1200,
         "strength_attack_away": 1150, "strength_defence_home": 1200, "strength_defence_away": 1150},
        {"season": season, "id": 2, "name": "Rival FC", "strength_attack_home": 1100,
         "strength_attack_away": 1050, "strength_defence_home": 1100, "strength_defence_away": 1050},
    ]
    teams = pd.DataFrame(teams_rows)

    fixture_rows = []
    for r in range(1, n_rounds + 1):
        home, away = (1, 2) if r % 2 == 1 else (2, 1)
        fixture_rows.append({
            "season": season, "event": r, "team_h": home, "team_a": away,
            "team_h_score": 1, "team_a_score": 1,
            "team_h_difficulty": 3, "team_a_difficulty": 3, "finished": True,
        })
    fixtures = pd.DataFrame(fixture_rows)

    gw_rows = []
    players = [
        (1, "Spike Player", "MID", "Test FC"),
        (2, "Steady Player", "MID", "Test FC"),
        (3, "Rival Player", "FWD", "Rival FC"),
    ]
    for element, name, position, team in players:
        for r in range(1, n_rounds + 1):
            pts = 0
            if element == 1 and r == spike_round:
                pts = spike_points
            gw_rows.append({
                "season": season, "element": element, "round": r, "name": name,
                "position": position, "team": team, "was_home": (r % 2 == 1) if team == "Test FC" else (r % 2 == 0),
                "value": 50, "selected": 1000, "minutes": 90, "total_points": pts,
                "goals_scored": 0, "assists": 0, "clean_sheets": 0, "goals_conceded": 1,
                "own_goals": 0, "penalties_missed": 0, "penalties_saved": 0, "saves": 0,
                "bonus": 0, "bps": 10, "yellow_cards": 0, "red_cards": 0,
                "ict_index": 1.0, "influence": 1.0, "creativity": 1.0, "threat": 1.0,
                "xP": 2.0,
            })
    gw = pd.DataFrame(gw_rows)
    players_raw = pd.DataFrame(
        [{"season": season, "element": element, "code": 10000 + element} for element, *_ in players]
    )
    return gw, fixtures, teams, players_raw


@pytest.fixture
def synthetic_feature_table():
    gw, fixtures, teams, players_raw = _synthetic_league()
    with mock.patch("fplxp.features.load_all", return_value=(gw, fixtures, teams, players_raw)):
        table = build_feature_table(["2099-00"])
    return table


def test_spike_gameweek_features_do_not_reflect_the_spike(synthetic_feature_table):
    table = synthetic_feature_table
    spike_row = table[(table["element"] == 1) & (table["round"] == 6)]
    assert len(spike_row) == 1
    spike_row = spike_row.iloc[0]

    control_row = table[(table["element"] == 2) & (table["round"] == 6)].iloc[0]  # never spikes

    # The rolling features at the spike gameweek must be identical in shape
    # to a player who never spiked (both are all-zero histories up to gw6),
    # since the spike itself must not have leaked backward into its own row.
    for col in ROLLING_FEATURE_COLUMNS:
        spike_val = spike_row[col]
        control_val = control_row[col]
        if pd.isna(spike_val) and pd.isna(control_val):
            continue
        assert spike_val == pytest.approx(control_val), (
            f"{col} at the spike gameweek ({spike_val}) differs from an "
            f"identically-scoring-until-now player ({control_val}) -- the "
            f"100-point spike leaked into its own gameweek's features."
        )

    # And the gameweek *after* the spike must show it (proves the pipeline
    # actually uses history at all, i.e. this isn't a vacuous pass).
    next_row = table[(table["element"] == 1) & (table["round"] == 7)].iloc[0]
    assert next_row["pts_per90_form3"] > 0


def test_spike_does_not_change_earlier_gameweeks(synthetic_feature_table):
    """A later gameweek's outcome can't retroactively change an earlier
    gameweek's features (sanity check on the shift direction itself)."""
    table = synthetic_feature_table
    for r in range(1, 6):
        row = table[(table["element"] == 1) & (table["round"] == r)].iloc[0]
        control = table[(table["element"] == 2) & (table["round"] == r)].iloc[0]
        for col in ["pts_per90_form3", "mins_form3"]:
            a, b = row[col], control[col]
            if pd.isna(a) and pd.isna(b):
                continue
            assert a == pytest.approx(b)


# ---------------------------------------------------------------------------
# Correlation smoke test on real train data: nothing should look "too good
# to be true", which is the fingerprint of leakage.
# ---------------------------------------------------------------------------


def test_no_feature_is_suspiciously_correlated_with_target():
    table = build_feature_table(TRAIN_SEASONS)
    corr = table[FEATURE_COLUMNS + [TARGET_COLUMN]].corr(method="spearman")[TARGET_COLUMN].drop(TARGET_COLUMN)
    offenders = corr[corr.abs() > 0.9]
    assert offenders.empty, f"suspiciously high |Spearman| with target (possible leakage): {offenders.to_dict()}"
