"""Build the leakage-free player-gameweek feature table.

Pipeline:
  1. Load raw merged_gw.csv (per-fixture rows) and aggregate to
     (season, element, round) i.e. one row per player per gameweek, summing
     counting stats across double-gameweek fixtures the way FPL itself does.
  2. Join team-id and fixture-difficulty info from fixtures.csv/teams.csv.
     Difficulty ratings are pre-set by FPL before kickoff, so they can be
     used directly (no shift needed).
  3. Build team-level rolling goals-for/goals-against from the *previous*
     N gameweeks only.
  4. Build player-level rolling form features (points, minutes, ICT,
     goals, assists, clean sheets, played-60 rate, ...) from the
     *previous* N gameweeks only.

LEAKAGE RULE: any feature derived from match events (goals, minutes, points,
bps, ict, saves, clean sheets, team goals for/against, ...) is shift(1)'d
within its (season, element) or (season, team) group *before* any rolling
window is applied, so a gameweek's features only ever see strictly earlier
gameweeks. Features that are set by FPL *before* kickoff and don't depend on
match outcomes (fixture difficulty, home/away, price, scheduled fixture
count) are used unshifted, since there is nothing to leak.

xP (FPL's own pre-computed expected points for that gameweek) is DROPPED
entirely rather than shifted. It is FPL's own model output for the same
target we're predicting; even shifted by one gameweek it would encode a
competitor model's judgement about a different player-gameweek, which adds
confounding signal we don't want to lean on for a v0.1 baseline-beating
check. See QUESTIONS.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fplxp.config import PLAYED_60_THRESHOLD, ROLLING_WINDOWS, TEAM_ROLLING_WINDOW
from fplxp.data import load_all

# Columns from merged_gw.csv that are match-event derived and must be summed
# across fixtures within a gameweek (double gameweeks) before any shifting.
_SUM_COLS = [
    "minutes",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "saves",
    "bonus",
    "bps",
    "yellow_cards",
    "red_cards",
]
_SUM_COLS_FLOAT = ["ict_index", "influence", "creativity", "threat"]

# Leakage-derived columns (raw match outcomes / same-gameweek stats) that
# must never appear in the model's feature list.
LEAK_COLUMNS = set(_SUM_COLS) | set(_SUM_COLS_FLOAT) | {
    "xP",
    "team_a_score",
    "team_h_score",
    "played_60",
}


def _aggregate_player_gw(gw: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-fixture rows to one row per (season, element, round)."""
    gw = gw.copy()
    if "starts" not in gw.columns:
        gw["starts"] = (gw["minutes"] > 0).astype(int)
    sum_cols = _SUM_COLS + ["starts"]

    agg = {c: "sum" for c in sum_cols}
    agg.update({c: "sum" for c in _SUM_COLS_FLOAT})
    agg.update({
        "was_home": "mean",  # fraction of this gw's fixtures played at home
        "value": "last",
        "name": "first",
        "position": "first",
        "team": "first",
    })
    grouped = gw.groupby(["season", "element", "round"], as_index=False).agg(agg)
    grouped = grouped.rename(columns={"was_home": "home_fraction"})

    # Normalise a stray "GKP" label (one season uses it for some GK rows) and
    # drop the "AM" (fantasy Manager) position: managers are a separate FPL
    # element type introduced in later seasons, always 0 minutes, and out of
    # scope for a GK/DEF/MID/FWD points model.
    grouped["position"] = grouped["position"].replace({"GKP": "GK"})
    grouped = grouped[grouped["position"].isin(["GK", "DEF", "MID", "FWD"])]

    n_fixtures = (
        gw.groupby(["season", "element", "round"]).size().rename("num_fixtures").reset_index()
    )
    grouped = grouped.merge(n_fixtures, on=["season", "element", "round"])
    return grouped


def _team_gw_results(fixtures: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, team_id, event): goals for/against and difficulty."""
    home = fixtures.rename(
        columns={
            "team_h": "team_id",
            "team_h_score": "goals_for",
            "team_a_score": "goals_against",
            "team_h_difficulty": "difficulty",
        }
    )[["season", "event", "team_id", "goals_for", "goals_against", "difficulty"]]
    away = fixtures.rename(
        columns={
            "team_a": "team_id",
            "team_a_score": "goals_for",
            "team_h_score": "goals_against",
            "team_a_difficulty": "difficulty",
        }
    )[["season", "event", "team_id", "goals_for", "goals_against", "difficulty"]]
    both = pd.concat([home, away], ignore_index=True)
    # Sum goals across fixtures in a double gameweek; average the difficulty
    # rating FPL assigned to each of those fixtures.
    team_gw = both.groupby(["season", "team_id", "event"], as_index=False).agg(
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
        difficulty=("difficulty", "mean"),
    )
    return team_gw


def _add_team_rolling(team_gw: pd.DataFrame) -> pd.DataFrame:
    team_gw = team_gw.sort_values(["season", "team_id", "event"]).copy()
    grp = team_gw.groupby(["season", "team_id"], group_keys=False)
    for col, out in [("goals_for", "team_gf_form"), ("goals_against", "team_ga_form")]:
        shifted = grp[col].shift(1)
        team_gw[f"{out}{TEAM_ROLLING_WINDOW}"] = shifted.groupby(
            [team_gw["season"], team_gw["team_id"]]
        ).transform(lambda s: s.rolling(TEAM_ROLLING_WINDOW, min_periods=1).mean())
    return team_gw


def _add_player_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["season", "element", "round"]).copy()
    df["played_60"] = (df["minutes"] >= PLAYED_60_THRESHOLD).astype(int)

    roll_source_cols = {
        "total_points": "pts",
        "minutes": "mins",
        "played_60": "played60_rate",
        "ict_index": "ict",
        "goals_scored": "goals",
        "assists": "assists",
        "bps": "bps",
        "bonus": "bonus",
        "clean_sheets": "cs",
        "goals_conceded": "gc",
        "saves": "saves",
        "starts": "starts_rate",
    }

    key = [df["season"], df["element"]]
    for src, prefix in roll_source_cols.items():
        shifted = df.groupby(["season", "element"])[src].shift(1)
        for window in ROLLING_WINDOWS:
            colname = f"{prefix}_form{window}"
            df[colname] = shifted.groupby(key).transform(
                lambda s, w=window: s.rolling(w, min_periods=1).mean()
            )

    # Season-to-date expanding availability signal (independent of window size).
    mins_shifted = df.groupby(["season", "element"])["minutes"].shift(1)
    df["mins_season_avg"] = mins_shifted.groupby(key).transform(
        lambda s: s.expanding(min_periods=1).mean()
    )
    pts_shifted = df.groupby(["season", "element"])["total_points"].shift(1)
    df["pts_season_avg"] = pts_shifted.groupby(key).transform(
        lambda s: s.expanding(min_periods=1).mean()
    )
    games_played = df.groupby(["season", "element"]).cumcount()
    df["games_played_so_far"] = games_played
    return df


FEATURE_COLUMNS = (
    [f"pts_form{w}" for w in ROLLING_WINDOWS]
    + [f"mins_form{w}" for w in ROLLING_WINDOWS]
    + [f"played60_rate_form{w}" for w in ROLLING_WINDOWS]
    + [f"ict_form{w}" for w in ROLLING_WINDOWS]
    + [f"goals_form{w}" for w in ROLLING_WINDOWS]
    + [f"assists_form{w}" for w in ROLLING_WINDOWS]
    + [f"bps_form{w}" for w in ROLLING_WINDOWS]
    + [f"bonus_form{w}" for w in ROLLING_WINDOWS]
    + [f"cs_form{w}" for w in ROLLING_WINDOWS]
    + [f"gc_form{w}" for w in ROLLING_WINDOWS]
    + [f"saves_form{w}" for w in ROLLING_WINDOWS]
    + [f"starts_rate_form{w}" for w in ROLLING_WINDOWS]
    + [
        "mins_season_avg",
        "pts_season_avg",
        "games_played_so_far",
        "team_gf_form5",
        "team_ga_form5",
        "fixture_difficulty",
        "home_fraction",
        "num_fixtures",
        "value",
        "gw_number",
    ]
)

ID_COLUMNS = ["season", "element", "round", "name", "position", "team"]
TARGET_COLUMN = "total_points"


def build_feature_table(seasons: list[str]) -> pd.DataFrame:
    gw, fixtures, teams = load_all(seasons)
    player_gw = _aggregate_player_gw(gw)

    team_gw = _team_gw_results(fixtures)
    team_gw = _add_team_rolling(team_gw)

    # Map team name (in player_gw) -> team id (in fixtures/teams) per season.
    name_to_id = teams.rename(columns={"id": "team_id", "name": "team"})
    player_gw = player_gw.merge(name_to_id, on=["season", "team"], how="left")

    team_gw_small = team_gw[
        ["season", "team_id", "event", "team_gf_form5", "team_ga_form5", "difficulty"]
    ].rename(columns={"event": "round", "difficulty": "fixture_difficulty"})
    player_gw = player_gw.merge(
        team_gw_small, on=["season", "team_id", "round"], how="left"
    )

    player_gw = _add_player_rolling(player_gw)
    player_gw["gw_number"] = player_gw["round"]

    out = player_gw[ID_COLUMNS + [TARGET_COLUMN] + FEATURE_COLUMNS].copy()
    out = out.sort_values(["season", "round", "element"]).reset_index(drop=True)
    return out
