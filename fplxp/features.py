"""Build the leakage-free player-gameweek feature table (v0.2).

Pipeline:
  1. Aggregate raw merged_gw.csv (per-fixture rows) to one row per
     (season, element, round), summing counting stats across
     double-gameweek fixtures the way FPL itself totals a DGW score.
     played_60 is derived from the *max* per-fixture minutes, not summed
     minutes, so a DGW of two 45-minute appearances doesn't falsely count
     as "played 60+".
  2. Reindex each player's rows to a complete (season, element, round)
     grid over their active span (first..last round they have any row in
     that season). A gap is one of two genuinely different things, and
     they're now told apart using the team's real fixture list
     (fixtures.csv): if the player's team had NO fixture that round, it's
     a true blank (is_blank=True, num_fixtures=0, all stats 0). If the
     team DID play and the row is simply missing from the source dump
     (an unregistered/omitted player), num_fixtures is set to the team's
     actual fixture count that round and is_blank stays False -- treated
     like any other unused-squad-member row, since 0 is still the least
     wrong guess for what an omitted player scored, but num_fixtures=0
     would have been factually wrong. Either way this makes rolling
     windows span *calendar* gameweeks rather than silently skipping
     straight past a gap as if it never happened.
  3. Join per-team fixture info from fixtures.csv/teams.csv: goals
     for/against (for a team-level rolling GF/GA feature), fixture
     difficulty (min/max/mean across the gameweek's fixtures), and
     opponent-strength ratings side-matched to home/away. All of this is
     set by FPL before kickoff and doesn't depend on the match outcome,
     so it's used unshifted -- there's nothing to leak.
  4. Build player-level rolling features over the previous N calendar
     gameweeks: minutes/availability ("volume") kept separate from
     per-90 output rates ("quality"), so the model isn't forced to infer
     "how much did they play" and "how good were they when they played"
     from one conflated number.
  5. Market features (price, ownership, and their rolling deltas) are
     added and labelled distinctly in the report -- they're crowd/FPL
     pricing-algorithm wisdom, not football signal we computed.
  6. For a player's gameweek-1 row (no in-season history yet), missing
     rolling features are backfilled from their *prior season's* per-90
     rates and average minutes (matched by `code`, FPL's own persistent
     player identifier -- `element` is NOT stable across seasons, e.g.
     Harry Kane is element 357/427/500 across three different seasons)
     rather than left for positional-median imputation, when a prior
     season exists for them.

LEAKAGE RULE: any feature derived from match events (goals, minutes,
points, bps, ict, saves, clean sheets, team goals for/against, whether
they played 60+ minutes, ...) is `.shift(1)`'d within its (season,
element) or (season, team) group *before* any rolling window or expanding
mean is applied, so a gameweek's features only ever see strictly earlier
gameweeks. Features set by FPL *before* kickoff and independent of the
match outcome (fixture difficulty, opponent strength, home/away, price,
ownership, scheduled fixture count) are used unshifted. See
docs/leakage-audit.md and tests/test_no_leakage.py.

xP (FPL's own pre-computed expected points for that gameweek) is DROPPED
entirely rather than shifted -- see docs/judgment-calls.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fplxp.config import ALL_SEASONS, PLAYED_60_THRESHOLD, ROLLING_WINDOWS, TEAM_ROLLING_WINDOW
from fplxp.data import load_all

# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

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

# Stats that get a rolling *per-90* rate feature (quality, decoupled from
# how much the player was rotated/rested -- see PER_90_STATS below).
_PER90_STAT_MAP = {
    "goals_scored": "goals",
    "assists": "assists",
    "bps": "bps",
    "bonus": "bonus",
    "ict_index": "ict",
    "saves": "saves",
    "goals_conceded": "gc",
    "total_points": "pts",
}

# Raw same-gameweek columns that must never appear in FEATURE_COLUMNS.
LEAK_COLUMNS = set(_SUM_COLS) | set(_SUM_COLS_FLOAT) | {
    "xP",
    "team_a_score",
    "team_h_score",
    "played_60",
    "max_fixture_minutes",
    "n_fixtures_60plus",
}


def _aggregate_player_gw(gw: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-fixture rows to one row per (season, element, round)."""
    gw = gw.copy()
    gw["_fixture_played60"] = (gw["minutes"] >= PLAYED_60_THRESHOLD).astype(int)

    agg = {c: "sum" for c in _SUM_COLS}
    agg.update({c: "sum" for c in _SUM_COLS_FLOAT})
    agg.update({
        "was_home": "mean",  # fraction of this gw's fixtures played at home
        "value": "last",
        "selected": "last",
        "name": "first",
        "position": "first",
        "team": "first",
        # xP is carried through ONLY as the baseline_xp benchmark (fplxp.baselines) --
        # never as a model feature. See docs/judgment-calls.md v0.1 #3.
        "xP": "sum",
    })
    grouped = gw.groupby(["season", "element", "round"], as_index=False).agg(agg)
    grouped = grouped.rename(columns={"was_home": "home_fraction"})

    # Normalise a stray "GKP" label (one season uses it for some GK rows) and
    # drop the "AM" (fantasy Manager) position: managers are a separate FPL
    # element type introduced in later seasons, always 0 minutes, and out of
    # scope for a GK/DEF/MID/FWD points model.
    grouped["position"] = grouped["position"].replace({"GKP": "GK"})
    grouped = grouped[grouped["position"].isin(["GK", "DEF", "MID", "FWD"])]

    extra = gw.groupby(["season", "element", "round"]).agg(
        num_fixtures=("minutes", "size"),
        max_fixture_minutes=("minutes", "max"),
        n_fixtures_60plus=("_fixture_played60", "sum"),
    ).reset_index()
    grouped = grouped.merge(extra, on=["season", "element", "round"])
    grouped["played_60"] = (grouped["max_fixture_minutes"] >= PLAYED_60_THRESHOLD).astype(int)
    grouped["is_blank"] = False
    return grouped


def _reindex_blank_gameweeks(df: pd.DataFrame, team_schedule: dict) -> pd.DataFrame:
    """Fill gaps in each player's (season, round) span with explicit rows.

    merged_gw.csv has no row at all for a gameweek where either (a) the
    player's team had no fixture (a true blank), or (b) the team DID play
    but the source data simply has no row for this player (an
    unregistered/omitted player -- e.g. sold, injured and dropped from the
    squad list entirely). `team_schedule` -- {(season, team_id, round):
    team's actual fixture count that round}, built from fixtures.csv by
    the caller -- tells the two apart: case (a) gets is_blank=True,
    num_fixtures=0; case (b) gets is_blank=False and the team's real
    num_fixtures, since claiming zero scheduled fixtures for a team that
    played would be factually wrong. Both cases still get zeroed counting
    stats (0 is the least-wrong guess for what an omitted player scored),
    so rolling windows span *calendar* gameweeks either way rather than
    silently skipping straight past a gap as if it never happened.

    Requires `team_id` and `code` already merged onto `df` (so they can be
    forward-filled across a gap along with the other identity columns).
    """
    fill_zero = [c for c in _SUM_COLS + _SUM_COLS_FLOAT] + [
        "num_fixtures", "max_fixture_minutes", "n_fixtures_60plus", "played_60", "home_fraction", "xP",
    ]
    carry_forward = ["name", "position", "team", "team_id", "code", "value", "selected"]

    out_parts = []
    for (season, element), sub in df.groupby(["season", "element"], sort=False):
        sub = sub.sort_values("round")
        full_rounds = pd.RangeIndex(sub["round"].min(), sub["round"].max() + 1)
        sub = sub.set_index("round").reindex(full_rounds)
        sub.index.name = "round"
        sub["season"] = season
        sub["element"] = element
        was_missing = sub["is_blank"].isna()
        sub[carry_forward] = sub[carry_forward].ffill()

        if was_missing.any():
            team_played = pd.Series(
                [team_schedule.get((season, tid, rnd), 0) for tid, rnd in zip(sub["team_id"], sub.index)],
                index=sub.index,
            )
            is_data_gap = was_missing & (team_played > 0)
            sub.loc[is_data_gap, "num_fixtures"] = team_played[is_data_gap]

        sub[fill_zero] = sub[fill_zero].fillna(0)
        sub["is_blank"] = was_missing & (sub["num_fixtures"] == 0)
        out_parts.append(sub.reset_index())

    out = pd.concat(out_parts, ignore_index=True)
    int_cols = ["num_fixtures", "max_fixture_minutes", "n_fixtures_60plus", "played_60"] + _SUM_COLS
    out[int_cols] = out[int_cols].astype(int)
    return out


# --------------------------------------------------------------------------
# Team-level, fixture-level features (all same-round-legitimate: FPL sets
# these before kickoff and they don't depend on the match outcome)
# --------------------------------------------------------------------------

def _team_fixture_features(fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, team_id, event): goals for/against, difficulty
    min/max/mean, and opponent-strength ratings side-matched to home/away."""
    strength = teams.set_index(["season", "id"])[
        ["strength_attack_home", "strength_attack_away", "strength_defence_home", "strength_defence_away"]
    ]

    def _side(df, own_col, opp_col, is_home):
        side = df.rename(columns={
            own_col: "team_id",
            opp_col: "opponent_id",
        })[["season", "event", "team_id", "opponent_id", "team_h_score", "team_a_score",
            "team_h_difficulty", "team_a_difficulty"]].copy()
        if is_home:
            side["goals_for"] = side["team_h_score"]
            side["goals_against"] = side["team_a_score"]
            side["difficulty"] = side["team_h_difficulty"]
            own_attack, own_defence = "strength_attack_home", "strength_defence_home"
            opp_attack, opp_defence = "strength_attack_away", "strength_defence_away"
        else:
            side["goals_for"] = side["team_a_score"]
            side["goals_against"] = side["team_h_score"]
            side["difficulty"] = side["team_a_difficulty"]
            # Team is playing away, so its own ratings are the *_away pair;
            # the opponent is at home, so the opponent's ratings are *_home.
            own_attack, own_defence = "strength_attack_away", "strength_defence_away"
            opp_attack, opp_defence = "strength_attack_home", "strength_defence_home"

        own_strength = strength.reindex(
            pd.MultiIndex.from_arrays([side["season"], side["team_id"]])
        ).reset_index(drop=True)
        opp_strength = strength.reindex(
            pd.MultiIndex.from_arrays([side["season"], side["opponent_id"]])
        ).reset_index(drop=True)
        side["own_attack_strength"] = own_strength[own_attack].values
        side["own_defence_strength"] = own_strength[own_defence].values
        side["opp_attack_strength"] = opp_strength[opp_attack].values
        side["opp_defence_strength"] = opp_strength[opp_defence].values
        return side[["season", "event", "team_id", "goals_for", "goals_against", "difficulty",
                      "own_attack_strength", "own_defence_strength", "opp_attack_strength", "opp_defence_strength"]]

    home = _side(fixtures, "team_h", "team_a", is_home=True)
    away = _side(fixtures, "team_a", "team_h", is_home=False)
    both = pd.concat([home, away], ignore_index=True)

    team_gw = both.groupby(["season", "team_id", "event"], as_index=False).agg(
        num_fixtures=("goals_for", "size"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
        fixture_difficulty_min=("difficulty", "min"),
        fixture_difficulty_max=("difficulty", "max"),
        fixture_difficulty_mean=("difficulty", "mean"),
        own_attack_strength=("own_attack_strength", "mean"),
        own_defence_strength=("own_defence_strength", "mean"),
        opp_attack_strength=("opp_attack_strength", "mean"),
        opp_defence_strength=("opp_defence_strength", "mean"),
    )
    return team_gw


def _add_team_rolling(team_gw: pd.DataFrame) -> pd.DataFrame:
    team_gw = team_gw.sort_values(["season", "team_id", "event"]).copy()
    key = [team_gw["season"], team_gw["team_id"]]
    for col, out in [("goals_for", "team_gf_form"), ("goals_against", "team_ga_form")]:
        shifted = team_gw.groupby(["season", "team_id"])[col].shift(1)
        team_gw[f"{out}{TEAM_ROLLING_WINDOW}"] = shifted.groupby(key).transform(
            lambda s: s.rolling(TEAM_ROLLING_WINDOW, min_periods=1).mean()
        )
    return team_gw


# --------------------------------------------------------------------------
# Player-level rolling features
# --------------------------------------------------------------------------

def _add_player_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume (minutes/availability) and per-90 rate (quality) rolling
    features, all shift(1)'d within (season, element) before any window."""
    df = df.sort_values(["season", "element", "round"]).copy()
    key = [df["season"], df["element"]]
    grp = df.groupby(["season", "element"])

    def shifted(col):
        return grp[col].shift(1)

    # --- volume / availability (kept separate from quality) ---
    for src, prefix in [
        ("minutes", "mins"),
        ("played_60", "played60_rate"),
        ("clean_sheets", "cs_rate"),
    ]:
        s = shifted(src)
        for w in ROLLING_WINDOWS:
            df[f"{prefix}_form{w}"] = s.groupby(key).transform(
                lambda x, w=w: x.rolling(w, min_periods=1).mean()
            )

    # starts_rate: uniformly minutes > 0 across every season (docs/judgment-calls.md).
    started = (df["minutes"] > 0).astype(int)
    started_shifted = started.groupby(key).shift(1)
    for w in ROLLING_WINDOWS:
        df[f"starts_rate_form{w}"] = started_shifted.groupby(key).transform(
            lambda x, w=w: x.rolling(w, min_periods=1).mean()
        )

    # --- quality: rolling per-90 rate = rolling_sum(stat) / rolling_sum(minutes) * 90 ---
    mins_shifted = shifted("minutes")
    for w in ROLLING_WINDOWS:
        sum_mins_w = mins_shifted.groupby(key).transform(
            lambda x, w=w: x.rolling(w, min_periods=1).sum()
        )
        for src, prefix in _PER90_STAT_MAP.items():
            sum_stat_w = shifted(src).groupby(key).transform(
                lambda x, w=w: x.rolling(w, min_periods=1).sum()
            )
            # Three cases: (1) sum_mins_w > 0 -> a real per-90 rate. (2)
            # sum_mins_w == 0 exactly (the window has real history, all of
            # it 0 minutes -- e.g. an unused sub or a long-term injury) ->
            # 0.0, not NaN: there is no evidence of output, which is a
            # different, stronger claim than "we don't know" and must not
            # be washed out to the position median at fit time (that would
            # describe an injured player as league-average quality). (3)
            # sum_mins_w is NaN (no history at all yet this season) -> stays
            # NaN, genuinely unknown, left for prior-season carryover /
            # median imputation.
            per90 = pd.Series(np.nan, index=df.index)
            has_mins = sum_mins_w > 0
            per90[has_mins] = sum_stat_w[has_mins] / sum_mins_w[has_mins] * 90
            zero_mins = sum_mins_w == 0
            per90[zero_mins] = 0.0
            df[f"{prefix}_per90_form{w}"] = per90
        # pts_form{w}: plain rolling mean of raw points, kept ONLY to drive
        # the two naive baselines (fplxp.baselines) -- not a model feature,
        # since per-90 quality + volume are the decomposed versions of it.
        df[f"pts_form{w}"] = shifted("total_points").groupby(key).transform(
            lambda x, w=w: x.rolling(w, min_periods=1).mean()
        )

    # --- season-to-date summary ---
    df["mins_season_avg"] = mins_shifted.groupby(key).transform(
        lambda s: s.expanding(min_periods=1).mean()
    )
    df["pts_season_avg"] = shifted("total_points").groupby(key).transform(
        lambda s: s.expanding(min_periods=1).mean()
    )
    df["gws_of_history"] = df.groupby(["season", "element"]).cumcount()
    df["is_first_gw_of_season"] = (df["round"] == df.groupby(["season", "element"])["round"].transform("min")).astype(int)
    return df


# --------------------------------------------------------------------------
# Market features (price / ownership -- crowd-wisdom, not football signal)
# --------------------------------------------------------------------------

MARKET_FEATURE_COLUMNS = (
    ["value", "selected_log"]
    + [f"value_delta_form{w}" for w in ROLLING_WINDOWS]
    + [f"selected_delta_form{w}" for w in ROLLING_WINDOWS]
)


def _add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["selected_log"] = np.log1p(df["selected"].astype(float))
    key = [df["season"], df["element"]]
    grp = df.groupby(["season", "element"])
    for w in ROLLING_WINDOWS:
        df[f"value_delta_form{w}"] = df["value"] - grp["value"].shift(w)
        df[f"selected_delta_form{w}"] = df["selected_log"] - grp["selected_log"].shift(w)
    return df


# --------------------------------------------------------------------------
# Prior-season GW1 carryover
# --------------------------------------------------------------------------

# The rolling-feature columns that GW1 carryover backfills, and the
# prior-season summary statistic used to fill each.
_CARRYOVER_MAP = {
    "mins_form3": "prior_mins_per_gw", "mins_form5": "prior_mins_per_gw",
    "played60_rate_form3": "prior_played60_rate", "played60_rate_form5": "prior_played60_rate",
    "starts_rate_form3": "prior_starts_rate", "starts_rate_form5": "prior_starts_rate",
    "cs_rate_form3": "prior_cs_rate", "cs_rate_form5": "prior_cs_rate",
    "mins_season_avg": "prior_mins_per_gw",
}
for _w in ROLLING_WINDOWS:
    for _stat, _prefix in _PER90_STAT_MAP.items():
        _CARRYOVER_MAP[f"{_prefix}_per90_form{_w}"] = f"prior_{_prefix}_per90"


def _prior_season_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, code): that whole season's per-90 rates / rates,
    to be looked up as *next* season's GW1 carryover. Keyed by `code`
    (FPL's persistent player id), not `name` or `element` -- `element`
    resets every season, and player name strings aren't even consistent
    across seasons in this data source (e.g. "Ben White" one year,
    "Benjamin White" the next), so a name join under- and mis-matches in
    a way `code` doesn't. See docs/judgment-calls.md."""
    g = df.groupby(["season", "code"])
    total_mins = g["minutes"].sum()
    n_gw = g["round"].count()
    summary = pd.DataFrame({
        "prior_mins_per_gw": total_mins / n_gw,
        "prior_played60_rate": g["played_60"].mean(),
        "prior_starts_rate": (df["minutes"] > 0).astype(int).groupby([df["season"], df["code"]]).mean(),
        "prior_cs_rate": g["clean_sheets"].mean(),
    })
    for stat, prefix in _PER90_STAT_MAP.items():
        summary[f"prior_{prefix}_per90"] = np.where(
            total_mins > 0, g[stat].sum() / total_mins * 90, np.nan
        )
    return summary.reset_index()


def _add_prior_season_carryover(df: pd.DataFrame) -> pd.DataFrame:
    ordered = sorted(set(ALL_SEASONS) | set(df["season"].unique()))
    prior_season_of = {ordered[i]: ordered[i - 1] for i in range(1, len(ordered))}

    summary = _prior_season_summary(df)
    summary = summary.rename(columns={"season": "prior_season"})

    df = df.copy()
    df["prior_season"] = df["season"].map(prior_season_of)
    df = df.merge(summary, on=["prior_season", "code"], how="left", suffixes=("", "_prior"))
    df["has_prior_season"] = df[list(_CARRYOVER_MAP.values())[0]].notna().astype(int)

    is_gw1 = (df["round"] == 1)
    for feat_col, prior_col in _CARRYOVER_MAP.items():
        mask = is_gw1 & df[feat_col].isna() & df[prior_col].notna()
        df.loc[mask, feat_col] = df.loc[mask, prior_col]

    df = df.drop(columns=["prior_season"] + list(set(_CARRYOVER_MAP.values())))
    return df


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

ROLLING_FEATURE_COLUMNS = (
    [f"mins_form{w}" for w in ROLLING_WINDOWS]
    + [f"played60_rate_form{w}" for w in ROLLING_WINDOWS]
    + [f"starts_rate_form{w}" for w in ROLLING_WINDOWS]
    + [f"cs_rate_form{w}" for w in ROLLING_WINDOWS]
    + [f"{p}_per90_form{w}" for w in ROLLING_WINDOWS for p in _PER90_STAT_MAP.values()]
    + ["mins_season_avg", "gws_of_history", "team_gf_form5", "team_ga_form5"]
)

SAME_ROUND_FEATURE_COLUMNS = [
    "num_fixtures",
    "home_fraction",
    "fixture_difficulty_min",
    "fixture_difficulty_max",
    "fixture_difficulty_mean",
    "own_attack_strength",
    "own_defence_strength",
    "opp_attack_strength",
    "opp_defence_strength",
    "is_blank",
    "gw_number",
    "is_first_gw_of_season",
    "has_prior_season",
]

FEATURE_COLUMNS = ROLLING_FEATURE_COLUMNS + SAME_ROUND_FEATURE_COLUMNS + MARKET_FEATURE_COLUMNS

ID_COLUMNS = ["season", "element", "code", "round", "name", "position", "team", "value", "selected", "is_blank"]
TARGET_COLUMN = "total_points"


def build_feature_table(seasons: list[str]) -> pd.DataFrame:
    gw, fixtures, teams, players = load_all(seasons)
    player_gw = _aggregate_player_gw(gw)

    # team_id and code must be known BEFORE reindexing: team_id so a filled
    # gap can be looked up against the team's real fixture schedule
    # (true blank vs. data gap -- see _reindex_blank_gameweeks), code so it
    # forward-fills across a gap like any other identity column.
    name_to_id = teams[["season", "id", "name"]].rename(columns={"id": "team_id", "name": "team"})
    player_gw = player_gw.merge(name_to_id, on=["season", "team"], how="left")
    player_gw = player_gw.merge(players[["season", "element", "code"]], on=["season", "element"], how="left")

    team_gw = _team_fixture_features(fixtures, teams)
    team_schedule = {
        (row.season, row.team_id, row.event): row.num_fixtures for row in team_gw.itertuples()
    }
    team_gw = _add_team_rolling(team_gw)

    player_gw = _reindex_blank_gameweeks(player_gw, team_schedule)

    team_gw_cols = [
        "season", "team_id", "event", "team_gf_form5", "team_ga_form5",
        "fixture_difficulty_min", "fixture_difficulty_max", "fixture_difficulty_mean",
        "own_attack_strength", "own_defence_strength", "opp_attack_strength", "opp_defence_strength",
    ]
    player_gw = player_gw.merge(
        team_gw[team_gw_cols].rename(columns={"event": "round"}),
        on=["season", "team_id", "round"], how="left",
    )

    player_gw = _add_player_rolling(player_gw)
    player_gw = _add_market_features(player_gw)
    player_gw = _add_prior_season_carryover(player_gw)
    player_gw["gw_number"] = player_gw["round"]
    player_gw["is_blank"] = player_gw["is_blank"].astype(int)

    keep = list(dict.fromkeys(
        ID_COLUMNS + [TARGET_COLUMN] + FEATURE_COLUMNS
        + ["pts_form3", "pts_form5", "pts_season_avg", "xP", "minutes"]
    ))
    out = player_gw[keep].copy()
    out = out.sort_values(["season", "round", "element"]).reset_index(drop=True)
    return out
