"""Download and cache raw season CSVs from the vaastav/Fantasy-Premier-League repo.

Only three files per season are needed:
  - gws/merged_gw.csv  (per-fixture player rows: minutes, points, xP, ...)
  - fixtures.csv       (per-fixture team difficulty ratings, scores)
  - teams.csv          (team id -> name, for the given season)

Files are cached under data/raw/<season>/ and never re-downloaded once
present, so re-running the pipeline offline works as long as the cache is
warm. This module never calls the live FPL API.
"""
from __future__ import annotations

import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd

from fplxp.config import GITHUB_RAW_BASE, RAW_DIR

_FILES = ["gws/merged_gw.csv", "fixtures.csv", "teams.csv"]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to download {url}: {exc}") from exc
    dest.write_bytes(data)


def ensure_season_cached(season: str, raw_dir: Path = RAW_DIR) -> None:
    """Download the season's CSVs into the cache if not already present."""
    for rel in _FILES:
        dest = raw_dir / season / rel
        if dest.exists():
            continue
        url = f"{GITHUB_RAW_BASE}/{season}/{rel}"
        _download(url, dest)


def load_merged_gw(season: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    ensure_season_cached(season, raw_dir)
    df = pd.read_csv(raw_dir / season / "gws" / "merged_gw.csv")
    df["season"] = season
    return df


def load_fixtures(season: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    ensure_season_cached(season, raw_dir)
    cols = [
        "event",
        "team_h",
        "team_a",
        "team_h_score",
        "team_a_score",
        "team_h_difficulty",
        "team_a_difficulty",
        "finished",
    ]
    df = pd.read_csv(raw_dir / season / "fixtures.csv", usecols=cols)
    df["season"] = season
    return df


def load_teams(season: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    ensure_season_cached(season, raw_dir)
    df = pd.read_csv(raw_dir / season / "teams.csv", usecols=["id", "name"])
    df["season"] = season
    return df


def load_all(seasons: list[str], raw_dir: Path = RAW_DIR):
    gw = pd.concat([load_merged_gw(s, raw_dir) for s in seasons], ignore_index=True)
    fixtures = pd.concat([load_fixtures(s, raw_dir) for s in seasons], ignore_index=True)
    teams = pd.concat([load_teams(s, raw_dir) for s in seasons], ignore_index=True)
    return gw, fixtures, teams
