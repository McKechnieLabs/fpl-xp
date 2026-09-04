"""Download and cache raw season CSVs, behind a swappable data-source interface.

Four files per season are needed:
  - gws/merged_gw.csv  (per-fixture player rows: minutes, points, xP, ...)
  - fixtures.csv       (per-fixture team difficulty ratings, scores)
  - teams.csv          (team id -> name and strength ratings, for the season)
  - players_raw.csv    (that season's element id -> `code`, FPL's own
                         persistent player identifier -- unlike `element`,
                         `code` is stable across seasons; e.g. Harry Kane is
                         element 357/427/500 in three different seasons but
                         code 78830 in all of them. Used for cross-season
                         GW1 carryover matching -- see docs/judgment-calls.md.)

`HistoricalCSVDataSource` (the only implementation used by this project)
fetches them from the vaastav/Fantasy-Premier-League GitHub repo via plain
HTTPS GET (raw.githubusercontent.com) and caches them under
data/raw/<season>/. Every downloaded file is recorded in data/manifest.json
(source URL, first-download date, SHA256). On a warm cache nothing is
re-downloaded; if a file *is* re-downloaded and its hash no longer matches
the manifest, loading fails loudly instead of silently training on changed
upstream data.

This module never calls the live FPL API. `DataSource` is defined as an
interface specifically so that a future live source (FPL's
`bootstrap-static` and `fixtures` endpoints) can be swapped in without
touching fplxp/features.py -- see LiveFPLDataSource below and
docs/judgment-calls.md for why that's out of scope for v0.2.
"""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fplxp.config import GITHUB_RAW_BASE, MANIFEST_PATH, RAW_DIR

_FILES = ["gws/merged_gw.csv", "fixtures.csv", "teams.csv", "players_raw.csv"]

_FIXTURES_COLS = [
    "event",
    "team_h",
    "team_a",
    "team_h_score",
    "team_a_score",
    "team_h_difficulty",
    "team_a_difficulty",
    "finished",
]
_TEAMS_COLS = [
    "id",
    "name",
    "strength_attack_home",
    "strength_attack_away",
    "strength_defence_home",
    "strength_defence_away",
]


class DataSource(ABC):
    """Interface a season data provider must satisfy.

    Anything upstream of this (features.py, model.py, ...) only ever talks
    to a DataSource, never to a URL or a file path -- swapping the
    historical CSV dump for a live API means writing one new class here.
    """

    @abstractmethod
    def merged_gw(self, season: str) -> pd.DataFrame: ...

    @abstractmethod
    def fixtures(self, season: str) -> pd.DataFrame: ...

    @abstractmethod
    def teams(self, season: str) -> pd.DataFrame: ...

    @abstractmethod
    def players_raw(self, season: str) -> pd.DataFrame: ...


def _load_manifest(path: Path = MANIFEST_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_manifest(manifest: dict, path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _record_or_verify(manifest: dict, key: str, url: str, content: bytes) -> dict:
    digest = hashlib.sha256(content).hexdigest()
    prior = manifest.get(key)
    if prior is not None and prior["sha256"] != digest:
        raise RuntimeError(
            f"upstream content changed for {key}: manifest recorded sha256 "
            f"{prior['sha256']} on {prior['downloaded_at']}, but a fresh "
            f"download now hashes to {digest}. Refusing to silently train "
            f"on changed historical data -- delete the stale manifest entry "
            f"deliberately if this change is expected."
        )
    now = datetime.now(timezone.utc).isoformat()
    manifest[key] = {
        "url": url,
        "downloaded_at": prior["downloaded_at"] if prior else now,
        "last_verified_at": now,
        "sha256": digest,
    }
    return manifest


class HistoricalCSVDataSource(DataSource):
    """The vaastav/Fantasy-Premier-League historical CSV dumps (default source)."""

    def __init__(self, raw_dir: Path = RAW_DIR, manifest_path: Path = MANIFEST_PATH):
        self.raw_dir = raw_dir
        self.manifest_path = manifest_path

    def _ensure_season_cached(self, season: str) -> None:
        manifest = _load_manifest(self.manifest_path)
        changed = False
        for rel in _FILES:
            dest = self.raw_dir / season / rel
            key = f"{season}/{rel}"
            if dest.exists():
                continue
            url = f"{GITHUB_RAW_BASE}/{season}/{rel}"
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    content = resp.read()
            except urllib.error.URLError as exc:
                raise RuntimeError(f"failed to download {url}: {exc}") from exc
            manifest = _record_or_verify(manifest, key, url, content)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            changed = True
        if changed:
            _save_manifest(manifest, self.manifest_path)

    def merged_gw(self, season: str) -> pd.DataFrame:
        self._ensure_season_cached(season)
        df = pd.read_csv(self.raw_dir / season / "gws" / "merged_gw.csv")
        df["season"] = season
        return df

    def fixtures(self, season: str) -> pd.DataFrame:
        self._ensure_season_cached(season)
        df = pd.read_csv(self.raw_dir / season / "fixtures.csv", usecols=_FIXTURES_COLS)
        df["season"] = season
        return df

    def teams(self, season: str) -> pd.DataFrame:
        self._ensure_season_cached(season)
        # Older seasons predate some strength columns; fill with NaN rather
        # than fail so the schema stays uniform across seasons.
        available = pd.read_csv(self.raw_dir / season / "teams.csv", nrows=0).columns
        cols = [c for c in _TEAMS_COLS if c in available]
        df = pd.read_csv(self.raw_dir / season / "teams.csv", usecols=cols)
        for c in _TEAMS_COLS:
            if c not in df.columns:
                df[c] = pd.NA
        df["season"] = season
        return df

    def players_raw(self, season: str) -> pd.DataFrame:
        self._ensure_season_cached(season)
        df = pd.read_csv(self.raw_dir / season / "players_raw.csv", usecols=["id", "code"])
        df = df.rename(columns={"id": "element"})
        df["season"] = season
        return df


class LiveFPLDataSource(DataSource):
    """Placeholder for a live-season source using FPL's own endpoints.

    Out of scope for v0.2 (docs/judgment-calls.md): live predictions would
    need `bootstrap-static` (current prices/ownership/teams) and `fixtures`
    from FPL's own API, which is not reachable from this sandboxed
    environment. The class exists so the *interface* is proven swappable;
    it is not implemented.
    """

    def merged_gw(self, season: str) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError("live FPL source is out of scope for v0.2")

    def fixtures(self, season: str) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError("live FPL source is out of scope for v0.2")

    def teams(self, season: str) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError("live FPL source is out of scope for v0.2")

    def players_raw(self, season: str) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError("live FPL source is out of scope for v0.2")


def load_all(seasons: list[str], source: DataSource | None = None):
    source = source or HistoricalCSVDataSource()
    gw = pd.concat([source.merged_gw(s) for s in seasons], ignore_index=True)
    fixtures = pd.concat([source.fixtures(s) for s in seasons], ignore_index=True)
    teams = pd.concat([source.teams(s) for s in seasons], ignore_index=True)
    players = pd.concat([source.players_raw(s) for s in seasons], ignore_index=True)
    return gw, fixtures, teams, players
