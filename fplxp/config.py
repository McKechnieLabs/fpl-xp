"""Shared constants: seasons, paths, train/val/test split, seeds.

Season coverage starts at 2020-21 because that is the first season in the
vaastav/Fantasy-Premier-League repo where merged_gw.csv carries `position`
and `team` columns directly (earlier seasons only have a numeric `element`
id and require joining separate id-list files we don't otherwise need).
See docs/judgment-calls.md.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.json"
REPORTS_DIR = REPO_ROOT / "reports"
DOCS_DIR = REPO_ROOT / "docs"
SITE_DATA_DIR = REPO_ROOT / "site" / "data"

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

ALL_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

# v0.2 chronological split (docs/judgment-calls.md #11): the test set is
# frozen and touched exactly once, at the end of Phase 4. Every iteration
# before that uses TRAIN_SEASONS / VAL_SEASONS only.
TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23"]
VAL_SEASONS = ["2023-24"]
TEST_SEASONS = ["2024-25", "2025-26"]

POSITIONS = ["GK", "DEF", "MID", "FWD"]

ROLLING_WINDOWS = [3, 5]
TEAM_ROLLING_WINDOW = 5

PLAYED_60_THRESHOLD = 60

TOP_K_VALUES = [1, 11, 15]

# Single seed used everywhere (numpy RNGs, sklearn `random_state`, any
# resampling). Nothing in this pipeline is stochastic downstream of data
# loading except model fitting itself, but every entry point still takes
# this explicitly so a cold run is bit-for-bit reproducible.
RANDOM_STATE = 42
