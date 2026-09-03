"""Shared constants: seasons, paths, train/test split.

Season coverage starts at 2020-21 because that is the first season in the
vaastav/Fantasy-Premier-League repo where merged_gw.csv carries `position`
and `team` columns directly (earlier seasons only have a numeric `element`
id and require joining separate id-list files we don't otherwise need).
See QUESTIONS.md.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
REPORTS_DIR = REPO_ROOT / "reports"

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

ALL_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

# Chronological split: train on strictly earlier seasons, test on later ones.
TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24"]
TEST_SEASONS = ["2024-25", "2025-26"]

POSITIONS = ["GK", "DEF", "MID", "FWD"]

ROLLING_WINDOWS = [3, 5]
TEAM_ROLLING_WINDOW = 5

PLAYED_60_THRESHOLD = 60

RANDOM_STATE = 42
