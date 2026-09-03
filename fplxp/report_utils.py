"""Tiny shared helpers for writing markdown reports (no extra deps: no tabulate)."""
from __future__ import annotations

import pandas as pd


def df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal DataFrame -> GitHub-flavored markdown table, standard-stack only."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |" for row in df[cols].itertuples(index=False)
    ]
    return "\n".join([header, sep] + rows)
