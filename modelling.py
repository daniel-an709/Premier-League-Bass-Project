"""Loading, splitting and feature selection for EPL pre-match result prediction.

Reads: epl_features.csv  (leak-free pre-match features, see EPLdatacleaningscript.py)

The last three seasons are held out. Feature sets and targets are selectable so
binary and three-class models can share one pipeline.

Run: python modelling.py
"""

import numpy as np
import pandas as pd

# Metadata
FEATURES_PATH = "epl_features.csv"
RESULTS_PATH = "model_results.csv"

HOLDOUT_SEASONS = ["2022/23", "2023/24", "2024/25"]
RANDOM_STATE = 42
N_SPLITS = 5

# Identifiers and targets: everything else in epl_features.csv is a usable
# pre-match feature.
ID_COLS = [
    "MatchID", "Season", "MatchDate", "HomeTeam", "AwayTeam",
    "FullTimeResult", "Result", "HomeWin",
]

# 1. ---Load and split---
def load_features(path=FEATURES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["MatchDate"])
    df = df.sort_values(["MatchDate", "HomeTeam"]).reset_index(drop=True)
    return df

def split_by_season(df, holdout=HOLDOUT_SEASONS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole seasons, so every training match precedes every test match."""
    mask = df["Season"].isin(holdout)
    test = df[mask].reset_index(drop=True)
    train = df[~mask].reset_index(drop=True)

    print(f"Split on {', '.join(holdout)}: {len(train)} train / {len(test)} test")
    return train, test


# 2. ---Feature sets and targets---
def feature_columns(df: pd.DataFrame, framing: str = "diff") -> list[str]:
    """Select the 11 differentials, the 22 per-side levels, or all 33."""
    diffs = [c for c in df.columns if c.endswith("Diff")]
    levels = [c for c in df.columns if c not in ID_COLS and c not in diffs]
    return {"diff": diffs, "levels": levels, "both": levels + diffs}[framing]


def get_target(df: pd.DataFrame, target: str = "binary") -> pd.Series:
    """Return the binary home-win column or the three-way result."""
    return df["HomeWin"] if target == "binary" else df["Result"]

def main() -> None:
    df = load_features()
    for i, col in enumerate(df.columns):
        print(f"{i}. {col}")

if __name__ == "__main__":
    main()
