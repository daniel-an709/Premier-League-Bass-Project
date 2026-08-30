"""Loading, splitting and feature selection for EPL pre-match result prediction.

Reads: epl_features.csv  (leak-free pre-match features, see EPLdatacleaningscript.py)

The last three seasons are held out. Feature sets and targets are selectable so
binary and three-class models can share one pipeline.

Run: python modelling.py
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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


# 3. ---Models---
def build_models(random_state: int = RANDOM_STATE) -> dict:
    """Five models, in increasing order of what they are allowed to learn."""
    return {
        # Predicts the class base rates. The floor every other model must clear.
        "Baseline": DummyClassifier(strategy="prior", random_state=random_state),

        # Same pipeline as EDA.py, so we can compare results to EDA.md.
        # The only model here that needs imputing and scaling.
        "LogisticRegression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]),

        # Left unconstrained on purpose: the expected gap between its train and holdout
        # scores supports the following two ensembles
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),

        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=random_state,
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=random_state,
        ),
    }


def main() -> None:
    df = load_features()
    train, test = split_by_season(df)
    print(f"Models: {', '.join(build_models())}")


if __name__ == "__main__":
    main()
