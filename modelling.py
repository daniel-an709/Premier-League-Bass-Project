"""Loading, splitting and feature selection for EPL pre-match result prediction.

Reads: epl_features.csv  (leak-free pre-match features, see EPLdatacleaningscript.py)

The last three seasons are held out. Feature sets and targets are selectable so
binary and three-class models can share one pipeline.

Run: python modelling.py
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
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


# 4. ---Evaluation---
def score(model, X: pd.DataFrame, y: pd.Series, target: str) -> dict[str, float]:
    """Score one fitted model. Probability metrics first, accuracy last.

    Match outcomes are genuinely uncertain, so how well the predicted
    probabilities are calibrated matters more than how often the argmax is right.
    """
    proba = model.predict_proba(X)
    labels = list(model.classes_)

    if target == "binary":
        p = proba[:, 1]
        log, brier = log_loss(y, p), brier_score_loss(y, p)
        auc = roc_auc_score(y, p)
    else:
        log = log_loss(y, proba, labels=labels)
        brier = brier_score_loss(y, proba, labels=labels)
        auc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=labels)

    return {
        "LogLoss": log,
        "Brier": brier,
        "AUC": auc,
        "Accuracy": accuracy_score(y, model.predict(X)),
    }


def cross_validate_models(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    target: str,
    n_splits: int = N_SPLITS,
) -> pd.DataFrame:
    """Mean scores over expanding time-ordered folds of the training seasons."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for name, model in models.items():
        folds = [
            score(clone(model).fit(X.iloc[tr], y.iloc[tr]), X.iloc[va], y.iloc[va], target)
            for tr, va in tscv.split(X)
        ]
        rows.append({"Model": name, **pd.DataFrame(folds).mean().to_dict()})
    return pd.DataFrame(rows)


def evaluate_holdout(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target: str,
) -> pd.DataFrame:
    """Fit on all training seasons, then score on train and holdout side by side.

    Both are reported so that overfitting can be seen in the table.
    """
    rows = []
    for name, model in models.items():
        fitted = clone(model).fit(X_train, y_train)
        train = score(fitted, X_train, y_train, target)
        holdout = score(fitted, X_test, y_test, target)
        rows.append({
            "Model": name,
            **{f"Train{k}": v for k, v in train.items()},
            **{f"Holdout{k}": v for k, v in holdout.items()},
        })
    return pd.DataFrame(rows)


# 5. ---Run everything---
def run_all(
    train: pd.DataFrame,
    test: pd.DataFrame,
    framings: tuple[str, ...] = ("diff", "levels"),
    targets: tuple[str, ...] = ("binary", "multiclass"),
) -> pd.DataFrame:
    """Score every model on each feature framing and each target, in one table."""
    models = build_models()
    frames = []

    for target in targets:
        y_train, y_test = get_target(train, target), get_target(test, target)

        for framing in framings:
            cols = feature_columns(train, framing)
            print(f"  {target} / {framing} ({len(cols)} features)")

            cv = cross_validate_models(models, train[cols], y_train, target)
            cv = cv.rename(columns=lambda c: c if c == "Model" else f"CV{c}")
            holdout = evaluate_holdout(
                models, train[cols], y_train, test[cols], y_test, target
            )

            merged = cv.merge(holdout, on="Model")
            merged.insert(0, "Target", target)
            merged.insert(1, "Framing", framing)
            frames.append(merged)

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    train, test = split_by_season(load_features())

    print()
    print("Fitting")
    results = run_all(train, test)
    results.to_csv(RESULTS_PATH, index=False)

    summary = ["Target", "Framing", "Model", "CVLogLoss", "CVAUC",
               "HoldoutLogLoss", "HoldoutAUC", "HoldoutAccuracy"]
    print()
    print("Results (sorted by holdout log loss)")
    print(
        results[summary]
        .sort_values(["Target", "HoldoutLogLoss"])
        .round(3)
        .to_string(index=False)
    )
    print()
    print(f"Wrote {RESULTS_PATH} ({len(results)} rows)")


if __name__ == "__main__":
    main()
