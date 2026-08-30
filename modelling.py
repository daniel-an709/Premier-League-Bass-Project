"""Loading, splitting and feature selection for EPL pre-match result prediction.

Reads: epl_features.csv  (leak-free pre-match features, see EPLdatacleaningscript.py)

The last three seasons are held out. Feature sets and targets are selectable so
binary and three-class models can share one pipeline.

Run: python modelling.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
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
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Metadata
FEATURES_PATH = "epl_features.csv"
RESULTS_PATH = "model_results.csv"
FIGURE_DIR = Path("figures")

# Same Material palette as EDA.py, so the figures in EDA.md and MODELLING.md
# read as one set.
BLUE, RED, GREEN, GREY = "#64B5F6", "#E57373", "#81C784", "gray"

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


# 6. ---Figures---
def plot_calibration(fitted: dict, X: pd.DataFrame, y: pd.Series) -> None:
    """Reliability diagram: predicted probability against observed frequency."""
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "--", color=GREY, linewidth=1, label="Perfect calibration")

    for name, model in fitted.items():
        prob = model.predict_proba(X)[:, 1]
        observed, predicted = calibration_curve(y, prob, n_bins=10, strategy="quantile")
        plt.plot(predicted, observed, marker="o", label=name,
                 color=dict(zip(fitted, (BLUE, RED, GREEN)))[name])

    plt.title("Calibration of Predicted Home-Win Probability (holdout)")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed home-win rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "calibration_binary.png", dpi=300)
    plt.close()


def plot_permutation_importance(model, X: pd.DataFrame, y: pd.Series) -> None:
    """Permutation importance on the holdout, not impurity importance.

    Several features correlate above r = 0.8, and impurity importance splits
    credit between correlated features arbitrarily.
    """
    result = permutation_importance(
        model, X, y, scoring="roc_auc", n_repeats=20, random_state=RANDOM_STATE
    )
    order = result.importances_mean.argsort()

    plt.figure(figsize=(9, 7))
    plt.barh(
        [X.columns[i] for i in order],
        result.importances_mean[order],
        xerr=result.importances_std[order],
        color=BLUE,
    )
    plt.axvline(0, color=GREY, linewidth=1)
    plt.title("Permutation Importance (drop in holdout AUC)")
    plt.xlabel("Mean decrease in ROC AUC")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "permutation_importance.png", dpi=300)
    plt.close()


def accuracy_by_season(df: pd.DataFrame, model, framing: str = "diff") -> pd.DataFrame:
    """Walk forward one season at a time, training only on earlier seasons."""
    cols = feature_columns(df, framing)
    seasons = sorted(df["Season"].unique())
    rows = []

    for season in seasons[5:]:
        past, current = df[df["Season"] < season], df[df["Season"] == season]
        fitted = clone(model).fit(past[cols], past["HomeWin"])
        prob = fitted.predict_proba(current[cols])[:, 1]
        rows.append({
            "Season": season,
            "AUC": roc_auc_score(current["HomeWin"], prob),
            "Accuracy": accuracy_score(current["HomeWin"], fitted.predict(current[cols])),
            "HomeWinRate": current["HomeWin"].mean(),
        })

    return pd.DataFrame(rows)


def plot_accuracy_by_season(scores: pd.DataFrame) -> None:
    """Walk-forward performance per season, against that season's home-win rate."""
    plt.figure(figsize=(11, 5))
    plt.plot(scores["Season"], scores["AUC"], marker="o", color=BLUE, label="AUC")
    plt.plot(scores["Season"], scores["Accuracy"], marker="o", color=GREEN,
             label="Accuracy")
    plt.plot(scores["Season"], scores["HomeWinRate"], marker="o", color=RED,
             linestyle="--", label="Home-win rate (base rate)")

    plt.title("Walk-Forward Performance by Season")
    plt.xlabel("Season")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "accuracy_by_season.png", dpi=300)
    plt.close()


def build_figures(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Fit the models once more on the training seasons and draw every figure."""
    FIGURE_DIR.mkdir(exist_ok=True)
    cols = feature_columns(train, "diff")
    y_train, y_test = train["HomeWin"], test["HomeWin"]

    models = build_models()
    fitted = {
        name: clone(models[name]).fit(train[cols], y_train)
        for name in ("LogisticRegression", "RandomForest", "HistGradientBoosting")
    }

    plot_calibration(fitted, test[cols], y_test)
    plot_permutation_importance(fitted["LogisticRegression"], test[cols], y_test)
    plot_accuracy_by_season(accuracy_by_season(df, models["LogisticRegression"]))


def main() -> None:
    df = load_features()
    train, test = split_by_season(df)

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
    print("Figures")
    build_figures(df, train, test)
    print(f"  wrote 3 figures to {FIGURE_DIR}/")
    print(f"Wrote {RESULTS_PATH} ({len(results)} rows)")


if __name__ == "__main__":
    main()
