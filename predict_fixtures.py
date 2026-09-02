"""Predict upcoming EPL fixtures with the model selected in MODELLING.md.

Reads : epl_clean.csv     completed matches, used to compute each team's form
        epl_features.csv  the training table for the model

The model is refitted on every completed match each time this runs, which takes
under a second, so nothing is serialised to disk.

IMPORTANT: form is only as current as epl_clean.csv. Predictions for fixtures
long after the last completed match in that file are extrapolations from stale
form, and the script says so.

Run: python predict_fixtures.py
"""

import pandas as pd
from sklearn.base import clone

from EPLdatacleaningscript import FORM_STATS, FORM_WINDOW, to_team_match_log
from modelling import build_models, feature_columns, load_features

CLEAN_PATH = "epl_clean.csv"

# Fixtures to predict: (home team, away team, kick-off date).
FIXTURES = [
    ("Man United", "Ipswich", "2026-08-31"),
    ("Aston Villa", "Arsenal", "2026-09-01")
]


def team_form(log: pd.DataFrame, team: str, as_of: pd.Timestamp) -> dict:
    """Form going into a fixture: the team's last completed matches, unshifted.

    The Form* columns in epl_features.csv are shifted back one match so that a
    played match never sees its own result. An upcoming fixture has no result to
    hide, so its form must include the most recent completed match. Reading the
    stored columns instead would leave every prediction one match out of date.
    """
    history = log[(log["Team"] == team) & (log["MatchDate"] < as_of)]
    history = history.sort_values("MatchDate")
    if history.empty:
        raise ValueError(f"No completed matches on record for {team}.")

    recent = history.tail(FORM_WINDOW)
    form = {f"Form{stat}": recent[stat].mean() for stat in FORM_STATS}

    latest_season = history["Season"].iloc[-1]
    form["SeasonPPG"] = history.loc[history["Season"] == latest_season, "Points"].mean()
    form["DaysSinceLastMatch"] = (as_of - history["MatchDate"].iloc[-1]).days
    form["MatchesPlayed"] = len(history)
    return form


def fixture_features(log: pd.DataFrame, home: str, away: str, date: str) -> pd.Series:
    """Build the home-minus-away differentials for one unplayed fixture."""
    as_of = pd.Timestamp(date)
    home_form = team_form(log, home, as_of)
    away_form = team_form(log, away, as_of)
    return pd.Series({f"{k}Diff": home_form[k] - away_form[k] for k in home_form})


def predict(fixtures: list[tuple[str, str, str]] = FIXTURES) -> pd.DataFrame:
    """Refit the chosen model on all completed matches and score each fixture."""
    clean = pd.read_csv(CLEAN_PATH, parse_dates=["MatchDate"])
    log = to_team_match_log(clean)
    last_played = clean["MatchDate"].max()

    train = load_features()
    cols = feature_columns(train, "diff")
    model = clone(build_models()["LogisticRegression"])
    model.fit(train[cols], train["HomeWin"])

    rows = []
    for home, away, date in fixtures:
        features = fixture_features(log, home, away, date)
        probability = model.predict_proba(features[cols].to_frame().T)[0, 1]
        rows.append({
            "Fixture": f"{home} vs {away}",
            "Date": date,
            "P(home win)": probability,
            "FormAgeDays": (pd.Timestamp(date) - last_played).days,
        })

    return pd.DataFrame(rows)


def main() -> None:
    clean = pd.read_csv(CLEAN_PATH, parse_dates=["MatchDate"])
    print(f"Form is current to {clean['MatchDate'].max().date()}")

    predictions = predict()
    print()
    print(predictions.round(3).to_string(index=False))

    stale = predictions[predictions["FormAgeDays"] > 100]
    if not stale.empty:
        print()
        print("WARNING: form for these fixtures predates the match by over 100 days.")
        print("Refresh epl_clean.csv before treating these as real predictions.")


if __name__ == "__main__":
    main()
