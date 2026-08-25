"""Clean the raw EPL match data and build feature sets for predicting match results.

Reads  : epl_final.csv      (raw, 2000/01 - 2024/25 match-level data)
Writes : epl_clean.csv      cleaned match data + targets + in-match differentials
         epl_features.csv   leak-free pre-match features (rolling form), model-ready

Why two files? Almost every statistic in the raw data (shots, corners, cards,
half-time goals) is recorded DURING the match, so it is not available before
kick-off. Using those columns to "predict" the result is target leakage - the
model looks accurate but could never be used on an upcoming fixture.

    epl_clean.csv    -> use to ASK "which match statistics separate winners
                        from losers?" (explanatory / descriptive)
    epl_features.csv -> use to ASK "can I forecast the result of a fixture
                        that has not been played yet?" (predictive)

Run: python EPLdatacleaningscript.py
"""

import numpy as np
import pandas as pd

RAW_PATH = "epl_final.csv"
CLEAN_PATH = "epl_clean.csv"
FEATURES_PATH = "epl_features.csv"

# Number of previous matches used to summarise a team's form.
FORM_WINDOW = 5

# Per-team statistics carried into the rolling form features.
FORM_STATS = [
    "GoalsFor",
    "GoalsAgainst",
    "Shots",
    "ShotsOnTarget",
    "Corners",
    "Fouls",
    "YellowCards",
    "Points",
]


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

def load_raw(path=RAW_PATH):
    """Load the raw file, parse dates and put matches in chronological order."""
    df = pd.read_csv(path, parse_dates=["MatchDate"])
    df = df.sort_values(["MatchDate", "HomeTeam"]).reset_index(drop=True)
    df.insert(0, "MatchID", np.arange(1, len(df) + 1))
    return df


# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------

def flag_impossible_stats(df):
    """Replace physically impossible shot counts with NaN instead of trusting them.

    Two problems exist in the source data (football-data.co.uk style feeds):

    1. Shots on target greater than total shots - a recording error, so both
       numbers for that team are untrustworthy.
    2. A team credited with 0 shots AND 0 shots on target AND 0 corners. A
       Premier League side has never actually failed to register a single shot;
       these are missing values that were stored as 0, and left as 0 they would
       drag down every average they touch.

    Rather than dropping the whole match (the goals and the result are still
    good), only the affected shot columns are set to NaN and the row is flagged.
    """
    df = df.copy()
    df["HadStatError"] = False

    for side in ("Home", "Away"):
        shots = f"{side}Shots"
        on_target = f"{side}ShotsOnTarget"
        corners = f"{side}Corners"

        inconsistent = df[on_target] > df[shots]
        blank = (df[shots] == 0) & (df[on_target] == 0) & (df[corners] == 0)
        bad = inconsistent | blank

        df.loc[bad, [shots, on_target]] = np.nan
        df.loc[bad, "HadStatError"] = True

    return df


def verify_result_labels(df):
    """Confirm FullTimeResult / HalfTimeResult agree with the goals scored.

    Raises if not, because a mislabelled target would silently corrupt every
    model trained downstream.
    """
    def label(home_goals, away_goals):
        return np.select(
            [home_goals > away_goals, home_goals < away_goals],
            ["H", "A"],
            default="D",
        )

    full = label(df["FullTimeHomeGoals"], df["FullTimeAwayGoals"])
    half = label(df["HalfTimeHomeGoals"], df["HalfTimeAwayGoals"])

    bad_full = int((full != df["FullTimeResult"]).sum())
    bad_half = int((half != df["HalfTimeResult"]).sum())
    if bad_full or bad_half:
        raise ValueError(
            f"Result labels disagree with goals: {bad_full} full-time, "
            f"{bad_half} half-time rows"
        )
    return df


def check_duplicates(df):
    """Drop exact duplicate fixtures (same season, date, and both teams)."""
    keys = ["Season", "MatchDate", "HomeTeam", "AwayTeam"]
    duplicated = int(df.duplicated(keys).sum())
    if duplicated:
        df = df.drop_duplicates(keys).reset_index(drop=True)
    return df, duplicated


def add_targets(df):
    """Add the prediction targets.

    HomeWin is the binary target (1 = home win, 0 = draw or away win).
    Result is the three-class target, and points are the football-standard
    3/1/0 reward used later as a form measure.
    """
    df = df.copy()
    df["HomeWin"] = (df["FullTimeResult"] == "H").astype(int)
    df["AwayWin"] = (df["FullTimeResult"] == "A").astype(int)
    df["Draw"] = (df["FullTimeResult"] == "D").astype(int)
    df["Result"] = df["FullTimeResult"].map(
        {"H": "HomeWin", "D": "Draw", "A": "AwayWin"}
    )
    df["HomePoints"] = df["FullTimeResult"].map({"H": 3, "D": 1, "A": 0})
    df["AwayPoints"] = df["FullTimeResult"].map({"H": 0, "D": 1, "A": 3})
    return df


def add_match_differentials(df):
    """Add home-minus-away differentials and shooting efficiency ratios.

    A differential answers "who dominated this match?" in one number, which is
    far more informative than the two raw counts on their own. NOTE: these are
    in-match statistics - explanatory only, never inputs to a real forecast.
    """
    df = df.copy()

    for stat in ("Shots", "ShotsOnTarget", "Corners", "Fouls",
                 "YellowCards", "RedCards"):
        df[f"{stat}Diff"] = df[f"Home{stat}"] - df[f"Away{stat}"]

    df["GoalDiff"] = df["FullTimeHomeGoals"] - df["FullTimeAwayGoals"]
    df["HalfTimeGoalDiff"] = df["HalfTimeHomeGoals"] - df["HalfTimeAwayGoals"]
    df["TotalGoals"] = df["FullTimeHomeGoals"] + df["FullTimeAwayGoals"]

    # Shot accuracy = share of shots on target; conversion = goals per shot on
    # target. Guarded against divide-by-zero, which becomes NaN not infinity.
    for side, goals in (("Home", "FullTimeHomeGoals"), ("Away", "FullTimeAwayGoals")):
        df[f"{side}ShotAccuracy"] = (
            df[f"{side}ShotsOnTarget"] / df[f"{side}Shots"].replace(0, np.nan)
        )
        df[f"{side}Conversion"] = (
            df[goals] / df[f"{side}ShotsOnTarget"].replace(0, np.nan)
        )

    df["ShotAccuracyDiff"] = df["HomeShotAccuracy"] - df["AwayShotAccuracy"]
    return df


def clean(df):
    """Run the full cleaning pipeline and report what changed."""
    n_start = len(df)
    df = verify_result_labels(df)
    df, n_dupes = check_duplicates(df)
    df = flag_impossible_stats(df)
    df = add_targets(df)
    df = add_match_differentials(df)

    print("Cleaning")
    print(f"  rows in / out            : {n_start} / {len(df)}")
    print(f"  duplicate fixtures dropped: {n_dupes}")
    print(f"  rows with repaired stats  : {int(df['HadStatError'].sum())}")
    print("  result labels             : verified against goals")
    return df


# ---------------------------------------------------------------------------
# 3. Pre-match features (leak-free)
# ---------------------------------------------------------------------------

def to_team_match_log(df):
    """Reshape one row per match into two rows: one per team.

    This 'long' shape is what makes rolling form easy to compute - a team's
    history is just its own rows in date order, whether it played home or away.
    """
    shared = ["MatchID", "Season", "MatchDate"]

    home = df[shared].copy()
    home["Team"] = df["HomeTeam"]
    home["Opponent"] = df["AwayTeam"]
    home["Venue"] = "H"
    home["GoalsFor"] = df["FullTimeHomeGoals"]
    home["GoalsAgainst"] = df["FullTimeAwayGoals"]
    home["Shots"] = df["HomeShots"]
    home["ShotsOnTarget"] = df["HomeShotsOnTarget"]
    home["Corners"] = df["HomeCorners"]
    home["Fouls"] = df["HomeFouls"]
    home["YellowCards"] = df["HomeYellowCards"]
    home["Points"] = df["HomePoints"]

    away = df[shared].copy()
    away["Team"] = df["AwayTeam"]
    away["Opponent"] = df["HomeTeam"]
    away["Venue"] = "A"
    away["GoalsFor"] = df["FullTimeAwayGoals"]
    away["GoalsAgainst"] = df["FullTimeHomeGoals"]
    away["Shots"] = df["AwayShots"]
    away["ShotsOnTarget"] = df["AwayShotsOnTarget"]
    away["Corners"] = df["AwayCorners"]
    away["Fouls"] = df["AwayFouls"]
    away["YellowCards"] = df["AwayYellowCards"]
    away["Points"] = df["AwayPoints"]

    log = pd.concat([home, away], ignore_index=True)
    return log.sort_values(["Team", "MatchDate"]).reset_index(drop=True)


def add_rolling_form(log, window=FORM_WINDOW):
    """Average each team's stats over its previous `window` matches.

    The .shift(1) is the critical line: it moves the window back by one match so
    the current result is never part of its own feature. Without it, form would
    contain the answer and the model would be cheating.

    Form carries across season boundaries, so a team's opening fixtures use the
    tail of its previous campaign rather than being thrown away.
    """
    log = log.sort_values(["Team", "MatchDate"]).copy()
    grouped = log.groupby("Team", sort=False)

    for stat in FORM_STATS:
        log[f"Form{stat}"] = grouped[stat].transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).mean()
        )

    # Season-to-date points per game: a longer-run strength signal that the
    # 5-match window is too short to capture.
    log["SeasonPPG"] = log.groupby(["Team", "Season"], sort=False)["Points"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    # Days since the team last played - a rest/fixture-congestion proxy.
    log["DaysSinceLastMatch"] = (
        grouped["MatchDate"].diff().dt.days
    )

    log["MatchesPlayed"] = grouped.cumcount()
    return log


def build_feature_table(df, log):
    """Join each team's pre-match form back onto its match, home and away.

    Produces Home*/Away* form columns plus their differentials, which are what a
    model actually keys on: not "how good is the home team" but "how much better
    is the home team than its opponent today".
    """
    form_cols = [f"Form{stat}" for stat in FORM_STATS] + [
        "SeasonPPG",
        "DaysSinceLastMatch",
        "MatchesPlayed",
    ]
    slim = log[["MatchID", "Team"] + form_cols]

    features = df[
        [
            "MatchID",
            "Season",
            "MatchDate",
            "HomeTeam",
            "AwayTeam",
            "FullTimeResult",
            "Result",
            "HomeWin",
        ]
    ].copy()

    features = features.merge(
        slim.add_prefix("Home"),
        left_on=["MatchID", "HomeTeam"],
        right_on=["HomeMatchID", "HomeTeam"],
        how="left",
    ).drop(columns=["HomeMatchID"])

    features = features.merge(
        slim.add_prefix("Away"),
        left_on=["MatchID", "AwayTeam"],
        right_on=["AwayMatchID", "AwayTeam"],
        how="left",
        suffixes=("", "_away"),
    ).drop(columns=["AwayMatchID"])

    for col in form_cols:
        features[f"{col}Diff"] = features[f"Home{col}"] - features[f"Away{col}"]

    return features.sort_values("MatchDate").reset_index(drop=True)


def build_features(df):
    """Build the model-ready pre-match feature table."""
    log = add_rolling_form(to_team_match_log(df))
    features = build_feature_table(df, log)

    n_all = len(features)
    # A team's first `FORM_WINDOW` matches have no complete history, so their
    # form is undefined. Those rows cannot be modelled and are dropped.
    features = features.dropna(subset=["FormPointsDiff"]).reset_index(drop=True)

    print("\nPre-match features")
    print(f"  form window              : previous {FORM_WINDOW} matches")
    print(f"  rows kept                : {len(features)} of {n_all} "
          f"({n_all - len(features)} dropped for insufficient history)")
    print(f"  feature columns          : "
          f"{sum(c.endswith('Diff') for c in features.columns)} differentials")

    # Some NaNs legitimately survive and are left in place rather than filled
    # with a guess: SeasonPPG is undefined for a team's first match of a season,
    # and form averages over the shot columns inherit the NaNs written by
    # flag_impossible_stats. Tree models (e.g. HistGradientBoostingClassifier)
    # accept NaN directly; impute or drop only if your model requires it.
    incomplete = int(features.isna().any(axis=1).sum())
    print(f"  rows with some NaN       : {incomplete} "
          f"(first match of a season, or repaired shot stats)")
    return features


# ---------------------------------------------------------------------------
# 4. Data quality notes worth knowing before you model
# ---------------------------------------------------------------------------

def report_caveats(df):
    """Print the structural issues that cleaning cannot fix."""
    per_season = df.groupby("Season").size()
    short = per_season[per_season < 380]

    print("\nCaveats")
    if len(short):
        print("  incomplete seasons (fewer than 380 matches):")
        for season, n in short.items():
            print(f"    {season}: {n} matches ({380 - n} missing)")
        print("    -> exclude these if you compare seasons or build league tables;")
        print("       individual match rows are still valid for modelling.")

    own_goals = int(
        (df["FullTimeHomeGoals"] > df["HomeShotsOnTarget"]).sum()
        + (df["FullTimeAwayGoals"] > df["AwayShotsOnTarget"]).sum()
    )
    print(f"  rows where goals exceed shots on target: {own_goals}")
    print("    -> expected, not an error: an own goal counts for the scoring")
    print("       team but is not one of that team's shots on target.")


def main():
    raw = load_raw()
    print(f"Loaded {len(raw)} matches, {raw['Season'].nunique()} seasons "
          f"({raw['Season'].min()} to {raw['Season'].max()})\n")

    cleaned = clean(raw)
    cleaned.to_csv(CLEAN_PATH, index=False)

    features = build_features(cleaned)
    features.to_csv(FEATURES_PATH, index=False)

    report_caveats(cleaned)

    print(f"\nWrote {CLEAN_PATH}    ({len(cleaned)} rows, {cleaned.shape[1]} cols)")
    print(f"Wrote {FEATURES_PATH} ({len(features)} rows, {features.shape[1]} cols)")


if __name__ == "__main__":
    main()
