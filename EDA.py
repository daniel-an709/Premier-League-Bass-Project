import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# --------------------------------------------------
# Colour settings
# --------------------------------------------------

# Team-level outcome colours
outcome_palette = {
    "Loss": "#E57373",   # red
    "Draw": "#FFD54F",   # yellow
    "Win": "#64B5F6"     # blue
}

# Match-level result colours
# Result is from the home team's perspective
result_palette = {
    "HomeWin": "#64B5F6",   # blue
    "Draw": "#FFD54F",       # yellow
    "AwayWin": "#E57373"     # red
}


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

df = pd.read_csv(
    "epl_clean.csv",
    parse_dates=["MatchDate"]
)

print("Dataset shape:", df.shape)
print("Number of seasons:", df["Season"].nunique())

print("\nMatch results:")
print(df["Result"].value_counts())

print("\nMatch result percentages:")
print(
    (df["Result"].value_counts(normalize=True) * 100).round(2)
)


# --------------------------------------------------
# 2. Create folder for figures
# --------------------------------------------------

figure_dir = Path("figures")
figure_dir.mkdir(exist_ok=True)


# --------------------------------------------------
# 3. Match result distribution
# --------------------------------------------------

order = ["HomeWin", "Draw", "AwayWin"]

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Result",
    order=order,
    hue="Result",
    palette=result_palette,
    legend=False,
    saturation=1
)

plt.title("Distribution of EPL Match Results")
plt.xlabel("Match Result")
plt.ylabel("Number of Matches")

plt.tight_layout()

plt.savefig(
    figure_dir / "result_distribution.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 4. Create team-level dataset
# --------------------------------------------------

home = df[
    [
        "MatchID",
        "Season",
        "MatchDate",
        "HomeTeam",
        "FullTimeResult",
        "HomeShots",
        "HomeShotsOnTarget",
        "HomeCorners",
        "HomeFouls",
        "HomeYellowCards",
        "HomeRedCards"
    ]
].copy()

home.columns = [
    "MatchID",
    "Season",
    "MatchDate",
    "Team",
    "FullTimeResult",
    "Shots",
    "ShotsOnTarget",
    "Corners",
    "Fouls",
    "YellowCards",
    "RedCards"
]

home["Venue"] = "Home"

home["Outcome"] = np.select(
    [
        home["FullTimeResult"] == "H",
        home["FullTimeResult"] == "D"
    ],
    [
        "Win",
        "Draw"
    ],
    default="Loss"
)


away = df[
    [
        "MatchID",
        "Season",
        "MatchDate",
        "AwayTeam",
        "FullTimeResult",
        "AwayShots",
        "AwayShotsOnTarget",
        "AwayCorners",
        "AwayFouls",
        "AwayYellowCards",
        "AwayRedCards"
    ]
].copy()

away.columns = [
    "MatchID",
    "Season",
    "MatchDate",
    "Team",
    "FullTimeResult",
    "Shots",
    "ShotsOnTarget",
    "Corners",
    "Fouls",
    "YellowCards",
    "RedCards"
]

away["Venue"] = "Away"

away["Outcome"] = np.select(
    [
        away["FullTimeResult"] == "A",
        away["FullTimeResult"] == "D"
    ],
    [
        "Win",
        "Draw"
    ],
    default="Loss"
)


team_df = pd.concat(
    [home, away],
    ignore_index=True
)

print("\nTeam-level dataset:")
print(team_df.head())
print("Shape:", team_df.shape)


# --------------------------------------------------
# 5. Summary statistics by outcome
# --------------------------------------------------

stats = [
    "Shots",
    "ShotsOnTarget",
    "Corners",
    "Fouls",
    "YellowCards",
    "RedCards"
]

summary = (
    team_df
    .groupby("Outcome")[stats]
    .mean()
    .round(2)
)

print("\nAverage statistics by match outcome:")
print(summary)


# --------------------------------------------------
# 6. Shots on target by match outcome
# --------------------------------------------------

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=team_df,
    x="Outcome",
    y="ShotsOnTarget",
    order=["Loss", "Draw", "Win"],
    hue="Outcome",
    palette=outcome_palette,
    legend=False,
    showfliers=False,
    saturation=1
)

plt.title("Shots on Target by Match Outcome")
plt.xlabel("Match Outcome")
plt.ylabel("Shots on Target")

plt.tight_layout()

plt.savefig(
    figure_dir / "shots_on_target_by_outcome.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# # 7. Total shots by match outcome
# # --------------------------------------------------
#
# plt.figure(figsize=(7, 5))
#
# sns.boxplot(
#     data=team_df,
#     x="Outcome",
#     y="Shots",
#     order=["Loss", "Draw", "Win"],
#     hue="Outcome",
#     palette=outcome_palette,
#     legend=False,
#     showfliers=False,
#     saturation=1
# )
#
# plt.title("Total Shots by Match Outcome")
# plt.xlabel("Match Outcome")
# plt.ylabel("Shots")
#
# plt.tight_layout()
#
# plt.savefig(
#     figure_dir / "shots_by_outcome.png",
#     dpi=300
# )
#
# plt.close()
#
#
# # --------------------------------------------------
# 8. Red card and match outcome
# --------------------------------------------------

team_df["HasRedCard"] = team_df["RedCards"] > 0

team_df["RedCardStatus"] = np.where(
    team_df["HasRedCard"],
    "Yes",
    "No"
)

red_summary = (
    pd.crosstab(
        team_df["RedCardStatus"],
        team_df["Outcome"],
        normalize="index"
    ) * 100
).round(2)

print("\nMatch outcome percentages by red card status:")
print(red_summary)


red_plot = red_summary[
    ["Loss", "Draw", "Win"]
]

red_plot = red_plot.loc[
    ["No", "Yes"]
]

red_plot.plot(
    kind="bar",
    figsize=(7, 5),
    color=[
        outcome_palette["Loss"],
        outcome_palette["Draw"],
        outcome_palette["Win"]
    ]
)

plt.title("Match Outcome by Red Card Status")
plt.xlabel("Received at Least One Red Card")
plt.ylabel("Percentage of Team Performances")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    figure_dir / "redcard_outcome.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# # 9. Shots on target difference by match result
# # --------------------------------------------------
#
# plt.figure(figsize=(7, 5))
#
# sns.boxplot(
#     data=df,
#     x="Result",
#     y="ShotsOnTargetDiff",
#     order=["AwayWin", "Draw", "HomeWin"],
#     hue="Result",
#     palette=result_palette,
#     legend=False,
#     showfliers=False,
#     saturation=1
# )
#
# # Zero means both teams had the same number of shots on target
# plt.axhline(
#     y=0,
#     color="gray",
#     linestyle="--",
#     linewidth=1
# )
#
# plt.title("Shots on Target Difference by Match Result")
# plt.xlabel("Match Result")
# plt.ylabel("Home Shots on Target - Away Shots on Target")
#
# plt.tight_layout()
#
# plt.savefig(
#     figure_dir / "shots_on_target_diff_by_result.png",
#     dpi=300
# )
#
# plt.close()
#
# # --------------------------------------------------
# 10. Correlation between match-stat differentials
# --------------------------------------------------

diff_columns = [
    "ShotsDiff",
    "ShotsOnTargetDiff",
    "CornersDiff",
    "FoulsDiff",
    "YellowCardsDiff",
    "RedCardsDiff",
    "ShotAccuracyDiff",
    "GoalDiff"
]

correlation_matrix = df[diff_columns].corr()

print("\nCorrelation matrix:")
print(correlation_matrix.round(2))


plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdYlBu",
    center=0,
    linewidths=0.5
)

plt.title("Correlation Between Match Statistics")
plt.tight_layout()

plt.savefig(
    figure_dir / "correlation_heatmap.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# # 11. Match outcome percentages by season
# # --------------------------------------------------
#
# # Keep only complete 380-match seasons
# season_counts = df.groupby("Season").size()
#
# complete_seasons = season_counts[
#     season_counts >= 380
# ].index
#
# season_df = df[
#     df["Season"].isin(complete_seasons)
# ].copy()
#
#
# # Calculate result percentages within each season
# season_results = (
#     pd.crosstab(
#         season_df["Season"],
#         season_df["Result"],
#         normalize="index"
#     ) * 100
# )
#
# season_results = season_results[
#     ["HomeWin", "Draw", "AwayWin"]
# ]
#
#
# # Plot
# plt.figure(figsize=(11, 6))
#
# plt.plot(
#     season_results.index,
#     season_results["HomeWin"],
#     marker="o",
#     color=result_palette["HomeWin"],
#     label="Home Win"
# )
#
# plt.plot(
#     season_results.index,
#     season_results["Draw"],
#     marker="o",
#     color=result_palette["Draw"],
#     label="Draw"
# )
#
# plt.plot(
#     season_results.index,
#     season_results["AwayWin"],
#     marker="o",
#     color=result_palette["AwayWin"],
#     label="Away Win"
# )
#
# plt.title("EPL Match Outcomes Across Seasons")
# plt.xlabel("Season")
# plt.ylabel("Percentage of Matches")
#
# plt.xticks(rotation=45)
# plt.legend()
#
# plt.tight_layout()
#
# plt.savefig(
#     figure_dir / "match_outcomes_by_season.png",
#     dpi=300
# )
#
# plt.close()
#
# # --------------------------------------------------
# 12. Feature predictive ranking
# --------------------------------------------------

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Load leak-free pre-match features
features_df = pd.read_csv(
    "epl_features.csv",
    parse_dates=["MatchDate"]
)

# Make sure matches are chronological
features_df = features_df.sort_values("MatchDate").reset_index(drop=True)

# Target: 1 = Home Win, 0 = Draw or Away Win
y = features_df["HomeWin"]

# Use the pre-match differential features
feature_cols = [
    col for col in features_df.columns
    if col.endswith("Diff")
]

print("\nFeatures used for predictive ranking:")
print(feature_cols)


# Time-based cross validation
tscv = TimeSeriesSplit(n_splits=5)

results = []

for feature in feature_cols:

    X = features_df[[feature]]

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(max_iter=1000)
        )
    ])

    auc_scores = cross_val_score(
        model,
        X,
        y,
        cv=tscv,
        scoring="roc_auc"
    )

    results.append({
        "Feature": feature,
        "MeanAUC": auc_scores.mean(),
        "StdAUC": auc_scores.std()
    })


# Convert to dataframe and rank
feature_ranking = (
    pd.DataFrame(results)
    .sort_values("MeanAUC", ascending=False)
    .reset_index(drop=True)
)

print("\nFeature predictive ranking:")
print(feature_ranking.round(3))


# --------------------------------------------------
# Ranking plot
# --------------------------------------------------

plot_data = feature_ranking.sort_values(
    "MeanAUC",
    ascending=True
)

plt.figure(figsize=(9, 7))

bars = plt.barh(
    plot_data["Feature"],
    plot_data["MeanAUC"],
    color="#64B5F6"
)

# Random prediction baseline
plt.axvline(
    x=0.5,
    color="#E57373",
    linestyle="--",
    linewidth=1.5,
    label="Random prediction (AUC = 0.5)"
)

# Add AUC numbers to bars
for bar, value in zip(bars, plot_data["MeanAUC"]):
    plt.text(
        value + 0.003,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        va="center"
    )

plt.title("Predictive Strength of Pre-Match Features")
plt.xlabel("Cross-Validated ROC AUC")
plt.ylabel("Feature")
plt.legend()

plt.xlim(
    0.45,
    max(0.60, plot_data["MeanAUC"].max() + 0.04)
)

plt.tight_layout()

plt.savefig(
    figure_dir / "feature_predictive_ranking.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 12. Home win rate across seasons
# --------------------------------------------------

# Keep only complete seasons
season_counts = df.groupby("Season").size()

complete_seasons = season_counts[
    season_counts >= 380
].index

season_df = df[
    df["Season"].isin(complete_seasons)
].copy()


# Calculate home win percentage by season
home_win_by_season = (
    season_df
    .groupby("Season")["HomeWin"]
    .mean()
    .mul(100)
)


# Plot
plt.figure(figsize=(11, 5))

plt.plot(
    home_win_by_season.index,
    home_win_by_season.values,
    marker="o",
    color="#64B5F6",
    linewidth=2
)

# Overall average reference line
overall_home_win = season_df["HomeWin"].mean() * 100

plt.axhline(
    y=overall_home_win,
    color="#E57373",
    linestyle="--",
    linewidth=1.5,
    label=f"Overall average ({overall_home_win:.1f}%)"
)

plt.title("Home Win Rate Across EPL Seasons")
plt.xlabel("Season")
plt.ylabel("Home Win Percentage")

plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()

plt.savefig(
    figure_dir / "home_win_rate_by_season.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# Pre-match bridge:
# Recent shots-on-target form vs future result
# --------------------------------------------------

features_df = pd.read_csv(
    "epl_features.csv",
    parse_dates=["MatchDate"]
)

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=features_df,
    x="Result",
    y="FormShotsOnTargetDiff",
    order=["AwayWin", "Draw", "HomeWin"],
    hue="Result",
    palette=result_palette,
    legend=False,
    showfliers=False,
    saturation=1
)

plt.axhline(
    y=0,
    color="gray",
    linestyle="--",
    linewidth=1
)

plt.title("Recent Shots on Target Form by Match Result")
plt.xlabel("Match Result")
plt.ylabel("Previous 5-Match Shots on Target Difference")

plt.tight_layout()

plt.savefig(
    figure_dir / "form_shots_on_target_diff_by_result.png",
    dpi=300
)

plt.close()

keep_figures = {
    "result_distribution.png",
    "home_win_rate_by_season.png",
    "shots_on_target_by_outcome.png",
    "correlation_heatmap.png",
    "form_shots_on_target_diff_by_result.png",
    "feature_predictive_ranking.png",
    "redcard_outcome.png"
}