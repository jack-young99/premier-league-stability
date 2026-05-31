from pathlib import Path

import numpy as np
import pandas as pd


PERF_PATH = "data/filtered/transfermarkt/player_performances.csv"
PROFILE_PATH = "data/filtered/transfermarkt/player_profiles.csv"
OUTPUT_PATH = "data/filtered/transfermarkt/team_season_profiles.csv"


def season_start_year(season_name: str):
    """
    Convert:
        20/21 -> 2020
        99/00 -> 1999
    """
    try:
        yy = int(str(season_name).split("/")[0])
        return 1900 + yy if yy >= 90 else 2000 + yy
    except Exception:
        return np.nan


# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------

perf = pd.read_csv(PERF_PATH)
profiles = pd.read_csv(PROFILE_PATH)

# ------------------------------------------------------------------
# Player-season aggregation
# ------------------------------------------------------------------

player_season = (
    perf.groupby(
        [
            "player_id",
            "season_name",
            "team_id",
            "team_name",
        ],
        as_index=False,
    )
    .agg(
        minutes_played=("minutes_played", "sum"),
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        clean_sheets=("clean_sheets", "sum"),
        goals_conceded=("goals_conceded", "sum"),
        appearances=("nb_in_group", "sum"),
    )
)

# ------------------------------------------------------------------
# Merge profiles
# ------------------------------------------------------------------

df = player_season.merge(
    profiles,
    on="player_id",
    how="left",
)

# ------------------------------------------------------------------
# Age calculation
# ------------------------------------------------------------------

df["date_of_birth"] = pd.to_datetime(
    df["date_of_birth"],
    errors="coerce",
)

df["season_start_year"] = df["season_name"].apply(
    season_start_year
)

df["season_start_date"] = pd.to_datetime(
    df["season_start_year"].astype("Int64").astype(str)
    + "-08-01",
    errors="coerce",
)

df["age"] = (
    (df["season_start_date"] - df["date_of_birth"])
    .dt.days
    / 365.25
)

# ------------------------------------------------------------------
# Tenure calculation
# ------------------------------------------------------------------

df["joined"] = pd.to_datetime(
    df["joined"],
    errors="coerce",
)

df["tenure_years"] = (
    (df["season_start_date"] - df["joined"])
    .dt.days
    / 365.25
)

# ------------------------------------------------------------------
# Position buckets
# ------------------------------------------------------------------

def simplify_position(x):
    if pd.isna(x):
        return "Unknown"

    x = str(x)

    if "Goalkeeper" in x:
        return "Goalkeeper"
    if "Defender" in x:
        return "Defender"
    if "Midfield" in x:
        return "Midfield"
    if "Forward" in x or "Striker" in x:
        return "Forward"

    return "Other"


df["position_group"] = df["position"].apply(
    simplify_position
)

# ------------------------------------------------------------------
# Foreign player flag
# ------------------------------------------------------------------

def foreign_player(row):
    """
    Simple heuristic:
    country_of_birth != club country isn't available.

    Use citizenship count > 1 as proxy for international profile.
    """
    citizenship = row.get("citizenship")

    if pd.isna(citizenship):
        return np.nan

    return int(len(str(citizenship).split()) > 1)


df["multi_nationality"] = df.apply(
    foreign_player,
    axis=1,
)

# ------------------------------------------------------------------
# Team-season aggregation
# ------------------------------------------------------------------

results = []

group_cols = ["team_id", "team_name", "season_name"]

for (team_id, team_name, season_name), g in df.groupby(group_cols):

    squad_size = g["player_id"].nunique()

    total_minutes = g["minutes_played"].sum()

    top5_share = np.nan
    if total_minutes > 0:
        top5_share = (
            g["minutes_played"]
            .nlargest(5)
            .sum()
            / total_minutes
        )

    row = {
        "team_id": team_id,
        "team_name": team_name,
        "season_name": season_name,

        # squad size
        "squad_size": squad_size,

        # age
        "avg_age": g["age"].mean(),
        "median_age": g["age"].median(),
        "age_std": g["age"].std(),
        "young_u21_pct": (g["age"] < 21).mean(),
        "veteran_o30_pct": (g["age"] > 30).mean(),

        # height
        "avg_height": g["height"].mean(),
        "height_std": g["height"].std(),

        # footedness
        "left_foot_pct": (g["foot"] == "left").mean(),
        "right_foot_pct": (g["foot"] == "right").mean(),
        "both_foot_pct": (g["foot"] == "both").mean(),

        # nationality
        "eu_player_pct": (
            g["is_eu"]
            .astype(str)
            .str.lower()
            .eq("true")
            .mean()
        ),
        "multi_nationality_pct": g["multi_nationality"].mean(),
        "unique_citizenships": (
            g["citizenship"]
            .dropna()
            .nunique()
        ),

        # minutes
        "total_minutes": total_minutes,
        "avg_minutes_per_player":
            g["minutes_played"].mean(),
        "median_minutes_per_player":
            g["minutes_played"].median(),

        "top5_minutes_share":
            top5_share,

        "players_1000_plus_minutes":
            (g["minutes_played"] >= 1000).sum(),

        "players_2000_plus_minutes":
            (g["minutes_played"] >= 2000).sum(),

        # tenure
        "avg_tenure_years":
            g["tenure_years"].mean(),

        "median_tenure_years":
            g["tenure_years"].median(),

        "joined_within_last_year_pct":
            (g["tenure_years"] < 1).mean(),

        # performance
        "total_goals":
            g["goals"].sum(),

        "total_assists":
            g["assists"].sum(),

        "total_clean_sheets":
            g["clean_sheets"].sum(),

        "goals_per_player":
            g["goals"].sum() / max(squad_size, 1),

        "assists_per_player":
            g["assists"].sum() / max(squad_size, 1),
    }

    # Position shares
    pos_counts = (
        g["position_group"]
        .value_counts(normalize=True)
        .to_dict()
    )

    row["goalkeeper_pct"] = pos_counts.get(
        "Goalkeeper", 0
    )
    row["defender_pct"] = pos_counts.get(
        "Defender", 0
    )
    row["midfielder_pct"] = pos_counts.get(
        "Midfield", 0
    )
    row["forward_pct"] = pos_counts.get(
        "Forward", 0
    )

    results.append(row)

team_profiles = pd.DataFrame(results)

team_profiles = team_profiles.sort_values(
    ["season_name", "team_name"]
)

Path(OUTPUT_PATH).parent.mkdir(
    parents=True,
    exist_ok=True,
)

team_profiles.to_csv(
    OUTPUT_PATH,
    index=False,
)

print(
    f"Saved {len(team_profiles):,} rows to {OUTPUT_PATH}"
)