from pathlib import Path
import pandas as pd

TEAM_PROFILE_PATH = "data/filtered/transfermarkt/team_season_profiles.csv"
MASTER_PATH = "data/master/modelling_data.csv"
OUTPUT_PATH = "data/master/modelling_data_with_team_profiles.csv"

FEATURES = [
    "avg_age",
    "age_std",
    "young_u21_pct",
    "veteran_o30_pct",
    "avg_height",
    "squad_size",
]

team_profiles = pd.read_csv(TEAM_PROFILE_PATH)
master = pd.read_csv(MASTER_PATH)

# Convert Transfermarkt season format like 16/17 into year 2016
team_profiles["year"] = (
    team_profiles["season_name"]
    .astype(str)
    .str.split("/")
    .str[0]
    .astype(int)
    .apply(lambda x: 1900 + x if x >= 90 else 2000 + x)
)

# Ensure join keys match types
team_profiles["team_id"] = pd.to_numeric(team_profiles["team_id"], errors="coerce")
master["transfermarkt_id"] = pd.to_numeric(master["transfermarkt_id"], errors="coerce")
master["year"] = pd.to_numeric(master["year"], errors="coerce")

team_profiles_for_join = team_profiles[
    ["team_id", "year"] + FEATURES
].rename(columns={"team_id": "transfermarkt_id"})

merged = master.merge(
    team_profiles_for_join,
    on=["transfermarkt_id", "year"],
    how="left",
    validate="many_to_one",
)

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(merged):,} rows to {OUTPUT_PATH}")
print("Missing values after join:")
print(merged[FEATURES].isna().mean().sort_values(ascending=False))