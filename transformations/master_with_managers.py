from pathlib import Path
import pandas as pd

INPUT_PATH = "data/master/modelling_data_with_team_profiles.csv"
MANAGER_PATH = "data/raw/understat/managerial_stability.csv"
OUTPUT_PATH = "data/master/modelling_data_with_team_profiles_and_managers.csv"

MANAGER_FEATURES = [
    "manager_count_season",
    "primary_manager_tenure",
    "manager_change_flag",
]

master = pd.read_csv(INPUT_PATH)
managers = pd.read_csv(MANAGER_PATH)

# Ensure join keys are clean and same type
master["team_id"] = pd.to_numeric(master["team_id"], errors="coerce")
master["year"] = pd.to_numeric(master["year"], errors="coerce")

managers["team_id"] = pd.to_numeric(managers["team_id"], errors="coerce")
managers["year"] = pd.to_numeric(managers["year"], errors="coerce")

managers_for_join = managers[
    ["team_id", "year"] + MANAGER_FEATURES
].drop_duplicates(subset=["team_id", "year"])

merged = master.merge(
    managers_for_join,
    on=["team_id", "year"],
    how="left",
    validate="many_to_one",
)

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(merged):,} rows to {OUTPUT_PATH}")

print("\nMissing values after managerial stability join:")
print(
    merged[MANAGER_FEATURES]
    .isna()
    .mean()
    .sort_values(ascending=False)
)