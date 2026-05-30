from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/transfermarkt")
OUTPUT_DIR = Path("data/filtered/transfermarkt")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dfs = []

for file in sorted(RAW_DIR.glob("2*.csv")):
    season = file.stem.split("_")[0]  # e.g. 2020 from 2020_transfers.csv

    df = pd.read_csv(file)
    df["source_file"] = file.name
    df["season"] = season

    dfs.append(df)

master_df = pd.concat(dfs, ignore_index=True)

master_df.to_csv(
    OUTPUT_DIR / "master_transfer_dataset.csv",
    index=False,
)

print(f"Combined {len(dfs)} files")
print(f"Rows: {len(master_df):,}")
print(f"Saved to {OUTPUT_DIR / 'master_transfer_dataset.csv'}")