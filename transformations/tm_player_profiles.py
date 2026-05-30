from pathlib import Path

import pandas as pd


def main() -> None:
    raw_dir = Path("data/raw/transfermarkt")

    profiles_path = raw_dir / "player_profiles.csv"
    performances_path = raw_dir / "player_performances_pl.csv"

    profiles_df = pd.read_csv(profiles_path)
    performances_df = pd.read_csv(performances_path)

    player_ids = set(performances_df["player_id"].dropna().unique())

    filtered_profiles_df = profiles_df[
        profiles_df["player_id"].isin(player_ids)
    ]

    output_dir = Path("data/filtered/transfermarkt")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "player_profiles.csv"
    filtered_profiles_df.to_csv(output_path, index=False)

    print(
        f"Saved {len(filtered_profiles_df):,} rows "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()