import pandas as pd
from utils.utils import extract_pl_ids


def main():
    # Load source files
    financials_df = pd.read_csv("data/raw/financials/pl_financials.csv")
    team_competitions_df = pd.read_csv(
        "data/raw/transfermarkt/team_competitions_seasons.csv"
    )

    # Get unique Premier League club IDs
    pl_club_ids = extract_pl_ids(financials_df, "club_id")

    # Filter rows
    filtered_df = team_competitions_df[
        (team_competitions_df["club_id"].isin(pl_club_ids))
        & (team_competitions_df["season_id"] > 2016)
    ]

    # Save output
    output_path = (
        "data/raw/transfermarkt/team_competitions_seasons_filtered.csv"
    )
    filtered_df.to_csv(output_path, index=False)

    print(f"Saved {len(filtered_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
