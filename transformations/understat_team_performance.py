import pandas as pd
from utils.utils import extract_pl_ids


def main():
    teams_df = pd.read_csv("data/mapping/understat_transfermarkt_club_mapping.csv")
    understat_df = pd.read_csv(
        "data/raw/understat/understat_team_performance.csv"
    )

    pl_team_ids = set(extract_pl_ids(teams_df, "understat_id"))

    filtered_df = understat_df[
        understat_df["team_id"].isin(pl_team_ids)
    ]

    filtered_df = filtered_df[filtered_df["year"] >= 2015]

    filtered_df.to_csv(
        "data/filtered/understat/team_performances_pl.csv",
        index=False
    )

if __name__ == "__main__":
    main()