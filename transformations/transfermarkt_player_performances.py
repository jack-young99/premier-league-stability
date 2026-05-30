import pandas as pd
from utils.utils import extract_pl_ids


def main():
    teams_df = pd.read_csv("data/mapping/understat_transfermarkt_club_mapping.csv")
    transfermarkt_df = pd.read_csv(
        "data/raw/transfermarkt/player_performances.csv"
    )

    pl_team_ids = set(extract_pl_ids(teams_df, "transfermarkt_id"))

    filtered_df = transfermarkt_df[
        transfermarkt_df["team_id"].isin(pl_team_ids)
    ]

    filtered_df.to_csv(
        "data/raw/transfermarkt/player_performances_pl.csv",
        index=False
    )

if __name__ == "__main__":
    main()