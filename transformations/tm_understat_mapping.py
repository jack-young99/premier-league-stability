import pandas as pd


UNDERSTAT_PATH = "data/raw/understat/understat_clubs.csv"
TRANSFERMARKT_PATH = "data/filtered/transfermarkt/team_competitions_seasons_filtered.csv"
OUTPUT_PATH = "data/mapping/understat_transfermarkt_club_mapping.csv"


def normalise_name(name):
    return (
        str(name)
        .lower()
        .replace("&", "and")
        .replace("fc", "")
        .replace("afc", "")
        .replace(".", "")
        .strip()
    )


def main():
    understat_df = pd.read_csv(UNDERSTAT_PATH)
    transfermarkt_df = pd.read_csv(TRANSFERMARKT_PATH)

    transfermarkt_clubs = (
        transfermarkt_df[["club_id", "team_name"]]
        .drop_duplicates()
        .rename(
            columns={
                "club_id": "transfermarkt_id",
                "team_name": "transfermarkt_name",
            }
        )
    )

    understat_df["join_name"] = understat_df["understat_name"].apply(normalise_name)
    transfermarkt_clubs["join_name"] = transfermarkt_clubs[
        "transfermarkt_name"
    ].apply(normalise_name)

    mapping_df = understat_df.merge(
        transfermarkt_clubs,
        on="join_name",
        how="left",
    )

    mapping_df = mapping_df[
        [
            "understat_id",
            "understat_name",
            "transfermarkt_id",
            "transfermarkt_name",
        ]
    ]

    mapping_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved mapping to {OUTPUT_PATH}")
    print(f"Unmatched rows: {mapping_df['transfermarkt_id'].isna().sum()}")


if __name__ == "__main__":
    main()