from pathlib import Path

import pandas as pd


UNDERSTAT_PATH = Path("data/filtered/understat/team_performances_pl.csv")
TRANSFERS_PATH = Path("data/filtered/transfermarkt/transfers_aggregated.csv")
MAPPING_PATH = Path("data/mapping/understat_transfermarkt_club_mapping.csv")
OUTPUT_PATH = Path("data/master/modelling_data.csv")


def main() -> None:
    understat = pd.read_csv(UNDERSTAT_PATH)
    transfers = pd.read_csv(TRANSFERS_PATH)
    mapping = pd.read_csv(MAPPING_PATH)

    understat.columns = understat.columns.str.strip()
    transfers.columns = transfers.columns.str.strip()
    mapping.columns = mapping.columns.str.strip()

    # Drop 2015/16 because transfer data starts from 2016/17
    understat["year"] = understat["year"].astype(int)
    understat = understat[understat["year"] >= 2016].copy()

    # Clean join keys
    understat["team_id"] = understat["team_id"].astype(str).str.strip()
    mapping["understat_id"] = mapping["understat_id"].astype(str).str.strip()

    transfers["season"] = transfers["season"].astype(int)
    transfers["club"] = transfers["club"].astype(str).str.strip()
    mapping["transfermarkt_name"] = (
        mapping["transfermarkt_name"].astype(str).str.strip()
    )

    # Add Transfermarkt club name to Understat data
    understat_mapped = understat.merge(
        mapping[
            [
                "understat_id",
                "understat_name",
                "transfermarkt_id",
                "transfermarkt_name",
            ]
        ],
        left_on="team_id",
        right_on="understat_id",
        how="left",
        validate="many_to_one",
    )

    understat_mapped = understat_mapped.rename(
        columns={"transfermarkt_name": "transfermarkt_club"}
    )

    missing_mapping = understat_mapped[
        understat_mapped["transfermarkt_club"].isna()
    ]

    if not missing_mapping.empty:
        print("\nWARNING: missing Understat to Transfermarkt mappings:")
        print(
            missing_mapping[
                ["team_id", "team_name", "season", "year"]
            ]
            .drop_duplicates()
            .to_string(index=False)
        )

    transfer_cols = [
        "season",
        "league",
        "club",
        "total_spent",
        "players_bought",
        "loans",
        "funds_received",
        "net_spend",
    ]

    transfers = transfers[transfer_cols].copy()

    # Join on Understat year = Transfermarkt season
    # and mapped Transfermarkt club name = transfer club name.
    modelling_data = understat_mapped.merge(
        transfers,
        left_on=["year", "transfermarkt_club"],
        right_on=["season", "club"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_transfermarkt"),
    )

    missing_transfer_data = modelling_data[
        modelling_data["net_spend"].isna()
    ]

    if not missing_transfer_data.empty:
        print("\nWARNING: rows missing transfer data:")
        print(
            missing_transfer_data[
                [
                    "team_id",
                    "team_name",
                    "season",
                    "year",
                    "transfermarkt_club",
                ]
            ]
            .drop_duplicates()
            .to_string(index=False)
        )

    # Tidy duplicate / helper columns
    modelling_data = modelling_data.rename(
        columns={
            "season": "understat_season",
            "season_transfermarkt": "transfermarkt_season",
            "league": "understat_league",
            "league_transfermarkt": "transfermarkt_league",
            "club": "transfermarkt_joined_club",
        }
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    modelling_data.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved modelling data to: {OUTPUT_PATH}")
    print(f"Rows: {len(modelling_data):,}")
    print(f"Columns: {len(modelling_data.columns):,}")

    print("\nJoined transfer columns preview:")
    print(
        modelling_data[
            [
                "team_name",
                "understat_season",
                "year",
                "transfermarkt_club",
                "total_spent",
                "players_bought",
                "loans",
                "funds_received",
                "net_spend",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()