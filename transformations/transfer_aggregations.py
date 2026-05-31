import pandas as pd
import argparse


def build_club_season_summary(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Ensure numeric columns are clean
    df["fee"] = pd.to_numeric(df["fee"], errors="coerce").fillna(0)
    df["is_loan"] = pd.to_numeric(df["is_loan"], errors="coerce").fillna(0).astype(int)

    # Money spent = incoming transfers
    incoming = (
        df[df["movement"] == "in"]
        .groupby(["season", "league", "club"], as_index=False)
        .agg(
            total_spent=("fee", "sum"),
            players_bought=("player_id", "count"),
            loans=("is_loan", "sum"),
        )
    )

    # Funds received = outgoing transfers
    outgoing = (
        df[df["movement"] == "out"]
        .groupby(["season", "league", "club"], as_index=False)
        .agg(
            funds_received=("fee", "sum"),
        )
    )

    summary = incoming.merge(
        outgoing,
        on=["season", "league", "club"],
        how="outer",
    )

    summary[["total_spent", "funds_received", "players_bought", "loans"]] = (
        summary[["total_spent", "funds_received", "players_bought", "loans"]]
        .fillna(0)
    )

    summary["players_bought"] = summary["players_bought"].astype(int)
    summary["loans"] = summary["loans"].astype(int)

    summary["net_spend"] = summary["total_spent"] - summary["funds_received"]

    summary = summary.sort_values(["season", "league", "club"])

    summary.to_csv(output_csv, index=False)


if __name__ == "__main__":
    build_club_season_summary(
        "data/filtered/transfermarkt/master_transfer_dataset.csv",
        "data/filtered/transfermarkt/transfers_aggregated.csv",
    )