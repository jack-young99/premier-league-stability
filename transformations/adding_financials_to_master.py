from pathlib import Path

import pandas as pd


MODELLING_PATH = Path(
    "data/master/modelling_data_with_team_profiles_and_managers.csv"
)

FINANCIALS_PATH = Path(
    "data/raw/financials/pl_financials.csv"
)

OUTPUT_PATH = Path(
    "data/master/modelling_data_with_team_profiles_and_managers.csv"
)


def main() -> None:
    modelling = pd.read_csv(MODELLING_PATH)
    financials = pd.read_csv(FINANCIALS_PATH)

    modelling.columns = modelling.columns.str.strip()
    financials.columns = financials.columns.str.strip()

    financials["club_id"] = financials["club_id"].astype(int)
    financials["season_id"] = financials["season_id"].astype(int)

    modelling["transfermarkt_id"] = modelling["transfermarkt_id"].astype(int)
    modelling["year"] = modelling["year"].astype(int)

    financials = financials[
        [
            "club_id",
            "season_id",
            "revenue",
            "wage_bill",
        ]
    ].copy()

    merged = modelling.merge(
        financials,
        left_on=["transfermarkt_id", "year"],
        right_on=["club_id", "season_id"],
        how="left",
        validate="many_to_one",
    )

    merged = merged.drop(
        columns=[
            "club_id",
            "season_id",
        ]
    )

    missing = merged["revenue"].isna().sum()

    print(f"Rows: {len(merged):,}")
    print(f"Missing financial rows: {missing:,}")

    if missing:
        print("\nExample missing rows:")
        print(
            merged.loc[
                merged["revenue"].isna(),
                [
                    "team_name",
                    "year",
                    "transfermarkt_id",
                    "transfermarkt_club",
                ],
            ]
            .head(20)
            .to_string(index=False)
        )

    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved: {OUTPUT_PATH}")

    print("\nPreview:")
    print(
        merged[
            [
                "team_name",
                "year",
                "revenue",
                "wage_bill",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()