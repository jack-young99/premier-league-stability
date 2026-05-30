import pandas as pd


def main():
    player_performances_df = pd.read_csv(
        "data/raw/transfermarkt/player_performances_pl.csv"
    )

    filtered_df = player_performances_df[
        player_performances_df["competition_id"] == "GB1"
    ]

    season_start = player_performances_df["season_name"].str[:2].astype(int)

    filtered_df = filtered_df[
        (season_start >= 16) &
        (season_start <= 30)
    ]

    filtered_df.to_csv(
        "data/filtered/transfermarkt/player_performances.csv",
        index=False
    )


if __name__ == "__main__":
    main()