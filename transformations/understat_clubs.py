from understatapi import UnderstatClient
import pandas as pd

START_SEASON = 2016
END_SEASON = 2025

clubs = {}

with UnderstatClient() as understat:
    league = understat.league("EPL")

    for season in range(START_SEASON, END_SEASON + 1):
        matches = league.get_match_data(str(season))

        for match in matches:
            home = match["h"]
            away = match["a"]

            clubs[home["id"]] = home["title"]
            clubs[away["id"]] = away["title"]

club_df = pd.DataFrame(
    [
        {
            "understat_id": club_id,
            "understat_name": club_name,
        }
        for club_id, club_name in clubs.items()
    ]
).sort_values("understat_name")

club_df.to_csv("understat_clubs.csv", index=False)