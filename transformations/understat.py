from understatapi import UnderstatClient
import pandas as pd


START_SEASON = 2016
END_SEASON = 2025  # 2025/26 season

rows = []

with UnderstatClient() as understat:
    league = understat.league(league="EPL")

    for season in range(START_SEASON, END_SEASON + 1):
        print(f"Fetching EPL {season}/{str(season + 1)[-2:]}...")

        matches = league.get_match_data(season=str(season))

        for match in matches:
            rows.append(
                {
                    "season": f"{season}/{str(season + 1)[-2:]}",
                    "match_id": match["id"],
                    "date": match["datetime"],
                    "home_team": match["h"]["title"],
                    "away_team": match["a"]["title"],
                    "home_goals": int(match["goals"]["h"]),
                    "away_goals": int(match["goals"]["a"]),
                    "home_xg": float(match["xG"]["h"]),
                    "away_xg": float(match["xG"]["a"]),
                }
            )

df = pd.DataFrame(rows)

print(df.head())
print(f"Total matches: {len(df)}")

df.to_csv("data/raw/epl_match_xg_2016_17_onwards.csv", index=False)