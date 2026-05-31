from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/master/modelling_data_with_team_profiles_and_managers.csv")
OUTPUT_PATH = Path("data/master/modelling_features.csv")


BIG_SIX = {
    "Arsenal",
    "Chelsea",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Tottenham",
}


def safe_divide(a, b):
    return np.where((b.notna()) & (b != 0), a / b, np.nan)


def add_rank(df, col, rank_col, ascending=False):
    if col in df.columns:
        df[rank_col] = df.groupby("year")[col].rank(
            ascending=ascending,
            method="min",
        )
    return df


def main():
    df = pd.read_csv(INPUT_PATH)

    # Basic performance indicators
    df["gd"] = df["scored"] - df["conceded"]
    df["xGD"] = df["xG"] - df["xGA"]
    df["npxGD_calc"] = df["npxG"] - df["npxGA"]

    df["goals_minus_xG"] = df["scored"] - df["xG"]
    df["goals_conceded_minus_xGA"] = df["conceded"] - df["xGA"]
    df["gd_minus_xGD"] = df["gd"] - df["xGD"]
    df["pts_minus_xpts"] = df["pts"] - df["xpts"]

    df["deep_diff"] = df["deep"] - df["deep_allowed"]
    df["deep_ratio"] = safe_divide(df["deep"], df["deep_allowed"])

    # Club state flags
    df["big_six"] = df["team_name"].isin(BIG_SIX).astype(int)
    df["qualified_for_europe"] = (df["position"] <= 7).astype(int)
    df["top_half"] = (df["position"] <= 10).astype(int)
    df["mid_table_band"] = df["position"].between(8, 15).astype(int)
    df["at_risk_position"] = (df["position"] >= 16).astype(int)
    df["relegated"] = (df["position"] >= 18).astype(int)

    # Transfer / recruitment indicators
    df["net_spend_per_point"] = safe_divide(df["net_spend"], df["pts"])
    df["total_spent_per_point"] = safe_divide(df["total_spent"], df["pts"])
    df["net_spend_per_xpt"] = safe_divide(df["net_spend"], df["xpts"])
    df["total_spent_per_xpt"] = safe_divide(df["total_spent"], df["xpts"])

    # Season ranks: rank 1 = best / highest unless otherwise stated
    df = add_rank(df, "pts", "points_rank", ascending=False)
    df = add_rank(df, "xpts", "xpts_rank", ascending=False)
    df = add_rank(df, "gd", "gd_rank", ascending=False)
    df = add_rank(df, "xGD", "xGD_rank", ascending=False)
    df = add_rank(df, "npxGD", "npxGD_rank", ascending=False)
    df = add_rank(df, "xGA", "xGA_rank", ascending=True)
    df = add_rank(df, "npxGA", "npxGA_rank", ascending=True)

    df = add_rank(df, "total_spent", "total_spent_rank", ascending=False)
    df = add_rank(df, "net_spend", "net_spend_rank", ascending=False)
    df = add_rank(df, "funds_received", "funds_received_rank", ascending=False)
    df = add_rank(df, "avg_age", "avg_age_rank", ascending=True)
    df = add_rank(df, "squad_size", "squad_size_rank", ascending=False)

    # Recruitment efficiency proxies:
    # positive = better underlying performance than spend rank implies
    if {"net_spend_rank", "npxGD_rank"}.issubset(df.columns):
        df["net_spend_efficiency_npxGD"] = df["net_spend_rank"] - df["npxGD_rank"]

    if {"total_spent_rank", "npxGD_rank"}.issubset(df.columns):
        df["total_spend_efficiency_npxGD"] = (
            df["total_spent_rank"] - df["npxGD_rank"]
        )

    if {"net_spend_rank", "position"}.issubset(df.columns):
        df["net_spend_efficiency_position"] = df["net_spend_rank"] - df["position"]

    # Squad profile indicators
    df["young_plus_peak_balance"] = df["young_u21_pct"] - df["veteran_o30_pct"]
    df["age_risk_flag"] = (
        (df["veteran_o30_pct"] > df["veteran_o30_pct"].median())
        & (df["young_u21_pct"] < df["young_u21_pct"].median())
    ).astype(int)

    # Managerial stability
    df["multiple_managers"] = (df["manager_count_season"] > 1).astype(int)

    # Financial indicators, only if columns exist
    if {"revenue", "wage_bill"}.issubset(df.columns):
        df["wage_to_revenue"] = safe_divide(df["wage_bill"], df["revenue"])
        df["net_spend_to_revenue"] = safe_divide(df["net_spend"], df["revenue"])
        df["total_spent_to_revenue"] = safe_divide(df["total_spent"], df["revenue"])
        df["wage_per_point"] = safe_divide(df["wage_bill"], df["pts"])
        df["revenue_per_point"] = safe_divide(df["revenue"], df["pts"])

        df = add_rank(df, "revenue", "revenue_rank", ascending=False)
        df = add_rank(df, "wage_bill", "wage_rank", ascending=False)

        if {"wage_rank", "npxGD_rank"}.issubset(df.columns):
            df["wage_efficiency_npxGD"] = df["wage_rank"] - df["npxGD_rank"]

        if {"revenue_rank", "npxGD_rank"}.issubset(df.columns):
            df["revenue_efficiency_npxGD"] = df["revenue_rank"] - df["npxGD_rank"]

        if {"wage_rank", "position"}.issubset(df.columns):
            df["wage_efficiency_position"] = df["wage_rank"] - df["position"]

    else:
        print(
            "WARNING: revenue and/or wage_bill not found. "
            "Skipping wage/revenue indicators."
        )

    # Future outcome labels
    df = df.sort_values(["team_name", "year"])

    df["next_position"] = df.groupby("team_name")["position"].shift(-1)
    df["next_pts"] = df.groupby("team_name")["pts"].shift(-1)
    df["next_npxGD"] = df.groupby("team_name")["npxGD"].shift(-1)

    df["survived_next_season"] = df["next_position"].notna().astype(int)
    df["top_15_next_season"] = (
        df["next_position"].notna() & (df["next_position"] <= 15)
    ).astype(int)
    df["mid_table_next_season"] = (
        df["next_position"].notna()
        & df["next_position"].between(8, 15)
    ).astype(int)

    # Two-year stability target
    df["position_t_plus_2"] = df.groupby("team_name")["position"].shift(-2)

    df["survived_next_2_seasons"] = (
        df["next_position"].notna() & df["position_t_plus_2"].notna()
    ).astype(int)

    df["top_15_next_2_seasons"] = (
        df["next_position"].notna()
        & df["position_t_plus_2"].notna()
        & (df["next_position"] <= 15)
        & (df["position_t_plus_2"] <= 15)
    ).astype(int)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")


if __name__ == "__main__":
    main()