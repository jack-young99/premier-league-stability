from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT = Path("data/filtered/understat/team_performances_pl.csv")
OUTPUT_DIR = Path("data/outputs/model1_outputs")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df.rename(
        columns={
            "team_name": "team",
            "xG": "xg",
            "xGA": "xga",
            "npxG": "npxg",
            "npxGA": "npxga",
            "npxGD": "npxgd",
        }
    )

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["gd"] = df["scored"] - df["conceded"]
    df["xgd"] = df["xg"] - df["xga"]
    df["npxgd_calc"] = df["npxg"] - df["npxga"]

    # Over/under-performance
    df["goals_minus_xg"] = df["scored"] - df["xg"]
    df["goals_conceded_minus_xga"] = df["conceded"] - df["xga"]
    df["gd_minus_xgd"] = df["gd"] - df["xgd"]
    df["pts_minus_xpts"] = df["pts"] - df["xpts"]

    # Territory/style proxies
    df["deep_diff"] = df["deep"] - df["deep_allowed"]

    return df


def add_future_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["team", "year"])

    df["next_position"] = df.groupby("team")["position"].shift(-1)
    df["next_pts"] = df.groupby("team")["pts"].shift(-1)

    # Target 1: simplest survival target
    df["survived_next_season"] = df["next_position"].notna().astype(int)

    # Target 2: stronger version of stability
    df["top_15_next_season"] = (
        (df["next_position"].notna()) & (df["next_position"] <= 15)
    ).astype(int)

    # Target 3: mid-table next season
    df["midtable_next_season"] = (
        (df["next_position"].notna())
        & (df["next_position"] >= 8)
        & (df["next_position"] <= 15)
    ).astype(int)

    return df


def make_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def run_model(
    df: pd.DataFrame,
    model_name: str,
    feature_cols: list[str],
    target_col: str,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_df = df.dropna(subset=feature_cols + [target_col]).copy()

    train_df = model_df[model_df["year"] <= 2021].copy()
    test_df = model_df[model_df["year"] >= 2022].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    pipeline = make_pipeline()
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    print(f"\n{model_name}")
    print("=" * len(model_name))
    print(f"Target: {target_col}")
    print(f"Features: {feature_cols}")

    print("\nModel performance")
    print("-----------------")
    print(classification_report(y_test, pred))

    if y_test.nunique() > 1:
        auc = roc_auc_score(y_test, proba)
        print("ROC AUC:", round(auc, 3))
    else:
        auc = np.nan
        print("ROC AUC: not available because test set has one class only")

    coef_df = pd.DataFrame(
        {
            "model": model_name,
            "target": target_col,
            "feature": feature_cols,
            "coefficient": pipeline.named_steps["model"].coef_[0],
        }
    )

    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

    print("\nFeature coefficients")
    print("--------------------")
    print(coef_df[["feature", "coefficient"]].to_string(index=False))

    scored = test_df[
        [
            "team",
            "season",
            "year",
            "position",
            "pts",
            "xpts",
            "next_position",
            "next_pts",
            target_col,
        ]
    ].copy()

    scored[f"{model_name}_probability"] = proba

    safe_model_name = model_name.lower().replace(" ", "_")
    scored_path = OUTPUT_DIR / f"{safe_model_name}_{target_col}_scores.csv"
    coef_path = OUTPUT_DIR / f"{safe_model_name}_{target_col}_coefficients.csv"

    scored.to_csv(scored_path, index=False)
    coef_df.to_csv(coef_path, index=False)

    print(f"\nSaved scores to {scored_path}")
    print(f"Saved coefficients to {coef_path}")


def main() -> None:
    df = load_data(INPUT)
    df = add_features(df)
    df = add_future_targets(df)

    target_col = "survived_next_season"

    table_features = [
        "pts",
        "position",
        "gd",
    ]

    underlying_features = [
        "npxgd",
        "xga",
        "xpts",
        "pts_minus_xpts",
        "deep_diff",
        "ppda_coef",
    ]

    combined_features = [
        "pts",
        "position",
        "gd",
        "npxgd",
        "xga",
        "xpts",
        "pts_minus_xpts",
        "deep_diff",
        "ppda_coef",
    ]

    run_model(
        df=df,
        model_name="table_only",
        feature_cols=table_features,
        target_col=target_col,
    )

    run_model(
        df=df,
        model_name="underlying_only",
        feature_cols=underlying_features,
        target_col=target_col,
    )

    run_model(
        df=df,
        model_name="combined",
        feature_cols=combined_features,
        target_col=target_col,
    )


if __name__ == "__main__":
    main()