from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT = Path("data/master/modelling_features.csv")
OUTPUT_DIR = Path("data/outputs/model_pl_stability")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


BIG_SIX = {
    "Arsenal",
    "Chelsea",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Tottenham",
}


TARGET = "top_15_next_2_seasons"


MODEL_SPECS = {
    "table_only": [
        "pts",
        "position",
        "gd",
    ],
    "underlying_performance": [
        "npxGD",
        "xGA",
        "xpts",
        "deep_diff",
        "ppda_coef",
        "xpts_diff",
    ],
    "financial_resource": [
        "revenue_rank",
        "wage_rank",
        "wage_to_revenue",
        "net_spend_to_revenue",
        "total_spent_to_revenue",
    ],
    "recruitment_efficiency": [
        "wage_efficiency_npxGD",
        "revenue_efficiency_npxGD",
        "net_spend_efficiency_npxGD",
        "total_spend_efficiency_npxGD",
    ],
    "squad_and_manager": [
        "avg_age",
        "age_std",
        "young_u21_pct",
        "veteran_o30_pct",
        "squad_size",
        "manager_count_season",
        "primary_manager_tenure",
        "manager_change_flag",
    ],
    "full_model": [
        "npxGD",
        "xGA",
        "xpts",
        "deep_diff",
        "ppda_coef",
        "revenue_rank",
        "wage_rank",
        "wage_to_revenue",
        "net_spend_to_revenue",
        "wage_efficiency_npxGD",
        "net_spend_efficiency_npxGD",
        "avg_age",
        "young_u21_pct",
        "veteran_o30_pct",
        "manager_change_flag",
    ],
}


def make_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def run_model(df, model_name, features, target, exclude_big_six=True):
    model_df = df.copy()

    if exclude_big_six:
        model_df = model_df[~model_df["team_name"].isin(BIG_SIX)].copy()

    features = [f for f in features if f in model_df.columns]

    model_df = model_df.dropna(subset=[target]).copy()

    # Avoid training on seasons where future outcome cannot be known reliably
    model_df = model_df[model_df["year"] <= 2022].copy()

    # Time-aware split
    train_df = model_df[model_df["year"] <= 2020].copy()
    test_df = model_df[model_df["year"] > 2020].copy()

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    pipe = make_pipeline()
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba) if y_test.nunique() > 1 else np.nan

    print(f"\n{model_name}")
    print("=" * len(model_name))
    print(f"Exclude Big Six: {exclude_big_six}")
    print(f"Features: {features}")
    print(f"ROC AUC: {auc:.3f}" if not np.isnan(auc) else "ROC AUC: n/a")
    print(classification_report(y_test, pred))

    coef_df = pd.DataFrame(
        {
            "model": model_name,
            "exclude_big_six": exclude_big_six,
            "feature": features,
            "coefficient": pipe.named_steps["model"].coef_[0],
        }
    )

    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

    scored = test_df[
        [
            "team_name",
            "year",
            "understat_season",
            "position",
            "pts",
            "npxGD",
            "xGA",
            "revenue",
            "wage_bill",
            target,
        ]
    ].copy()

    scored["predicted_stability_probability"] = proba
    scored["model"] = model_name
    scored["exclude_big_six"] = exclude_big_six

    return {
        "model": model_name,
        "exclude_big_six": exclude_big_six,
        "roc_auc": auc,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "features": features,
        "coefficients": coef_df,
        "scores": scored,
    }


def main():
    df = pd.read_csv(INPUT)

    if "big_six" not in df.columns:
        df["big_six"] = df["team_name"].isin(BIG_SIX).astype(int)

    results = []

    for model_name, features in MODEL_SPECS.items():
        results.append(
            run_model(
                df=df,
                model_name=model_name,
                features=features,
                target=TARGET,
                exclude_big_six=True,
            )
        )

    # Big Six sensitivity check
    results.append(
        run_model(
            df=df,
            model_name="full_model_all_clubs_sensitivity",
            features=MODEL_SPECS["full_model"],
            target=TARGET,
            exclude_big_six=False,
        )
    )

    summary = pd.DataFrame(
        [
            {
                "model": r["model"],
                "exclude_big_six": r["exclude_big_six"],
                "roc_auc": r["roc_auc"],
                "n_train": r["n_train"],
                "n_test": r["n_test"],
                "features": ", ".join(r["features"]),
            }
            for r in results
        ]
    )

    coefficients = pd.concat([r["coefficients"] for r in results], ignore_index=True)
    scores = pd.concat([r["scores"] for r in results], ignore_index=True)

    summary.to_csv(OUTPUT_DIR / "model_summary.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "model_coefficients.csv", index=False)
    scores.to_csv(OUTPUT_DIR / "model_scores.csv", index=False)

    print("\nSaved outputs to:")
    print(OUTPUT_DIR / "model_summary.csv")
    print(OUTPUT_DIR / "model_coefficients.csv")
    print(OUTPUT_DIR / "model_scores.csv")


if __name__ == "__main__":
    main()