from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "Data" / "processed"

MERGED_FILE = PROCESSED_DIR / "customer_merged_for_product_model.csv"
FEATURE_DATA_FILE = PROCESSED_DIR / "product_model_dataset.csv"
TRAIN_FILE = PROCESSED_DIR / "product_model_train.csv"
TEST_FILE = PROCESSED_DIR / "product_model_test.csv"
META_FILE = PROCESSED_DIR / "product_model_prep_metadata.txt"

TARGET = "product_category"


def main() -> None:
    df = pd.read_csv(MERGED_FILE)

    df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")
    df["purchase_day_of_year"] = df["purchase_date"].dt.dayofyear

    selected_columns = [
        "purchase_amount",
        "customer_rating",
        "social_media_platform",
        "engagement_score_mean",
        "purchase_interest_score_mean",
        "review_sentiment_score_mean",
        "social_profile_rows",
        "purchase_month",
        "purchase_day_of_week",
        "is_weekend_purchase",
        "high_value_purchase",
        "purchase_day_of_year",
        TARGET,
    ]

    model_df = df[selected_columns].copy()

    numeric_columns = [
        "purchase_amount",
        "customer_rating",
        "engagement_score_mean",
        "purchase_interest_score_mean",
        "review_sentiment_score_mean",
        "social_profile_rows",
        "purchase_month",
        "purchase_day_of_week",
        "is_weekend_purchase",
        "high_value_purchase",
        "purchase_day_of_year",
    ]
    categorical_columns = ["social_media_platform"]

    for col in numeric_columns:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
        model_df[col] = model_df[col].fillna(model_df[col].median())

    for col in categorical_columns:
        model_df[col] = model_df[col].fillna("Unknown").astype(str)

    model_df = model_df.dropna(subset=[TARGET]).reset_index(drop=True)

    train_df, test_df = train_test_split(
        model_df,
        test_size=0.2,
        random_state=42,
        stratify=model_df[TARGET],
    )

    model_df.to_csv(FEATURE_DATA_FILE, index=False)
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    metadata_lines = [
        "Product Model Data Preparation Metadata",
        "======================================",
        f"source_rows: {len(df)}",
        f"model_rows: {len(model_df)}",
        f"train_rows: {len(train_df)}",
        f"test_rows: {len(test_df)}",
        f"target: {TARGET}",
        f"classes: {', '.join(sorted(model_df[TARGET].unique()))}",
        f"numeric_features: {', '.join(numeric_columns)}",
        f"categorical_features: {', '.join(categorical_columns)}",
    ]
    META_FILE.write_text("\n".join(metadata_lines), encoding="utf-8")

    print(f"Saved: {FEATURE_DATA_FILE}")
    print(f"Saved: {TRAIN_FILE}")
    print(f"Saved: {TEST_FILE}")
    print(f"Saved: {META_FILE}")


if __name__ == "__main__":
    main()
