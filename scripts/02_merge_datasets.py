from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
PROCESSED_DIR = DATA_DIR / "processed"

CLEAN_TRANSACTIONS_FILE = PROCESSED_DIR / "customer_transactions_clean.csv"
CLEAN_SOCIAL_FILE = PROCESSED_DIR / "customer_social_profiles_clean.csv"
MERGED_FILE = PROCESSED_DIR / "customer_merged_for_product_model.csv"


def build_customer_social_features(social: pd.DataFrame) -> pd.DataFrame:
    sentiment_map = {"Negative": -1, "Neutral": 0, "Positive": 1}

    social = social.copy()
    social["review_sentiment_score"] = social["review_sentiment"].map(sentiment_map).fillna(0)

    # Aggregate social activity to one row per customer to avoid row multiplication on merge.
    customer_social = (
        social.groupby("customer_id_new", dropna=False)
        .agg(
            social_media_platform=("social_media_platform", lambda x: x.mode().iloc[0]),
            engagement_score_mean=("engagement_score", "mean"),
            purchase_interest_score_mean=("purchase_interest_score", "mean"),
            review_sentiment_score_mean=("review_sentiment_score", "mean"),
            social_profile_rows=("customer_id_new", "count"),
        )
        .reset_index()
    )

    return customer_social


def main() -> None:
    transactions = pd.read_csv(CLEAN_TRANSACTIONS_FILE, parse_dates=["purchase_date"])
    social = pd.read_csv(CLEAN_SOCIAL_FILE)

    customer_social = build_customer_social_features(social)

    merged = transactions.merge(customer_social, how="left", on="customer_id_new", validate="m:1")

    merged["purchase_year"] = merged["purchase_date"].dt.year
    merged["purchase_month"] = merged["purchase_date"].dt.month
    merged["purchase_day_of_week"] = merged["purchase_date"].dt.dayofweek
    merged["is_weekend_purchase"] = merged["purchase_day_of_week"].isin([5, 6]).astype(int)
    merged["high_value_purchase"] = (
        merged["purchase_amount"] >= merged["purchase_amount"].median()
    ).astype(int)

    merged.to_csv(MERGED_FILE, index=False)

    print(f"Saved: {MERGED_FILE}")
    print(f"Rows: {len(merged)}")
    print(f"Columns: {len(merged.columns)}")


if __name__ == "__main__":
    main()
