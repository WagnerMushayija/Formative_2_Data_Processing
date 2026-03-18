from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
PROCESSED_DIR = DATA_DIR / "processed"


TRANSACTIONS_FILE = DATA_DIR / "customer_transactions.csv"
SOCIAL_FILE = DATA_DIR / "customer_social_profiles.csv"

CLEAN_TRANSACTIONS_FILE = PROCESSED_DIR / "customer_transactions_clean.csv"
CLEAN_SOCIAL_FILE = PROCESSED_DIR / "customer_social_profiles_clean.csv"


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    transactions = pd.read_csv(TRANSACTIONS_FILE)
    social = pd.read_csv(SOCIAL_FILE)

    transactions.columns = [c.strip() for c in transactions.columns]
    social.columns = [c.strip() for c in social.columns]

    transactions["customer_id_legacy"] = pd.to_numeric(
        transactions["customer_id_legacy"], errors="coerce"
    ).astype("Int64")
    transactions["purchase_amount"] = pd.to_numeric(
        transactions["purchase_amount"], errors="coerce"
    )
    transactions["customer_rating"] = pd.to_numeric(
        transactions["customer_rating"], errors="coerce"
    )
    transactions["purchase_date"] = pd.to_datetime(
        transactions["purchase_date"], errors="coerce"
    )

    transactions["customer_id_new"] = transactions["customer_id_legacy"].map(
        lambda x: f"A{int(x)}" if pd.notna(x) else pd.NA
    )

    social["engagement_score"] = pd.to_numeric(social["engagement_score"], errors="coerce")
    social["purchase_interest_score"] = pd.to_numeric(
        social["purchase_interest_score"], errors="coerce"
    )

    transactions = transactions.drop_duplicates(subset=["transaction_id"])
    social = social.drop_duplicates(
        subset=["customer_id_new", "social_media_platform"], keep="last"
    )

    transactions.to_csv(CLEAN_TRANSACTIONS_FILE, index=False)
    social.to_csv(CLEAN_SOCIAL_FILE, index=False)

    print(f"Saved: {CLEAN_TRANSACTIONS_FILE}")
    print(f"Saved: {CLEAN_SOCIAL_FILE}")


if __name__ == "__main__":
    main()
