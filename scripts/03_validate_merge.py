from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "Data" / "processed"
MERGED_FILE = PROCESSED_DIR / "customer_merged_for_product_model.csv"
REPORT_FILE = PROCESSED_DIR / "merge_validation_report.txt"


def main() -> None:
    merged = pd.read_csv(MERGED_FILE)

    total_rows = len(merged)
    duplicate_transaction_ids = int(merged["transaction_id"].duplicated().sum())
    missing_customer_key = int(merged["customer_id_new"].isna().sum())

    missing_social_fields = (
        merged[
            [
                "engagement_score_mean",
                "purchase_interest_score_mean",
                "review_sentiment_score_mean",
                "social_profile_rows",
            ]
        ]
        .isna()
        .all(axis=1)
        .sum()
    )

    social_match_rate = 1 - (missing_social_fields / total_rows if total_rows else 0)

    report_lines = [
        "Merge Validation Report",
        "=======================",
        f"total_rows: {total_rows}",
        f"duplicate_transaction_ids: {duplicate_transaction_ids}",
        f"missing_customer_id_new: {missing_customer_key}",
        f"rows_missing_all_social_features: {int(missing_social_fields)}",
        f"social_match_rate: {social_match_rate:.4f}",
        "",
        "Column Null Counts",
        "------------------",
    ]

    null_counts = merged.isna().sum().sort_values(ascending=False)
    report_lines.extend([f"{col}: {int(cnt)}" for col, cnt in null_counts.items()])

    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved: {REPORT_FILE}")
    print(f"social_match_rate: {social_match_rate:.4f}")


if __name__ == "__main__":
    main()
