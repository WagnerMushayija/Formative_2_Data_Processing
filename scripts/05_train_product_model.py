from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "Data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TRAIN_FILE = PROCESSED_DIR / "product_model_train.csv"
TEST_FILE = PROCESSED_DIR / "product_model_test.csv"
PREDICTIONS_FILE = PROCESSED_DIR / "product_model_test_predictions.csv"

METRICS_FILE = REPORTS_DIR / "product_model_metrics.csv"
EVALUATION_REPORT_FILE = REPORTS_DIR / "product_model_evaluation.txt"
BEST_MODEL_FILE = MODELS_DIR / "product_recommendation_best_model.joblib"

TARGET = "product_category"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_features = [c for c in X.columns if c not in categorical_features]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "loss": log_loss(y_test, y_proba, labels=model.named_steps["classifier"].classes_),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    preprocessor = build_preprocessor(X_train)

    candidate_models = {
        "logistic_regression": LogisticRegression(max_iter=1500, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
    }

    metrics_rows = []
    fitted_models = {}

    for model_name, classifier in candidate_models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(X_train, y_train)
        scores = evaluate_model(pipeline, X_test, y_test)

        metrics_rows.append(
            {
                "model": model_name,
                "accuracy": scores["accuracy"],
                "f1_macro": scores["f1_macro"],
                "loss": scores["loss"],
            }
        )
        fitted_models[model_name] = pipeline

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        by=["f1_macro", "accuracy"], ascending=False
    )
    metrics_df.to_csv(METRICS_FILE, index=False)

    best_model_name = metrics_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]
    joblib.dump(best_model, BEST_MODEL_FILE)

    y_pred = best_model.predict(X_test)
    predictions_df = test_df.copy()
    predictions_df["predicted_product_category"] = y_pred
    predictions_df.to_csv(PREDICTIONS_FILE, index=False)

    best_row = metrics_df.iloc[0]
    report_lines = [
        "Product Recommendation Model Evaluation",
        "=======================================",
        f"best_model: {best_model_name}",
        f"accuracy: {best_row['accuracy']:.4f}",
        f"f1_macro: {best_row['f1_macro']:.4f}",
        f"loss: {best_row['loss']:.4f}",
        "",
        "All Candidate Models",
        "--------------------",
    ]

    for _, row in metrics_df.iterrows():
        report_lines.append(
            f"{row['model']}: accuracy={row['accuracy']:.4f}, "
            f"f1_macro={row['f1_macro']:.4f}, loss={row['loss']:.4f}"
        )

    EVALUATION_REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved: {METRICS_FILE}")
    print(f"Saved: {EVALUATION_REPORT_FILE}")
    print(f"Saved: {BEST_MODEL_FILE}")
    print(f"Saved: {PREDICTIONS_FILE}")
    print(f"Best model: {best_model_name}")


if __name__ == "__main__":
    main()
