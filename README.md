# Formative 2 Data Processing

Production-style data processing and machine learning pipeline for customer product-category prediction.

## Overview

This project:

- Cleans raw transaction and social profile datasets.
- Merges both sources into a modeling-ready table.
- Validates data quality after merging.
- Prepares train/test data splits.
- Trains and evaluates candidate classification models.
- Saves the best model and prediction artifacts.

## Project Structure

```text
Formative_2_Data_Processing/
|-- Data/
|   |-- customer_transactions.csv
|   |-- customer_social_profiles.csv
|   |-- merged_customer_data.csv
|   `-- processed/
|       |-- customer_transactions_clean.csv
|       |-- customer_social_profiles_clean.csv
|       |-- customer_merged_for_product_model.csv
|       |-- merge_validation_report.txt
|       |-- product_model_dataset.csv
|       |-- product_model_train.csv
|       |-- product_model_test.csv
|       |-- product_model_test_predictions.csv
|       `-- product_model_prep_metadata.txt
|-- models/
|   |-- product_recommendation_best_model.joblib
|   `-- training_summary.csv
|-- reports/
|   |-- product_model_metrics.csv
|   `-- product_model_evaluation.txt
|-- scripts/
|-- requirements.txt
`-- README.md
```

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If script execution is restricted in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Pipeline Execution

Run from the repository root in this exact order:



## Pipeline Stages

1. `01_prepare_sources.py`
- Cleans and standardizes source files.
- Outputs clean transaction and social CSVs.

2. `02_merge_datasets.py`
- Aggregates social activity at customer level.
- Merges with transactions and creates date-derived features.

3. `03_validate_merge.py`
- Generates merge validation metrics such as duplicate IDs, null patterns, and social match rate.

4. `04_prepare_product_model_data.py`
- Selects model features.
- Handles missing values.
- Splits data into stratified train and test sets.

5. `05_train_product_model.py`
- Trains multiple candidate models.
- Compares metrics and selects best model.
- Exports evaluation reports and test predictions.

## Key Outputs

- Processed datasets:
	- `Data/processed/product_model_train.csv`
	- `Data/processed/product_model_test.csv`
	- `Data/processed/product_model_test_predictions.csv`
- Reports:
	- `reports/product_model_metrics.csv`
	- `reports/product_model_evaluation.txt`
- Model artifact:
	- `models/product_recommendation_best_model.joblib`

## Current Evaluation Snapshot

Latest evaluation report indicates:

- Best model: `random_forest`
- Accuracy: `0.1667`
- F1 macro: `0.1365`
- Log loss: `1.7924`

## Additional Components

This repository also includes optional multimodal scripts for audio/image feature processing and identity workflows:

- `scripts/audio_processing.py`
- `scripts/image_processing.py`
- `scripts/model_trainer.py`
- `scripts/predictor.py`
- `scripts/demo.py`

These are separate from the main product-category pipeline described above.

## Reproducibility Notes

- Keep input file names and paths unchanged under `Data/`.
- Run scripts sequentially to ensure dependent artifacts are available.
- Use the same Python environment and package versions for consistent results.
