# Formative 2 - Data Preprocessing Assignment

## User Identity and Product Recommendation System

### Project Overview

This project implements a multimodal authentication and product recommendation system that combines facial recognition, voice verification, and customer behavior analysis to provide personalized product recommendations. The system follows a sequential authentication flow where users must pass both facial and voice verification before receiving product recommendations.

### System Architecture

The system implements the following flow:

1. **Face Recognition**: User submits facial image for identity verification
2. **Product Recommendation**: If recognized, system retrieves customer data for product prediction
3. **Voice Verification**: User provides voice sample to confirm transaction
4. **Access Control**: System grants access only if both face and voice match the same user

### Team Members and Contributions

**Task Division:**

- **Task 1 - Data Merge**: Wagner Mushayija
- **Task 2 - Image Data Collection and Processing**: Sougnabe Payang
- **Task 3 - Sound Data Collection and Processing**: Lorita Sesame Icyeza
- **Task 4 - Model Creation**: Jacques Twizeyimana
- **Task 5 - System Demonstration**: Group Effort

**Detailed Contributions:**

- **Wagner Mushayija**: Data merging, customer social profiles and transactions integration, exploratory data analysis
- **Sougnabe Payang**: Image collection, facial expression processing, image augmentation, feature extraction
- **Lorita Sesame Icyeza**: Audio recording, sound processing, audio visualization, audio feature extraction
- **Jacques Twizeyimana**: Model training, facial recognition model, voice verification model, product recommendation model

### Project Structure

```text
Formative_2_Data_Processing/
├── Assets/
│   ├── Audios/          # Audio samples for each team member
│   └── Images/          # Facial images (neutral, smiling, surprised)
├── Data/
│   ├── audio_features.csv           # Extracted audio features
│   ├── image_features.csv           # Extracted image features
│   ├── merged_customer_data.csv     # Merged customer dataset
│   ├── customer_social_profiles.csv # Social media profiles
│   ├── customer_transactions.csv    # Transaction history
│   └── processed/                   # Intermediate processed files
├── models/              # Trained model files (.joblib)
├── reports/             # Evaluation metrics and logs
├── scripts/             # Python implementation scripts
│   ├── 01_prepare_sources.py
│   ├── 02_merge_datasets.py
│   ├── 03_validate_merge.py
│   ├── 04_prepare_product_model_data.py
│   ├── 05_train_product_model.py
│   ├── audio_processing.py
│   ├── image_processing.py
│   ├── model_trainer.py
│   ├── predictor.py
│   └── demo.py
├── data_merge.ipynb     # Jupyter notebook for data merge
└── requirements.txt     # Dependencies
```

## 1. Data Merge and Feature Engineering

### 1.1 Dataset Overview

**Customer Social Profiles Dataset:**

- Contains customer engagement metrics across social media platforms
- Features: customer_id_new, social_media_platform, engagement_score, purchase_interest_score, review_sentiment

**Customer Transactions Dataset:**

- Contains historical purchase data
- Features: customer_id_legacy, transaction_id, purchase_amount, purchase_date, product_category, customer_rating

### 1.2 Merge Strategy

The datasets were merged using an inner join on customer ID, bridging the format difference between "customer_id_new" (current format) and "customer_id_legacy" (old format).

### 1.3 Exploratory Data Analysis

**Key Insights:**

- Strong correlation between engagement_score and purchase_interest_score
- Product categories show varied distribution (Electronics, Sports, Books, etc.)

## 2. Image Data Collection and Processing

### 2.1 Image Collection

Each team member provided 3 facial expressions:

- **Neutral**: Baseline expression for recognition
- **Smiling**: Positive emotion state
- **Surprised**: High emotion state

**Image Specifications:**

- Format: JPG/PNG
- Naming convention: `{member}_{expression}.jpg`

### 2.2 Feature Extraction

Extracted features including color histograms, texture features, and deep learning embeddings (using a pre-trained model such as ResNet), which serve as inputs for the facial recognition model. Features are stored in `Data/image_features.csv`.

## 3. Audio Data Collection and Processing

### 3.1 Audio Collection

Each team member recorded 2 phrases:

- "Yes, approve" (approval phrase)
- "Confirm transaction" (confirmation phrase)

### 3.2 Audio Visualization & Feature Extraction

- Time-domain and Frequency-domain analysis
- Extracted features such as MFCCs (Mel-frequency cepstral coefficients) and spectral features to robustly identify speakers. Features are saved to `Data/audio_features.csv`.

## 4. Model Implementation

### 4.1 Facial Recognition Model

**Architecture:**

- Machine Learning Classifier (e.g. Random Forest or similar ensemble)
- Input: Combined image features (histograms, embeddings)

### 4.2 Voice Verification Model

**Architecture:**

- Machine Learning Classifier
- Input: Audio features like MFCCs and spectral features (mean, var)

### 4.3 Product Recommendation Model

**Architecture:**

- Classifier trained on merged demographic/transactional data
- Considers customer behavior patterns and social media engagement

## 5. System Demonstration

### 5.1 Command Line Interface

The system provides an interactive CLI for testing through `demo.py`:

```bash
# Run interactive demo
python scripts/demo.py
```

### 5.2 Security Features

- **Threshold-based Recognition**: e.g., >80% confidence threshold for face recognition / verified in `demo.py`
- **Multi-factor Authentication**: Requires both face and voice match

## 6. Technical Implementation

### 6.1 Dependencies

Install the necessary libraries via:

```bash
pip install -r requirements.txt
```

### 6.2 Data Preparation Pipeline

Run the scripts linearly for full data preparation:

```bash
python scripts/01_prepare_sources.py
python scripts/02_merge_datasets.py
python scripts/03_validate_merge.py
python scripts/04_prepare_product_model_data.py
python scripts/05_train_product_model.py
```

## 7. Conclusion

This project successfully demonstrates a multimodal authentication and recommendation system implemented collaboratively by the team.
