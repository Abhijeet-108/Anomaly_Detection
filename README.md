# KAVACAPay — Credit Card Anomaly & Fraud Detection

An end-to-end machine learning project that detects credit card fraud and transaction anomalies using a two-stage pipeline (Isolation Forest + Random Forest), wrapped in a Streamlit web application with user authentication and subscription-based access.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Authentication & User Management](#authentication--user-management)
- [Subscription & Payments](#subscription--payments)
- [Notebooks](#notebooks)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output & Export](#output--export)
- [Risk Assessment Logic](#risk-assessment-logic)
- [Gitignored Files](#gitignored-files)
- [Limitations & Notes](#limitations--notes)

---

## Overview

**KAVACAPay** (also branded as **Kavaca Pay**) is an enterprise-style fraud screening dashboard. Users upload CSV files containing credit card transaction data, and the system:

1. Detects statistical anomalies using **Isolation Forest** (trained on normal transactions only).
2. Scores fraud probability using a tuned **Random Forest** classifier.
3. Presents interactive metrics, charts, risk tiers, and downloadable reports.

The project combines exploratory data analysis notebooks, a production-style training script, and a polished Streamlit frontend with login/signup and freemium subscription tiers.

---

## Features

### Fraud Detection
- Two-stage detection pipeline: anomaly detection + fraud classification
- Adjustable fraud probability threshold via sidebar slider
- Risk tiers: **Confirmed Fraud**, **Suspicious**, **Low Risk**, **Normal**
- Overall risk level: **CRITICAL**, **HIGH**, **MEDIUM**, **LOW**
- Live metrics: transaction count, anomalies, confirmed fraud, clean transactions, average fraud score
- Visual analytics: anomaly split donut chart, risk breakdown bar chart, fraud score histogram
- Tabbed results: All Results, Anomalies, Confirmed Fraud, Raw Preview
- Top 5 highest-risk transactions table
- Full analysis CSV export

### User Experience
- Dark-themed, modern UI with custom CSS (DM Sans font, gradient hero, glassmorphism cards)
- Responsive wide layout with expandable sidebar
- File upload with plan-based size limits
- Session persistence via `last_login` flag in user store

### Business Model
- **Free plan**: 3 prediction scans, max 100 MB CSV upload
- **Premium plan**: Unlimited scans, max 200 MB CSV upload, 30-day subscription
- Razorpay payment link integration for upgrades (₹99.00 / 9900 paise)

---

## Project Structure

```
Anomaly_Detection/
├── app/
│   ├── app.py                 # Main Streamlit dashboard
│   ├── auth.py                # Login, signup, session, password hashing
│   ├── users.json             # Local user database (JSON)
│   └── .streamlit/
│       └── config.toml        # Streamlit server config (max upload 200 MB)
├── notebook/
│   ├── train_models.py        # Script to train and save ML models
│   ├── anomaly_detection.ipynb           # EDA + model experimentation
│   ├── credit-card-fraud-detection-predictive-models.ipynb  # Reference EDA & benchmark models
│   ├── isolation_forest_model.pkl        # (generated after training)
│   ├── random_forest_fraud_model.pkl       # (generated after training)
│   ├── scaler.pkl                          # (generated after training)
│   └── model_metadata.pkl                  # (generated after training)
├── data/
│   └── creditcard_2023.csv    # Training dataset (not in repo — see below)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3 |
| Web framework | Streamlit ≥ 1.32.0 |
| Data processing | pandas ≥ 2.0.0, numpy ≥ 1.24.0 |
| Machine learning | scikit-learn == 1.9.0 |
| Model persistence | joblib ≥ 1.3.0 |
| Visualization | matplotlib ≥ 3.7.0 |
| Payments | razorpay == 2.0.1 |
| Notebooks | Jupyter (anomaly_detection.ipynb, reference notebook) |

---

## Dataset

### Source
Credit card fraud detection data based on the well-known [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) dataset. The project uses extended versions:

- `creditcard_2023.csv` — primary training dataset (`train_models.py`, `anomaly_detection.ipynb`)
- `creditcard_2022.csv` — used in the reference notebook
- `creditcard.csv` — original dataset variant

### Characteristics
- Highly imbalanced: ~0.17% fraudulent transactions
- Features **V1–V28** are PCA-transformed principal components (original features not disclosed)
- **Amount** — transaction amount (not PCA-transformed)
- **Class** — target label (`0` = legitimate, `1` = fraud)
- **id** — transaction identifier (dropped during training to prevent leakage)

### Required Columns for Prediction
Upload CSV must include: **V1, V2, …, V28, Amount**

Optional columns `Class` and `id` are automatically ignored during inference.

---

## Machine Learning Pipeline

### Stage 1 — Isolation Forest (Anomaly Detection)
- Trained **only on normal (non-fraud) transactions** from the training set
- Contamination rate derived from training fraud rate, clamped between `0.001` and `0.05`
- Hyperparameters: `n_estimators=200`, `max_samples="auto"`, `random_state=42`
- Output: `Anomaly` flag (1 = outlier, 0 = normal) and `anomaly_score` (decision function)

### Stage 2 — Random Forest (Fraud Classification)
- Trained on full labeled training data with `class_weight="balanced_subsample"`
- Hyperparameters: `n_estimators=200`, `max_depth=12`, `min_samples_split=4`, `min_samples_leaf=2`
- Output: `fraud_probability` (0–1) and `fraud` binary flag based on threshold

### Preprocessing
- **StandardScaler** fit on training features, applied to all inference data
- 80/20 stratified train/test split (`random_state=42`)
- Columns dropped before training: `Class`, `id`

### Optimal Threshold
- Fraud threshold selected via precision-recall curve to maximize F1 score
- Stored in `model_metadata.pkl` and used as the default sidebar slider value

### Risk Tier Assignment
| Condition | Tier |
|-----------|------|
| `fraud == 1` | Confirmed Fraud |
| `Anomaly == 1` and `fraud == 0` | Suspicious |
| Anomaly score in bottom 5th percentile | Low Risk |
| Otherwise | Normal |

---

## Model Training

### Prerequisites
Place the dataset at:
```
data/creditcard_2023.csv
```

### Run Training
```bash
python notebook/train_models.py
```

### Generated Artifacts
Saved to `notebook/`:

| File | Description |
|------|-------------|
| `isolation_forest_model.pkl` | Isolation Forest model |
| `random_forest_fraud_model.pkl` | Random Forest classifier |
| `scaler.pkl` | Fitted StandardScaler |
| `model_metadata.pkl` | Features list, fraud threshold, contamination rate, metrics |

### Metrics Stored
- `isolation_forest_f1` — F1 score of Isolation Forest on test set
- `random_forest_auc` — ROC-AUC of Random Forest
- `random_forest_f1` — F1 at optimal threshold

---

## Web Application

### Entry Point
```bash
streamlit run app/app.py
```

### Pages & Flow
1. **Auth page** — shown when not authenticated (Login / Sign Up tabs)
2. **Dashboard** — main fraud detection interface after login

### Sidebar Sections
- User info and logout
- Subscription plan status (free scans remaining or premium badge)
- Upgrade to Premium / Verify Payment buttons (Razorpay)
- Fraud probability threshold slider (0.05–0.95)
- Model performance metrics (RF ROC-AUC, RF F1, IF F1)
- Pipeline steps and required column reference

### Main Dashboard Sections
- Hero banner with project description
- Pipeline cards (Isolation Forest, Random Forest, Smart Dashboard)
- CSV file uploader
- Live results metrics (6 KPI cards)
- Charts (anomaly split, risk breakdown, fraud score distribution)
- Risk assessment alerts (danger / warning / success)
- Progress bars for anomaly and fraud rates
- Tabbed data tables
- CSV download button

### Detection Logic (`run_detection`)
1. Scale input features with saved scaler
2. Run Isolation Forest → anomaly flags and scores
3. Run Random Forest → fraud probabilities
4. Apply user-selected threshold → fraud flags
5. Assign risk tiers

### Display Limits
- Results tables show up to **500 rows** (full data available via CSV download)
- Raw preview shows first **15 rows**

---

## Authentication & User Management

Implemented in `app/auth.py` with local JSON storage (`app/users.json`).

### Sign Up
- Fields: Full Name, Email, Password, Confirm Password, Age
- Minimum age: **15 years**
- Minimum password length: **6 characters**
- Email format validation via regex
- Passwords hashed with **PBKDF2-HMAC-SHA256** (100,000 iterations, per-user salt)

### Login
- Email + password verification
- Sets `last_login: true` for the user (only one active remembered session pattern)
- Populates Streamlit session state

### Session State Keys
- `authenticated`, `user_email`, `user_name`, `user_age`
- `user_plan`, `subscription_expiry`, `prediction_count`
- `payment_link_id`, `payment_status`

### Logout
- Clears session state and sets `last_login: false` for the user

### Auto-Login
On app load, if any user has `last_login: true` in `users.json`, they are automatically authenticated.

---

## Subscription & Payments

### Free Plan (default)
- `max_free_predictions`: 3 scans
- Max upload size: **100 MB**
- Counter increments on each CSV analysis

### Premium Plan
- Unlimited predictions
- Max upload size: **200 MB**
- Price: **₹99.00** (9900 paise) via Razorpay payment link
- Subscription expiry: 30 days from activation
- Activated when payment status is verified as `paid`

### Payment Flow
1. User clicks **Upgrade to Premium** → Razorpay payment link created
2. `payment_link_id` and `payment_status: pending` saved to user record
3. User pays via **Pay Now** link button
4. User clicks **Verify Payment** → fetches link status from Razorpay API
5. On `paid` status → plan set to `premium`, expiry date saved

> **Note:** Razorpay client in `app.py` is initialized with empty credentials (`auth=()`). You must configure your Razorpay API key and secret before payments work in production.

---

## Notebooks

### `notebook/anomaly_detection.ipynb`
Primary project notebook covering:
- Data loading from `creditcard_2023.csv`
- Exploratory data analysis (EDA)
- Feature/target separation
- Train/test split and StandardScaler
- **Logistic Regression** baseline
- **Random Forest** classifier (confusion matrix, ROC curve)
- **Isolation Forest** anomaly detection
- Model evaluation and visualization
- Model export via joblib (`.pkl` files)

### `notebook/credit-card-fraud-detection-predictive-models.ipynb`
Reference notebook (Kaggle-style) covering:
- Dataset introduction and class imbalance analysis
- Time and Amount distribution analysis
- Feature distribution comparisons (fraud vs non-fraud)
- Correlation heatmaps
- Hypothesis testing framework (Type I / Type II errors)
- Benchmark classifiers:
  - Random Forest (~0.85 ROC-AUC)
  - AdaBoost (~0.83 ROC-AUC)
  - CatBoost (~0.86 ROC-AUC)
  - XGBoost (~0.974 AUC)
  - LightGBM (~0.974 AUC validation, ~0.946 test)

> The production app uses only Isolation Forest + Random Forest from `train_models.py`, not the gradient boosting models from the reference notebook.

---

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd Anomaly_Detection
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Download or place `creditcard_2023.csv` in the `data/` folder.

### 5. Train models
```bash
python notebook/train_models.py
```

### 6. (Optional) Configure Razorpay
Set your Razorpay API credentials in `app/app.py` before enabling payments.

### 7. Run the app
```bash
streamlit run app/app.py
```

---

## Usage

### First-Time Setup
1. Open the app in your browser (default: `http://localhost:8501`)
2. Create an account on the **Sign Up** tab
3. Log in with your credentials

### Analyzing Transactions
1. Upload a CSV with columns `V1`–`V28` and `Amount`
2. Wait for the analysis spinner to complete
3. Review metrics, charts, and risk alerts
4. Browse results in tabs (All / Anomalies / Fraud / Preview)
5. Adjust the fraud threshold in the sidebar if needed
6. Download the full report as `fraud_detection_report.csv`

### Threshold Tips
- **Lower threshold** → more transactions flagged as fraud (higher recall)
- **Higher threshold** → fewer false alarms (higher precision)

---

## Configuration

### Streamlit (`app/.streamlit/config.toml`)
```toml
[server]
maxUploadSize = 200
```

### Model Directory
Models are loaded from `notebook/` relative to `app/app.py`:
- `isolation_forest_model.pkl`
- `random_forest_fraud_model.pkl`
- `scaler.pkl`
- `model_metadata.pkl`

If model files are missing, the app displays an error prompting you to run `python notebook/train_models.py`.

---

## Output & Export

### Result Columns
| Column | Description |
|--------|-------------|
| V1–V28, Amount | Original transaction features |
| `anomaly_score` | Isolation Forest decision function (hidden in UI table) |
| `Anomaly` | 1 if flagged as outlier, 0 otherwise |
| `fraud_probability` | Random Forest fraud probability (0–1) |
| `fraud` | 1 if probability ≥ threshold, 0 otherwise |
| `risk_tier` | Confirmed Fraud / Suspicious / Low Risk / Normal |

### Export
- Button: **Download Full Analysis (CSV)**
- Filename: `fraud_detection_report.csv`
- Excludes internal `anomaly_score` column

---

## Risk Assessment Logic

### Portfolio Risk Level (`risk_label`)
| Level | Condition |
|-------|-----------|
| CRITICAL | Fraud rate > 1% OR anomaly rate > 8% |
| HIGH | Fraud rate > 0.2% OR anomaly rate > 3% |
| MEDIUM | Fraud rate > 0.05% OR anomaly rate > 1% |
| LOW | Otherwise |

### Alert Cards
- **Danger** — confirmed fraud transactions detected
- **Warning** — anomalies detected but none confirmed at current threshold
- **Success** — no anomalies detected

---

## Gitignored Files

Per `.gitignore`, the following are excluded from version control:

| Path | Reason |
|------|--------|
| `.venv` | Virtual environment |
| `data/creditcard.csv` | Large/sensitive dataset |
| `data/*.csv` | All CSV datasets |
| `env`, `.env` | Environment secrets |

Model `.pkl` files are not gitignored but must be generated locally via training.

---

## Limitations & Notes

1. **Dataset not included** — You must obtain `creditcard_2023.csv` separately and place it in `data/`.
2. **Models not pre-trained** — Run `train_models.py` before launching the app.
3. **Local user storage** — `users.json` is a flat file; not suitable for multi-user production without a real database.
4. **Razorpay credentials** — Payment integration requires valid API keys; currently initialized with empty auth.
5. **Password security** — PBKDF2 hashing is used, but `users.json` should not be committed with real credentials in production.
6. **Single remembered login** — Only one user can have `last_login: true` at a time.
7. **No API layer** — The app is a standalone Streamlit UI, not a REST API.
8. **sklearn version pinned** — `scikit-learn==1.9.0` is required for model compatibility.

---

## Quick Reference Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python notebook/train_models.py

# Start the dashboard
streamlit run app/app.py
```

---

## License

No license file is included in this repository. Add one if you plan to distribute or open-source the project.
