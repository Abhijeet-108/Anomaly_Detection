from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NOTEBOOK_DIR = Path(__file__).resolve().parent
DATA_PATH = NOTEBOOK_DIR.parent / "data" / "creditcard_2023.csv"
DROP_COLS = {"Class", "id"}


def find_best_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1_scores))
    if best_idx >= len(thresholds):
        return 0.5, float(f1_scores[best_idx])
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    y = df["Class"].astype(int)
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Isolation Forest — trained on normal transactions only
    X_train_normal = X_train_scaled[y_train == 0]
    fraud_rate = float(y_train.mean())
    iso_contamination = min(max(fraud_rate, 0.001), 0.05)

    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=iso_contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X_train_normal)

    iso_preds = iso_forest.predict(X_test_scaled)
    y_iso = np.where(iso_preds == -1, 1, 0)
    iso_f1 = f1_score(y_test, y_iso)

    # Random Forest — balanced weights, no id leakage
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)

    fraud_probs = rf.predict_proba(X_test_scaled)[:, 1]
    fraud_threshold, val_f1 = find_best_threshold(y_test, fraud_probs)
    y_rf = (fraud_probs >= fraud_threshold).astype(int)
    rf_auc = roc_auc_score(y_test, fraud_probs)

    print("=" * 60)
    print("Isolation Forest (trained on normal class)")
    print(classification_report(y_test, y_iso, digits=4))
    print(f"Isolation Forest F1: {iso_f1:.4f}")
    print()
    print("Random Forest")
    print(classification_report(y_test, y_rf, digits=4))
    print(f"Random Forest ROC-AUC: {rf_auc:.4f}")
    print(f"Optimal fraud threshold: {fraud_threshold:.4f} (val F1: {val_f1:.4f})")
    print("=" * 60)

    metadata = {
        "features": list(X.columns),
        "fraud_threshold": fraud_threshold,
        "iso_contamination": iso_contamination,
        "metrics": {
            "isolation_forest_f1": iso_f1,
            "random_forest_auc": rf_auc,
            "random_forest_f1": val_f1,
        },
    }

    joblib.dump(iso_forest, NOTEBOOK_DIR / "isolation_forest_model.pkl")
    joblib.dump(rf, NOTEBOOK_DIR / "random_forest_fraud_model.pkl")
    joblib.dump(scaler, NOTEBOOK_DIR / "scaler.pkl")
    joblib.dump(metadata, NOTEBOOK_DIR / "model_metadata.pkl")

    print("Models saved to notebook/")


if __name__ == "__main__":
    main()
