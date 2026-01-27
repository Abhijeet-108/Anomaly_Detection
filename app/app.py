import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model & scaler
model = joblib.load("../notebook/isolation_forest_model.pkl")
model1 = joblib.load("../notebook/random_forest_fraud_model.pkl")
scaler = joblib.load("../notebook/scaler.pkl")

st.set_page_config(page_title="Fraud Anomaly Detection", layout="centered")
st.title("Credit Card Anomaly Detection System")

st.write(
    """
    Upload a CSV file containing credit card transaction data.
    The system will detect **anomalous (suspicious) transactions**
    using an **Isolation Forest model**.
    """
)

# File Upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    # Drop target column if exists
    if "Class" in data.columns:
        data = data.drop(columns=["Class"])

    # Ensure numeric data only
    data = data.select_dtypes(include=[np.number])

    # Feature alignment check
    expected_features = scaler.feature_names_in_

    missing_cols = set(expected_features) - set(data.columns)
    extra_cols = set(data.columns) - set(expected_features)

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    if extra_cols:
        data = data[expected_features]

    # Scaling
    data_scaled = scaler.transform(data)

    # Anomaly Prediction
    preds = model.predict(data_scaled)

    # Convert predictions
    #  1 → Normal
    # -1 → Anomaly
    data["Anomaly"] = np.where(preds == -1, 1, 0)
    
    # Fraud Prediction using Random Forest Model
    data["fraud"] = 0
    data["fraud_probability"] = 0.0
    
    suspicious_idx = data[data["Anomaly"] == 1].index
    
    if len(suspicious_idx) > 0:
        fraud_probs = model1.predict_proba(
            data_scaled[suspicious_idx]
        )[:, 1]
        fraud_preds = (fraud_probs >= 0.5).astype(int)
        
        data.loc[suspicious_idx, "fraud"] = fraud_preds
        data.loc[suspicious_idx, "fraud_probability"] = fraud_probs

    # Summary results
    total = len(data)
    anomaly_count = data["Anomaly"].sum()
    anomaly_percent = (anomaly_count / total) * 100
    fraud_count = data["fraud"].sum()

    # Results
    st.success(f"Total Transactions: {total}")
    st.warning(f"Detected Anomalies: {anomaly_count}")
    st.info(f"Anomaly Percentage: {anomaly_percent:.2f}%")
    st.error(f"Confirmed Fraud Transactions: {fraud_count}")

    # Show anomalies
    if anomaly_count > 0:
        st.subheader("Detected Anomalous Transactions")
        st.dataframe(data[data["Anomaly"] == 1].head(50))
    else:
        st.subheader("No anomalies detected")
        
    # Display full results for download
    st.subheader("Full Results")
    st.dataframe(
        data.sort_values(
            by=["fraud", "fraud_probability"],
            ascending=False
        ).head(50)
    )
    
    # Download link
    csv = data.to_csv(index=False).encode()
    st.download_button(
        label="Download Full Results as CSV",
        data=csv,
        file_name="anomaly_detection_results.csv",
        mime="text/csv",
    )

