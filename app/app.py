import streamlit as st
import pandas as pd
import joblib

model = joblib.load('../notebook/isolation_forest_model.pkl')
scaler = joblib.load('../notebook/scaler.pkl')

st.title("Anomaly Detection System")

uploaded_file = st.file_uploader("Upload CSV")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Drop target column if exists
    if 'Class' in data.columns:
        data = data.drop('Class', axis=1)

    # Scale data
    data_scaled = scaler.transform(data)

    # Predict anomalies
    preds = model.predict(data_scaled)

    # Count anomalies
    anomaly_count = (preds == -1).sum()

    st.success(f"Detected anomalies: {anomaly_count}")