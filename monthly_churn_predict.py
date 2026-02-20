#!/usr/bin/env python
"""
Monthly Churn Prediction Script
Loads the latest customer data, runs predictions, and saves high-risk customers.
"""

import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime

# ---------- CONFIGURATION ----------
MODEL_PATH = 'churn_model.pkl'
COLUMNS_PATH = 'model_columns.pkl'
INPUT_DATA_PATH = 'data/latest_customers.csv'
OUTPUT_PATH = 'output'
RISK_THRESHOLD = 0.5
# -----------------------------------


def load_model():
    """Load the trained model and the feature columns."""
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    return model, model_columns


def prepare_data(df, model_columns):
    """Apply the same preprocessing as during training."""

    # Preserve customerID for output
    customer_ids = df['customerID'] if 'customerID' in df.columns else None

    # Drop target if present
    if 'Churn' in df.columns:
        df = df.drop('Churn', axis=1)

    # -----------------------------
    # 🔧 FIX 1 — Convert TotalCharges
    # -----------------------------
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # -----------------------------
    # 🔧 FIX 2 — Feature Engineering (MUST match training)
    # -----------------------------
    if 'tenure' in df.columns:
        df['tenure_years'] = df['tenure'] / 12

        df['avg_monthly_charges'] = np.where(
            df['tenure'] > 0,
            df['TotalCharges'] / df['tenure'],
            df['MonthlyCharges']
        )

        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=[0, 12, 24, 48, 72],
            labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr']
        )

    # -----------------------------
    # 🔧 FIX 3 — Drop ID before encoding
    # -----------------------------
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # -----------------------------
    # 🔧 FIX 4 — Safe categorical detection (pandas 3 ready)
    # -----------------------------
    cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # -----------------------------
    # 🔧 FIX 5 — Align columns with training
    # -----------------------------
    for col in model_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Keep only training columns in correct order
    X = df_encoded[model_columns]

    return X, customer_ids


def main():
    print(f"{datetime.now()} - Starting churn prediction batch job")

    # Load model
    model, model_columns = load_model()

    # Check input exists
    if not os.path.exists(INPUT_DATA_PATH):
        print(f"❌ Error: Input file {INPUT_DATA_PATH} not found.")
        return

    # Load new data
    df_new = pd.read_csv(INPUT_DATA_PATH)
    print(f"Loaded {len(df_new)} customers.")

    # Prepare features
    X, customer_ids = prepare_data(df_new, model_columns)

    if X.shape[0] == 0:
        print("❌ No valid customers to score after preprocessing.")
        return

    # Predict probabilities
    probabilities = model.predict_proba(X)[:, 1]

    # Create results dataframe
    if customer_ids is not None:
        results = pd.DataFrame({
            'customerID': customer_ids,
            'churn_probability': probabilities,
            'prediction': (probabilities >= RISK_THRESHOLD).astype(int)
        })
    else:
        results = pd.DataFrame({
            'churn_probability': probabilities,
            'prediction': (probabilities >= RISK_THRESHOLD).astype(int)
        })

    # Sort by highest risk first
    results = results.sort_values('churn_probability', ascending=False)

    # Ensure output folder exists
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d')

    # Save full results
    results.to_csv(f'{OUTPUT_PATH}/churn_predictions_{timestamp}.csv', index=False)

    # Save high-risk customers
    high_risk = results[results['prediction'] == 1]
    high_risk.to_csv(f'{OUTPUT_PATH}/high_risk_customers_{timestamp}.csv', index=False)

    print(
        f"{datetime.now()} - Completed. "
        f"Scored {len(results)} customers, {len(high_risk)} high-risk."
    )
    print(f"Results saved to {OUTPUT_PATH}/")


if __name__ == "__main__":
    main()