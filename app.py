"""
Healthcare Provider Fraud Detection — Streamlit app (self-contained).
Needs only: app.py, requirements.txt, fraud_model.joblib, metrics.json
Run locally:  streamlit run app.py
"""
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Provider Fraud Detection", page_icon="🛡️", layout="wide")

# Median feature values (baked in so the app needs no data file)
DEFAULTS = {"TotalClaims": 31.0, "UniqueBeneficiaries": 25.0, "UniqueAttendingPhys": 6.0,
    "UniqueOperatingPhys": 3.0, "TotalReimbursed": 19805.0, "MeanReimbursed": 356.09,
    "StdReimbursed": 674.25, "MaxReimbursed": 3300.0, "MeanDeductible": 4.29,
    "MeanClaimDuration": 1.59, "MeanLengthOfStay": 0.0, "MeanNumDiagnosis": 2.81,
    "MeanNumProcedure": 0.0, "MeanNumPhysicians": 1.57, "MeanPatientAge": 73.82,
    "DeadPatientRatio": 0.0, "MeanChronicConditions": 4.51, "RenalDiseaseRatio": 0.19,
    "MeanIPAnnualReimb": 4729.05, "MeanOPAnnualReimb": 2044.36,
    "ClaimsPerBeneficiary": 1.1, "ClaimsPerAttendingPhys": 3.0}

@st.cache_resource
def load_artifacts():
    bundle = joblib.load("fraud_model.joblib")
    with open("metrics.json") as f:
        metrics = json.load(f)
    return bundle, metrics

bundle, metrics = load_artifacts()
model, scaler, features = bundle["model"], bundle["scaler"], bundle["features"]
threshold = bundle["threshold"]

st.title("🛡️ Healthcare Provider Fraud Detection")
st.caption("Predicts whether a healthcare provider is potentially fraudulent from their aggregated claim behaviour.")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📂 Batch scoring", "📊 Model performance"])

# ---------- Tab 1: single provider ----------
with tab1:
    st.subheader("Score a single provider")
    st.write("Enter a provider's aggregated claim statistics. Defaults are dataset medians.")
    cols = st.columns(3)
    vals = {}
    for i, f in enumerate(features):
        with cols[i % 3]:
            vals[f] = st.number_input(f, value=float(DEFAULTS.get(f, 0.0)))

    if st.button("Predict", type="primary"):
        x = scaler.transform(pd.DataFrame([vals])[features].values)
        proba = float(model.predict_proba(x)[0, 1])
        label = "FRAUD" if proba >= threshold else "NOT FRAUD"
        c1, c2 = st.columns(2)
        c1.metric("Fraud probability", f"{proba:.1%}")
        c2.metric("Prediction", label)
        st.progress(min(proba, 1.0))
        if proba >= threshold:
            st.error(f"⚠️ Flagged as potentially fraudulent (threshold {threshold:.0%}). Recommend manual audit.")
        else:
            st.success(f"✅ Below the fraud threshold ({threshold:.0%}).")

# ---------- Tab 2: batch ----------
with tab2:
    st.subheader("Batch scoring")
    st.write("Upload a provider-level CSV (same feature columns as the training data) to score many providers at once.")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        df = pd.read_csv(up)
        missing = [f for f in features if f not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            x = scaler.transform(df[features].values)
            df["FraudProbability"] = model.predict_proba(x)[:, 1].round(4)
            df["PotentialFraud"] = np.where(df["FraudProbability"] >= threshold, "Yes", "No")
            out = df[["Provider", "FraudProbability", "PotentialFraud"]] if "Provider" in df else df
            st.write(f"Flagged **{(df.PotentialFraud=='Yes').sum()}** of {len(df)} providers.")
            st.dataframe(out, use_container_width=True)
            st.download_button("Download predictions", out.to_csv(index=False),
                               "predictions.csv", "text/csv")

# ---------- Tab 3: performance ----------
with tab3:
    st.subheader("Model performance (validation)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Best model", metrics["best_model"])
    c2.metric("ROC-AUC", f"{metrics['roc_auc'][metrics['best_model']]:.3f}")
    c3.metric("Fraud F1", f"{metrics['val_f1_fraud']:.3f}")
    st.write("**ROC-AUC by model**")
    st.bar_chart(pd.Series(metrics["roc_auc"]))
    st.write("**Top features driving fraud prediction**")
    st.bar_chart(pd.Series(metrics["top_features"]))
    st.caption("Confidential case-study submission — data not to be shared or published.")
