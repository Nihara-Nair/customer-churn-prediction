"""Streamlit app for real-time churn risk scoring.   streamlit run app/streamlit_app.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from churn.config import OPTIONS  # noqa: E402
from churn.predict import load_model, predict  # noqa: E402

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")
st.title("📉 Customer Churn Predictor")
st.caption("Enter a customer's details to estimate their probability of cancelling the service.")


@st.cache_resource
def get_bundle():
    return load_model()


try:
    bundle = get_bundle()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Account")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0, step=0.5)
    contract = st.selectbox("Contract", OPTIONS["Contract"])
    paperless = st.selectbox("Paperless billing", OPTIONS["PaperlessBilling"])
    payment = st.selectbox("Payment method", OPTIONS["PaymentMethod"])
with c2:
    st.subheader("Customer")
    gender = st.selectbox("Gender", OPTIONS["gender"])
    senior = st.selectbox("Senior citizen", OPTIONS["SeniorCitizen"])
    partner = st.selectbox("Partner", OPTIONS["Partner"])
    dependents = st.selectbox("Dependents", OPTIONS["Dependents"])
    phone = st.selectbox("Phone service", OPTIONS["PhoneService"])
    multiple = st.selectbox("Multiple lines", OPTIONS["MultipleLines"])
with c3:
    st.subheader("Internet services")
    internet = st.selectbox("Internet service", OPTIONS["InternetService"])
    addons = {k: st.selectbox(label, OPTIONS[k]) for k, label in [
        ("OnlineSecurity", "Online security"), ("OnlineBackup", "Online backup"),
        ("DeviceProtection", "Device protection"), ("TechSupport", "Tech support"),
        ("StreamingTV", "Streaming TV"), ("StreamingMovies", "Streaming movies")]}

if st.button("Predict churn risk", type="primary"):
    record = {"tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": tenure * monthly, "gender": gender,
              "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents, "PhoneService": phone,
              "MultipleLines": multiple, "InternetService": internet, "Contract": contract,
              "PaperlessBilling": paperless, "PaymentMethod": payment, **addons}
    result = predict(record, bundle).iloc[0]
    p = float(result["churn_probability"])
    a, b = st.columns(2)
    a.metric("Churn probability", f"{p:.1%}")
    b.metric("Risk band", result["risk_band"])
    st.progress(min(max(p, 0.0), 1.0))
    if result["prediction"] == 1:
        st.warning("Flagged as likely to churn. Consider a retention offer (e.g. a longer-term contract discount).")
    else:
        st.success("Not flagged as likely to churn at the model's decision threshold.")

metrics_file = ROOT / "reports" / "metrics.json"
if metrics_file.exists():
    with st.expander("About the model"):
        m = json.loads(metrics_file.read_text())
        t = m["test_metrics_tuned_threshold"]
        st.write(f"**{m['best_model']}**: test ROC-AUC {t['roc_auc']:.3f}, recall {t['recall']:.3f}, "
                 f"precision {t['precision']:.3f} (decision threshold {t['threshold']}).")
