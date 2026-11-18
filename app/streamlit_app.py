
"""Customer Churn Analytics and Prediction Dashboard.

Run using:
    streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from churn.config import OPTIONS
from churn.predict import load_model, predict


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Customer Churn Analytics & Prediction")
st.caption(
    "Analyze customer behavior, understand churn patterns "
    "and predict customer churn risk using Machine Learning."
)

# --------------------------------------------------
# 2. FILE PATHS AND DATA LOADING
# --------------------------------------------------

DATA_FILE = ROOT / "data" / "telco_churn.csv"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_FILE = REPORTS_DIR / "metrics.json"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@st.cache_data
def load_dataset():
    if not DATA_FILE.exists():
        return None
    return pd.read_csv(DATA_FILE)


@st.cache_data
def load_metrics():
    if not METRICS_FILE.exists():
        return None
    return json.loads(METRICS_FILE.read_text(encoding="utf-8"))


@st.cache_resource
def get_bundle():
    return load_model()


df = load_dataset()
metrics = load_metrics()

# --------------------------------------------------
# 3. LOAD TRAINED MODEL
# --------------------------------------------------

try:
    bundle = get_bundle()
except FileNotFoundError as e:
    bundle = None
    st.error(str(e))


# --------------------------------------------------
# 4. NAVIGATION
# --------------------------------------------------

overview_tab, eda_tab, model_tab, prediction_tab = st.tabs(
    [
        "📊 Overview",
        "🔍 Exploratory Data Analysis",
        "🤖 Model Performance",
        "🎯 Customer Prediction"
    ]
)


# ==================================================
# PAGE 1: OVERVIEW DASHBOARD
# ==================================================

with overview_tab:

    st.header("Business Overview")

    if df is None:
        st.error("Dataset not found. Check data/telco_churn.csv.")

    else:
        total_customers = len(df)

        churned = int((df["Churn"] == "Yes").sum())
        retained = int((df["Churn"] == "No").sum())

        churn_rate = churned / total_customers * 100
        retention_rate = retained / total_customers * 100

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Customers", f"{total_customers:,}")
        c2.metric("Churned Customers", f"{churned:,}")
        c3.metric("Retained Customers", f"{retained:,}")
        c4.metric("Churn Rate", f"{churn_rate:.2f}%")

        st.divider()

        st.subheader("Customer Churn Distribution")

        distribution = pd.DataFrame({
            "Customer Status": ["Churned", "Retained"],
            "Customers": [churned, retained]
        }).set_index("Customer Status")

        col1, col2 = st.columns(2)

        with col1:
            st.bar_chart(distribution)

        with col2:
            st.write("### Customer Statistics")

            st.write(f"**Churn rate:** {churn_rate:.2f}%")
            st.write(f"**Retention rate:** {retention_rate:.2f}%")

            st.progress(churn_rate / 100)

            st.info(
                f"Approximately {churn_rate:.1f}% of customers "
                "in this dataset have discontinued the service."
            )

        st.divider()

        st.subheader("Dataset Preview")

        st.write("The dataset contains customer information, "
                 "subscription details, service usage and churn status.")

        st.dataframe(df.head(10), use_container_width=True)

        st.caption(
            f"Dataset contains {total_customers:,} records "
            f"and {df.shape[1]} columns."
        )


# ==================================================
# PAGE 2: EXPLORATORY DATA ANALYSIS
# ==================================================

with eda_tab:

    st.header("Exploratory Data Analysis")

    st.write(
        "Explore customer characteristics and identify "
        "patterns associated with customer churn."
    )

    if df is None:
        st.error("Dataset not found.")

    else:
        st.subheader("Churn by Contract Type")

        contract_churn = (
            df.groupby("Contract")["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .sort_values(ascending=False)
        )

        st.bar_chart(contract_churn)

        st.caption(
            "Month-to-month customers show a higher observed "
            "churn rate than customers with longer contracts."
        )

        st.divider()

        st.subheader("Churn by Internet Service")

        internet_churn = (
            df.groupby("InternetService")["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .sort_values(ascending=False)
        )

        st.bar_chart(internet_churn)

        st.divider()

        st.subheader("Churn by Customer Tenure")

        tenure_data = df.copy()

        tenure_data["Tenure Group"] = pd.cut(
            tenure_data["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=["0-12", "13-24", "25-48", "49-72"]
        )

        tenure_churn = (
            tenure_data.groupby("Tenure Group", observed=False)["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
        )

        st.bar_chart(tenure_churn)

        st.divider()

    st.subheader("Generated EDA Visualizations")

    if FIGURES_DIR.exists():

        image_files = sorted(
            [
                p for p in FIGURES_DIR.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )

        eda_images = [
            p for p in image_files
            if p.name.lower().startswith("eda")
        ]

        if eda_images:

            for image_path in eda_images:
                st.markdown(f"**{image_path.stem.replace('_', ' ').title()}**")
                st.image(str(image_path), use_container_width=True)

        else:
            st.info("No generated EDA image files were found.")

    else:
        st.info("The reports/figures folder was not found.")


# ==================================================
# PAGE 3: MODEL PERFORMANCE
# ==================================================

with model_tab:

    st.header("Machine Learning Model Performance")

    st.write(
        "Three classification algorithms were evaluated "
        "to identify the best customer churn prediction model."
    )

    if metrics is not None:

        st.subheader("Selected Model")

        st.success(f"Best Model: {metrics['best_model']}")

        tuned = metrics.get("test_metrics_tuned_threshold", {})

        if tuned:

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "ROC-AUC",
                f"{tuned.get('roc_auc', 0):.3f}"
            )

            c2.metric(
                "Recall",
                f"{tuned.get('recall', 0):.3f}"
            )

            c3.metric(
                "Precision",
                f"{tuned.get('precision', 0):.3f}"
            )

            c4.metric(
                "F1 Score",
                f"{tuned.get('f1', 0):.3f}"
            )

            if "threshold" in tuned:
                st.caption(
                    f"Selected decision threshold: "
                    f"{tuned['threshold']}"
                )

        else:
            st.info(
                "Detailed tuned-threshold metrics are unavailable."
            )

    else:
        st.warning("Model metrics file was not found.")

    st.divider()

    st.subheader("Model Evaluation Visualizations")

    if FIGURES_DIR.exists():

        image_files = sorted(
            [
                p for p in FIGURES_DIR.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )

        performance_images = [
            p for p in image_files
            if not p.name.lower().startswith("eda")
        ]

        if performance_images:

            for image_path in performance_images:

                st.markdown(
                    f"**{image_path.stem.replace('_', ' ').title()}**"
                )

                st.image(
                    str(image_path),
                    use_container_width=True
                )

        else:
            st.info("No model evaluation images were found.")

    else:
        st.info("The reports/figures folder was not found.")

    st.divider()

    st.subheader("Model Comparison")

    comparison_files = [
        REPORTS_DIR / "model_comparison.csv",
        REPORTS_DIR / "model_comparison.json"
    ]

    comparison_found = False

    for comparison_file in comparison_files:

        if comparison_file.exists():

            try:
                if comparison_file.suffix == ".csv":
                    comparison_data = pd.read_csv(comparison_file)
                else:
                    comparison_data = pd.read_json(comparison_file)

                st.dataframe(
                    comparison_data,
                    use_container_width=True
                )

                comparison_found = True
                break

            except (ValueError, OSError) as e:
                st.warning(f"Could not read comparison file: {e}")

    if not comparison_found:
        st.info(
            "A separate model comparison table was not found. "
            "See the evaluation figures and RESULTS.md report."
        )


# ==================================================
# PAGE 4: CUSTOMER CHURN PREDICTION
# ==================================================

with prediction_tab:

    st.header("Customer Churn Prediction")

    st.write(
        "Enter customer information to estimate the probability "
        "that the customer will discontinue the service."
    )

    if bundle is None:
        st.error("The trained model could not be loaded.")

    else:

        c1, c2, c3 = st.columns(3)

        with c1:

            st.subheader("Account Information")

            tenure = st.slider(
                "Tenure (months)", 0, 72, 12
            )

            monthly = st.number_input(
                "Monthly charges",
                0.0, 200.0, 70.0,
                step=0.5
            )

            contract = st.selectbox(
                "Contract", OPTIONS["Contract"]
            )

            paperless = st.selectbox(
                "Paperless billing",
                OPTIONS["PaperlessBilling"]
            )

            payment = st.selectbox(
                "Payment method",
                OPTIONS["PaymentMethod"]
            )

        with c2:

            st.subheader("Customer Information")

            gender = st.selectbox(
                "Gender", OPTIONS["gender"]
            )

            senior = st.selectbox(
                "Senior citizen", OPTIONS["SeniorCitizen"]
            )

            partner = st.selectbox(
                "Partner", OPTIONS["Partner"]
            )

            dependents = st.selectbox(
                "Dependents", OPTIONS["Dependents"]
            )

            phone = st.selectbox(
                "Phone service", OPTIONS["PhoneService"]
            )

            multiple = st.selectbox(
                "Multiple lines", OPTIONS["MultipleLines"]
            )

        with c3:

            st.subheader("Internet Services")

            internet = st.selectbox(
                "Internet service",
                OPTIONS["InternetService"]
            )

            addons = {
                key: st.selectbox(label, OPTIONS[key])
                for key, label in [
                    ("OnlineSecurity", "Online security"),
                    ("OnlineBackup", "Online backup"),
                    ("DeviceProtection", "Device protection"),
                    ("TechSupport", "Tech support"),
                    ("StreamingTV", "Streaming TV"),
                    ("StreamingMovies", "Streaming movies")
                ]
            }

        st.divider()

        if st.button(
            "Predict Churn Risk",
            type="primary",
            use_container_width=True
        ):

            record = {
                "tenure": tenure,
                "MonthlyCharges": monthly,
                "TotalCharges": tenure * monthly,
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "PhoneService": phone,
                "MultipleLines": multiple,
                "InternetService": internet,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                **addons
            }

            try:

                result = predict(record, bundle).iloc[0]

                probability = float(
                    result["churn_probability"]
                )

                risk_band = result["risk_band"]

                prediction = result["prediction"]

                st.subheader("Prediction Results")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Churn Probability",
                    f"{probability:.1%}"
                )

                col2.metric(
                    "Risk Category",
                    risk_band
                )

                st.progress(
                    min(max(probability, 0.0), 1.0)
                )

                if prediction == 1:

                    st.warning(
                        "This customer is flagged as likely to churn. "
                        "Consider a suitable customer retention offer."
                    )

                else:

                    st.success(
                        "This customer is not flagged as likely to "
                        "churn at the model's decision threshold."
                    )

                st.caption(
                    "Predictions are estimates produced by the "
                    "trained machine learning model."
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Customer Churn Prediction Project | "
    "Python • Pandas • Scikit-learn • Streamlit"
)
