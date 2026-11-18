"""Column definitions and allowed values for the Telco Customer Churn dataset."""

TARGET = "Churn"
ID_COL = "customerID"

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL

# Used by the Streamlit app to build input widgets.
_YN = ["No", "Yes"]
_ADDON = ["No", "Yes", "No internet service"]
OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": _YN,
    "Partner": _YN,
    "Dependents": _YN,
    "PhoneService": _YN,
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": _ADDON,
    "OnlineBackup": _ADDON,
    "DeviceProtection": _ADDON,
    "TechSupport": _ADDON,
    "StreamingTV": _ADDON,
    "StreamingMovies": _ADDON,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": _YN,
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}
