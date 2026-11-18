"""Synthetic stand-in with the SAME schema as the Telco dataset. Used ONLY by the unit tests.
Never report results from this data: it is random, not real customers."""
import numpy as np
import pandas as pd

from churn.config import OPTIONS


def make_telco_like(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customerID": [f"{i:04d}-TEST" for i in range(n)],
        "gender": rng.choice(OPTIONS["gender"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": rng.choice(OPTIONS["Partner"], n),
        "Dependents": rng.choice(OPTIONS["Dependents"], n, p=[0.7, 0.3]),
        "tenure": rng.integers(0, 73, n),
        "PhoneService": rng.choice(OPTIONS["PhoneService"], n, p=[0.1, 0.9]),
        "MultipleLines": rng.choice(OPTIONS["MultipleLines"], n),
        "InternetService": rng.choice(OPTIONS["InternetService"], n, p=[0.35, 0.45, 0.2]),
        "OnlineSecurity": rng.choice(OPTIONS["OnlineSecurity"], n),
        "OnlineBackup": rng.choice(OPTIONS["OnlineBackup"], n),
        "DeviceProtection": rng.choice(OPTIONS["DeviceProtection"], n),
        "TechSupport": rng.choice(OPTIONS["TechSupport"], n),
        "StreamingTV": rng.choice(OPTIONS["StreamingTV"], n),
        "StreamingMovies": rng.choice(OPTIONS["StreamingMovies"], n),
        "Contract": rng.choice(OPTIONS["Contract"], n, p=[0.55, 0.21, 0.24]),
        "PaperlessBilling": rng.choice(OPTIONS["PaperlessBilling"], n),
        "PaymentMethod": rng.choice(OPTIONS["PaymentMethod"], n),
        "MonthlyCharges": np.round(rng.uniform(18, 118, n), 2),
    })
    total = np.round(df["tenure"] * df["MonthlyCharges"], 2).astype(str)
    total[df["tenure"] == 0] = " "              # same quirk as the real file: blank strings
    df["TotalCharges"] = total
    logit = (-0.6 + 1.6 * (df["Contract"] == "Month-to-month") - 0.04 * df["tenure"]
             + 0.9 * (df["InternetService"] == "Fiber optic") + 0.012 * (df["MonthlyCharges"] - 65)
             + 0.5 * (df["PaymentMethod"] == "Electronic check"))
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    return df
