"""Loading and cleaning the raw Telco churn CSV."""
from __future__ import annotations

import pandas as pd

from .config import FEATURES, ID_COL, TARGET


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}. "
                         "Is this the Telco Customer Churn CSV?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy.

    Known quirks of this dataset that are handled here:
      * TotalCharges is stored as text and contains blank strings for customers with tenure == 0
        (brand-new customers who have not been billed yet). These are set to MonthlyCharges * tenure (= 0).
      * SeniorCitizen is 0/1 while every other binary column is Yes/No, so it is mapped to Yes/No.
      * The target 'Churn' is converted from Yes/No to 1/0.
    """
    validate(df)
    df = df.copy()
    df = df.drop(columns=[ID_COL], errors="ignore")

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])

    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str).replace({"0": "No", "1": "Yes"})

    target = df[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if target.isna().any():
        raise ValueError("Churn column contains values other than Yes/No")
    df[TARGET] = target.astype(int)
    return df.reset_index(drop=True)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES].copy(), df[TARGET].copy()


def load_clean(path: str) -> pd.DataFrame:
    return clean(load_raw(path))
