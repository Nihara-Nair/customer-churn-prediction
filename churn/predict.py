"""Score new customers with the saved model."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .config import FEATURES

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib"


def load_model(path: str | Path = DEFAULT_MODEL) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run: python -m churn.train --data data/telco_churn.csv")
    return joblib.load(path)


def risk_band(p: float) -> str:
    return "Low" if p < 0.3 else "Medium" if p < 0.6 else "High"


def predict(records, bundle: dict) -> pd.DataFrame:
    """records: dict, list of dicts or DataFrame with the 19 feature columns (TotalCharges optional)."""
    df = pd.DataFrame([records]) if isinstance(records, dict) else pd.DataFrame(records).copy()
    if "TotalCharges" not in df.columns and {"tenure", "MonthlyCharges"} <= set(df.columns):
        df["TotalCharges"] = df["tenure"] * df["MonthlyCharges"]
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    proba = bundle["pipeline"].predict_proba(df[FEATURES])[:, 1]
    return pd.DataFrame({
        "churn_probability": proba,
        "prediction": (proba >= bundle["threshold"]).astype(int),
        "risk_band": [risk_band(p) for p in proba],
    })
