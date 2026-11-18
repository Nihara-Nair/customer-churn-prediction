"""Exploratory data analysis figures.   python -m churn.eda --data data/telco_churn.csv"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import TARGET  # noqa: E402
from .data import load_clean  # noqa: E402


def churn_rate_by(df: pd.DataFrame, col: str) -> pd.Series:
    return (df.groupby(col, observed=True)[TARGET].mean() * 100).sort_values(ascending=False)


def tenure_bucket(df: pd.DataFrame) -> pd.Series:
    return pd.cut(df["tenure"], [-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"])


def run_eda(data_path: str, out_dir: str = ".") -> dict:
    fig_dir = Path(out_dir) / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = load_clean(data_path)

    summary = {"rows": int(len(df)), "churn_rate_pct": round(float(df[TARGET].mean() * 100), 2)}
    for col in ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling", "SeniorCitizen"]:
        summary[f"churn_rate_by_{col}"] = churn_rate_by(df, col).round(1).to_dict()
    summary["churn_rate_by_tenure"] = (df.groupby(tenure_bucket(df), observed=True)[TARGET].mean() * 100).round(1).to_dict()
    summary["churn_rate_by_tenure"] = {str(k): v for k, v in summary["churn_rate_by_tenure"].items()}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0, 0]
    counts = df[TARGET].map({0: "Stayed", 1: "Churned"}).value_counts()
    ax.bar(counts.index, counts.values, color=["#9ecae1", "#0995C2"]); ax.set_title("Churn distribution")
    for ax, col in zip([axes[0, 1], axes[0, 2], axes[1, 0]], ["Contract", "InternetService", "PaymentMethod"]):
        s = churn_rate_by(df, col)
        ax.bar(range(len(s)), s.values, color="#0995C2"); ax.set_xticks(range(len(s)), s.index, rotation=20, ha="right")
        ax.set_ylabel("Churn rate (%)"); ax.set_title(f"Churn rate by {col}")
    ax = axes[1, 1]
    b = df.groupby(tenure_bucket(df), observed=True)[TARGET].mean() * 100
    ax.bar([str(i) for i in b.index], b.values, color="#0995C2"); ax.set_title("Churn rate by tenure (months)")
    ax.set_ylabel("Churn rate (%)")
    ax = axes[1, 2]
    for label, g in df.groupby(TARGET):
        ax.hist(g["MonthlyCharges"], bins=30, alpha=0.6, label="Churned" if label else "Stayed")
    ax.set_title("Monthly charges"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "eda_overview.png", dpi=150); plt.close(fig)

    num = df[["tenure", "MonthlyCharges", "TotalCharges", TARGET]].corr()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(num, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num)), num.columns, rotation=30, ha="right"); ax.set_yticks(range(len(num)), num.columns)
    for i in range(len(num)):
        for j in range(len(num)):
            ax.text(j, i, f"{num.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im); ax.set_title("Correlation (numeric features and churn)")
    fig.tight_layout(); fig.savefig(fig_dir / "eda_correlation.png", dpi=150); plt.close(fig)

    (Path(out_dir) / "reports" / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/telco_churn.csv")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    s = run_eda(a.data, a.out)
    print(json.dumps(s, indent=2))
