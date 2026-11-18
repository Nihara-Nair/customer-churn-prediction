"""Train, compare and evaluate churn models.

    python -m churn.train --data data/telco_churn.csv

Outputs (relative to --out, default "."):
    models/churn_model.joblib        best pipeline + decision threshold
    reports/metrics.json             every number reported below
    reports/model_comparison.csv     CV + test metrics for each model
    reports/RESULTS.md               human readable summary
    reports/resume_bullets.txt       resume bullets auto-filled with YOUR real results
    reports/figures/*.png            ROC, PR, confusion matrix, permutation importance
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split

from . import evaluate as ev
from .config import FEATURES
from .data import load_clean, split_xy
from .models import get_models


def run_training(data_path: str, out_dir: str = ".", seed: int = 42, test_size: float = 0.2, cv: int = 5,
                 beta: float = 1.0, fast: bool = False, verbose: bool = True) -> dict:
    out = Path(out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    fig_dir = out / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    df = load_clean(data_path)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    log(f"Rows: {len(df):,} | churn rate: {y.mean():.1%} | train {len(X_train):,} / test {len(X_test):,}")

    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    fitted, rows, test_proba = {}, [], {}
    for name, (pipe, grid) in get_models(seed, fast).items():
        log(f"Tuning {name} ...")
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=folds, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        # decision threshold is chosen ONLY from out-of-fold predictions on the training data
        oof = cross_val_predict(clone(best), X_train, y_train, cv=folds, method="predict_proba")[:, 1]
        thr = ev.best_threshold(y_train, oof, beta)
        proba = best.predict_proba(X_test)[:, 1]
        m = ev.compute_metrics(y_test, proba, thr)
        fitted[name], test_proba[name] = (best, thr), proba
        rows.append({"model": name, "cv_roc_auc": float(search.best_score_), "best_params": search.best_params_, **m})
        log(f"  CV ROC-AUC {search.best_score_:.3f} | test ROC-AUC {m['roc_auc']:.3f} | "
            f"recall {m['recall']:.3f} | precision {m['precision']:.3f}")

    comparison = pd.DataFrame(rows)
    best_name = comparison.sort_values("cv_roc_auc", ascending=False).iloc[0]["model"]  # chosen on CV, never on test
    best_pipe, best_thr = fitted[best_name]
    best_proba = test_proba[best_name]
    final = ev.compute_metrics(y_test, best_proba, best_thr)
    at_half = ev.compute_metrics(y_test, best_proba, 0.5)
    capture = ev.top_fraction_capture(y_test, best_proba, 0.2)

    imp = permutation_importance(best_pipe, X_test, y_test, scoring="roc_auc", n_repeats=10,
                                 random_state=seed, n_jobs=-1)
    importance = (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean, "std": imp.importances_std})
                  .sort_values("importance", ascending=False).reset_index(drop=True))

    curves = {n: (y_test, p) for n, p in test_proba.items()}
    ev.plot_roc(curves, fig_dir / "roc_curves.png")
    ev.plot_pr(curves, fig_dir / "pr_curves.png")
    ev.plot_confusion(final, f"{best_name} (threshold {best_thr:.2f})", fig_dir / "confusion_matrix.png")
    ev.plot_importance(importance["feature"].tolist(), importance["importance"].to_numpy(),
                       importance["std"].to_numpy(), fig_dir / "feature_importance.png")

    joblib.dump({"pipeline": best_pipe, "threshold": best_thr, "model_name": best_name, "features": FEATURES},
                out / "models" / "churn_model.joblib")

    results = {
        "dataset": {"rows": int(len(df)), "churn_rate": float(y.mean()), "n_train": int(len(X_train)),
                    "n_test": int(len(X_test)), "source_file": Path(data_path).name},
        "settings": {"seed": seed, "cv_folds": cv, "test_size": test_size, "beta": beta,
                     "class_imbalance_handling": "class weights", "model_selection": "best CV ROC-AUC on train split"},
        "best_model": best_name,
        "test_metrics_tuned_threshold": final,
        "test_metrics_threshold_0.5": at_half,
        "top_20pct_capture": capture,
        "top_features": importance.head(5)["feature"].tolist(),
        "comparison": rows,
    }
    (out / "reports" / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    comparison.drop(columns=["best_params"]).round(4).to_csv(out / "reports" / "model_comparison.csv", index=False)
    importance.round(5).to_csv(out / "reports" / "feature_importance.csv", index=False)
    (out / "reports" / "RESULTS.md").write_text(_results_md(results, comparison, importance), encoding="utf-8")
    (out / "reports" / "resume_bullets.txt").write_text(_resume_bullets(results), encoding="utf-8")
    log(f"\nBest model: {best_name} | test ROC-AUC {final['roc_auc']:.3f} | recall {final['recall']:.3f} | "
        f"precision {final['precision']:.3f} | F1 {final['f1']:.3f}")
    log("Saved model, metrics, figures and resume bullets under", out.resolve())
    return results


def _results_md(r: dict, comparison: pd.DataFrame, importance: pd.DataFrame) -> str:
    f, h, c = r["test_metrics_tuned_threshold"], r["test_metrics_threshold_0.5"], r["top_20pct_capture"]
    table = comparison[["model", "cv_roc_auc", "roc_auc", "pr_auc", "precision", "recall", "f1", "threshold"]].round(3)
    header = "| " + " | ".join(table.columns) + " |\n|" + "---|" * len(table.columns) + "\n"
    body = "".join("| " + " | ".join(str(v) for v in row) + " |\n" for row in table.itertuples(index=False))
    top = "\n".join(f"{i + 1}. {row.feature} ({row.importance:.4f})" for i, row in importance.head(8).iterrows())
    d = r["dataset"]
    return (f"# Results\n\nDataset: `{d['source_file']}`, {d['rows']:,} customers, churn rate {d['churn_rate']:.1%}. "
            f"Stratified split: {d['n_train']:,} train / {d['n_test']:,} test (test set untouched until final evaluation).\n\n"
            f"## Model comparison (test set; `cv_roc_auc` is 5-fold CV on the training split)\n\n{header}{body}\n"
            f"## Selected model: {r['best_model']}\n\n"
            f"| Metric | Tuned threshold ({f['threshold']}) | Default 0.5 |\n|---|---|---|\n"
            f"| ROC-AUC | {f['roc_auc']:.3f} | {h['roc_auc']:.3f} |\n| PR-AUC | {f['pr_auc']:.3f} | {h['pr_auc']:.3f} |\n"
            f"| Precision | {f['precision']:.3f} | {h['precision']:.3f} |\n| Recall | {f['recall']:.3f} | {h['recall']:.3f} |\n"
            f"| F1 | {f['f1']:.3f} | {h['f1']:.3f} |\n| Accuracy | {f['accuracy']:.3f} | {h['accuracy']:.3f} |\n\n"
            f"Top 20% highest-risk customers contain **{c['captured_churners']:.1%}** of all churners "
            f"(lift {c['lift']:.1f}x vs. random targeting).\n\n"
            f"## Top drivers (permutation importance, drop in ROC-AUC)\n\n{top}\n\n"
            f"![ROC](figures/roc_curves.png) ![Importance](figures/feature_importance.png)\n")


def _resume_bullets(r: dict) -> str:
    d, f, c = r["dataset"], r["test_metrics_tuned_threshold"], r["top_20pct_capture"]
    drivers = ", ".join(r["top_features"][:3])
    return (
        "RESUME BULLETS (auto-generated from your real run, so every number is true for your results)\n\n"
        f"Customer Churn Prediction | Python, pandas, scikit-learn, matplotlib, Streamlit\n"
        f"- Built an end-to-end churn prediction pipeline on {d['rows']:,} telecom customer records, comparing Logistic "
        f"Regression, Random Forest and Gradient Boosting with 5-fold cross-validation; the best model ({r['best_model']}) "
        f"reached {f['roc_auc']:.2f} ROC-AUC and {f['recall']:.0%} recall on a held-out test set.\n"
        f"- Addressed class imbalance ({d['churn_rate']:.0%} churn) with class weights and out-of-fold threshold tuning; "
        f"the top 20% highest-risk customers captured {c['captured_churners']:.0%} of all churners ({c['lift']:.1f}x lift).\n"
        f"- Identified key churn drivers through EDA and permutation importance ({drivers}) and built a Streamlit app "
        f"for real-time churn risk scoring.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate churn models")
    ap.add_argument("--data", default="data/telco_churn.csv")
    ap.add_argument("--out", default=".")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--beta", type=float, default=1.0, help="F-beta used for threshold tuning (>1 favours recall)")
    args = ap.parse_args()
    if not Path(args.data).exists():
        raise SystemExit(f"Dataset not found at {args.data}. See data/README.md for download instructions.")
    run_training(args.data, args.out, args.seed, beta=args.beta)


if __name__ == "__main__":
    main()
