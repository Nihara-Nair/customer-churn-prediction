"""Metrics, threshold selection and plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_threshold(y_true, proba, beta: float = 1.0) -> float:
    """Threshold that maximises F-beta on (out-of-fold) predictions. beta > 1 favours recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]  # align with thresholds
    b2 = beta ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        fbeta = np.where((b2 * precision + recall) > 0, (1 + b2) * precision * recall / (b2 * precision + recall), 0.0)
    return float(thresholds[int(np.argmax(fbeta))])


def top_fraction_capture(y_true, proba, fraction: float = 0.2) -> dict:
    """Share of all churners found in the top `fraction` of customers ranked by predicted risk, and the lift."""
    y_true = np.asarray(y_true)
    k = max(1, int(round(len(y_true) * fraction)))
    order = np.argsort(-np.asarray(proba), kind="stable")[:k]
    captured = float(y_true[order].sum() / max(1, y_true.sum()))
    return {"fraction": fraction, "captured_churners": captured, "lift": captured / fraction}


def plot_roc(curves: dict, path: Path) -> None:
    plt.figure(figsize=(6, 5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves (test set)")
    plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_pr(curves: dict, path: Path) -> None:
    plt.figure(figsize=(6, 5))
    for name, (y, p) in curves.items():
        pr, rc, _ = precision_recall_curve(y, p)
        plt.plot(rc, pr, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    plt.axhline(float(np.mean(next(iter(curves.values()))[0])), color="k", ls="--", lw=1, label="Baseline (churn rate)")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-recall curves (test set)")
    plt.legend(loc="upper right"); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_confusion(metrics: dict, title: str, path: Path) -> None:
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Stay", "Churn"]); ax.set_yticks([0, 1], ["Stay", "Churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_importance(names, means, stds, path: Path, top: int = 12) -> None:
    order = np.argsort(means)[::-1][:top][::-1]
    plt.figure(figsize=(7, 5))
    plt.barh([names[i] for i in order], [means[i] for i in order], xerr=[stds[i] for i in order], color="#0995C2")
    plt.xlabel("Drop in ROC-AUC when the feature is shuffled"); plt.title("Permutation importance (test set)")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
