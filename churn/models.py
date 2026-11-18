"""Candidate models and their (small) hyper-parameter grids. Class imbalance is handled with class weights."""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import build_preprocessor


def get_models(seed: int = 42, fast: bool = False) -> dict[str, tuple[Pipeline, dict]]:
    def pipe(model):
        return Pipeline([("prep", build_preprocessor()), ("model", model)])

    if fast:  # tiny grids, used by the unit tests
        grids = {
            "Logistic Regression": {"model__C": [1]},
            "Random Forest": {"model__max_depth": [8]},
            "Gradient Boosting": {"model__max_iter": [60]},
        }
    else:
        grids = {
            "Logistic Regression": {"model__C": [0.1, 1, 10]},
            "Random Forest": {"model__max_depth": [6, 10, None], "model__min_samples_leaf": [1, 5, 10]},
            "Gradient Boosting": {"model__learning_rate": [0.05, 0.1], "model__max_depth": [3, 5],
                                  "model__max_iter": [150]},
        }
    return {
        "Logistic Regression": (pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
                                grids["Logistic Regression"]),
        "Random Forest": (pipe(RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                                      random_state=seed, n_jobs=-1)),
                          grids["Random Forest"]),
        "Gradient Boosting": (pipe(HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)),
                              grids["Gradient Boosting"]),
    }
