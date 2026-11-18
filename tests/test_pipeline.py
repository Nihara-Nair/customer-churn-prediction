import json
import tempfile
import unittest
from pathlib import Path

from churn.config import FEATURES
from churn.evaluate import best_threshold, compute_metrics, top_fraction_capture
from churn.predict import load_model, predict
from churn.train import run_training
from tests.synthetic import make_telco_like


class MetricTests(unittest.TestCase):
    def test_metrics_perfect_and_threshold(self):
        y, p = [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
        m = compute_metrics(y, p, 0.5)
        self.assertEqual((m["tn"], m["fp"], m["fn"], m["tp"]), (2, 0, 0, 2))
        self.assertEqual(m["roc_auc"], 1.0)
        self.assertTrue(0.2 < best_threshold(y, p) <= 0.8)

    def test_top_fraction_capture(self):
        y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        p = [0.9, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.05, 0.01]
        c = top_fraction_capture(y, p, 0.2)
        self.assertEqual(c["captured_churners"], 1.0)
        self.assertEqual(c["lift"], 5.0)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        cls.csv = cls.out / "telco.csv"
        make_telco_like(2000).to_csv(cls.csv, index=False)
        cls.results = run_training(str(cls.csv), str(cls.out), fast=True, verbose=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_artifacts_written(self):
        for f in ["models/churn_model.joblib", "reports/metrics.json", "reports/model_comparison.csv",
                  "reports/RESULTS.md", "reports/resume_bullets.txt", "reports/figures/roc_curves.png",
                  "reports/figures/feature_importance.png", "reports/figures/confusion_matrix.png",
                  "reports/figures/pr_curves.png"]:
            self.assertTrue((self.out / f).exists(), f)

    def test_split_is_disjoint_and_stratified(self):
        d = self.results["dataset"]
        self.assertEqual(d["n_train"] + d["n_test"], d["rows"])
        self.assertAlmostEqual(d["n_test"] / d["rows"], 0.2, places=2)

    def test_model_learns_signal_and_numbers_consistent(self):
        t = self.results["test_metrics_tuned_threshold"]
        self.assertGreater(t["roc_auc"], 0.7)
        self.assertEqual(t["tn"] + t["fp"] + t["fn"] + t["tp"], self.results["dataset"]["n_test"])
        self.assertTrue(0 < t["threshold"] < 1)
        saved = json.loads((self.out / "reports" / "metrics.json").read_text())
        self.assertEqual(saved["best_model"], self.results["best_model"])
        self.assertEqual(len(saved["comparison"]), 3)

    def test_best_model_chosen_on_cv_not_test(self):
        best_cv = max(self.results["comparison"], key=lambda r: r["cv_roc_auc"])["model"]
        self.assertEqual(self.results["best_model"], best_cv)

    def test_resume_bullets_contain_real_numbers(self):
        text = (self.out / "reports" / "resume_bullets.txt").read_text()
        self.assertIn(f"{self.results['dataset']['rows']:,}", text)
        self.assertIn(self.results["best_model"], text)
        self.assertNotIn("__", text)

    def test_predict_sensible_and_validates(self):
        bundle = load_model(self.out / "models" / "churn_model.joblib")
        risky = {f: v[0] for f, v in __import__("churn.config", fromlist=["OPTIONS"]).OPTIONS.items()}
        risky.update({"Contract": "Month-to-month", "InternetService": "Fiber optic", "PaymentMethod": "Electronic check",
                      "tenure": 2, "MonthlyCharges": 105.0})
        safe = dict(risky, Contract="Two year", InternetService="DSL", PaymentMethod="Credit card (automatic)",
                    tenure=60, MonthlyCharges=40.0)
        p_risky = predict(risky, bundle).iloc[0]["churn_probability"]   # TotalCharges auto-computed
        p_safe = predict(safe, bundle).iloc[0]["churn_probability"]
        self.assertGreater(p_risky, p_safe)
        with self.assertRaises(ValueError):
            predict({"tenure": 1}, bundle)
        self.assertEqual(len(FEATURES), 19)


if __name__ == "__main__":
    unittest.main()
