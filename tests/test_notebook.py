"""Executes every code cell of the notebook against synthetic data to prove it runs end to end."""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from tests.synthetic import make_telco_like  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class NotebookTests(unittest.TestCase):
    def test_notebook_runs(self):
        tmp = Path(tempfile.mkdtemp())
        proj = tmp / "proj"
        shutil.copytree(ROOT, proj, ignore=shutil.ignore_patterns(".git", "__pycache__", "models", "reports", "*.csv"))
        (proj / "data").mkdir(exist_ok=True)
        make_telco_like(1500).to_csv(proj / "data" / "telco_churn.csv", index=False)
        nb = json.loads((proj / "notebooks" / "churn_analysis.ipynb").read_text())
        old = os.getcwd()
        os.chdir(proj / "notebooks")
        try:
            ns = {}
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    src = "".join(cell["source"]).replace("run_training(DATA_PATH, out_dir=\"..\")",
                                                          "run_training(DATA_PATH, out_dir=\"..\", fast=True, verbose=False)")
                    exec(compile(src, "<cell>", "exec"), ns)
            self.assertIn("results", ns)
            self.assertTrue((proj / "reports" / "figures" / "roc_curves.png").exists())
        finally:
            os.chdir(old)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
