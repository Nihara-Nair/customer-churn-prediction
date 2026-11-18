# Adding this project to your resume and GitHub

## 1. Generate your real numbers
```bash
python -m churn.train --data data/telco_churn.csv
```
Open **`reports/resume_bullets.txt`**. The bullets are auto-filled with the metrics from YOUR run (ROC-AUC, recall,
row count, lift, top drivers), so copy them directly. Do not use numbers from anywhere else.

Resume heading format:
`Customer Churn Prediction | Python, pandas, scikit-learn, matplotlib, Streamlit        Nov 2026`
(Use the real month you finish it. Keep the bullets to 2 lines each so the resume stays on one page.)

## 2. Publish to GitHub
```bash
git init
git add .
git commit -m "Customer churn prediction: pipeline, evaluation and Streamlit app"
git branch -M main
git remote add origin https://github.com/Nihara-Nair/customer-churn-prediction.git
git push -u origin main
```
(Create the empty repo `customer-churn-prediction` on github.com first, without a README.)

Before pushing, check that:
- `reports/` contains your figures and `RESULTS.md` (they are what recruiters see)
- `models/churn_model.joblib` exists so others can run the app after cloning
- `data/telco_churn.csv` is NOT committed (it is git-ignored on purpose)
- the README "Results" section links to or pastes `reports/RESULTS.md`

Repo description: *End-to-end customer churn prediction with scikit-learn and a Streamlit app.*
Topics: `machine-learning`, `scikit-learn`, `churn-prediction`, `streamlit`, `data-science`, `python`.

## 3. Be ready to explain (interview / viva)
See `viva_notes.md`.
