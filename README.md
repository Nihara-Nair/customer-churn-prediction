# Customer Churn Prediction

End-to-end machine learning project that predicts which telecom customers are likely to cancel their service, explains
what drives churn, and serves predictions through a Streamlit app.

**Stack:** Python, pandas, scikit-learn, matplotlib, Streamlit.

## What it does

1. **Cleans** the Telco Customer Churn data (fixes the text-typed `TotalCharges` column that is blank for brand-new customers).
2. **Explores** churn by contract type, tenure, payment method and charges (`churn/eda.py`, notebook).
3. **Compares three models** (Logistic Regression, Random Forest, Gradient Boosting) with 5-fold cross-validated tuning.
4. **Handles class imbalance** (about 27% of customers churn) with class weights, and tunes the decision threshold on
   out-of-fold predictions so recall and precision are balanced.
5. **Evaluates honestly** on a held-out, stratified test set that is only used once at the end:
   ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix and "how many churners are in the top 20% highest-risk customers".
6. **Explains** the model with permutation importance on the original features.
7. **Serves** predictions in a Streamlit app (`app/streamlit_app.py`).

## Results

Run the training once and paste the generated summary here: **`reports/RESULTS.md`** (all numbers, tables and figures are
produced by the code, so nothing is typed by hand).

## Quick start (VS Code, Windows / Mac / Linux)

```bash
# 1. open this folder in VS Code, then open the terminal (Ctrl + `)
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# 2. get the data: see data/README.md (download from Kaggle, save as data/telco_churn.csv)

# 3. run the project
python -m churn.eda                 # EDA figures + reports/eda_summary.json
python -m churn.train               # trains, evaluates, saves model + reports (about 1-3 minutes)
streamlit run app/streamlit_app.py  # opens the prediction app in your browser
python -m pytest tests -q           # 14 tests (use synthetic data, so no dataset needed)
```

Optional: `jupyter notebook notebooks/churn_analysis.ipynb` (or open it directly in VS Code) for the step-by-step walkthrough.
Use `python -m churn.train --beta 2` to favour recall over precision when picking the threshold.

## Project structure

```
churn/
  config.py      column lists and allowed values
  data.py        loading, validation, cleaning
  features.py    preprocessing (imputation, scaling, one-hot) inside the sklearn Pipeline
  models.py      the 3 candidate models and tuning grids
  train.py       tuning, selection, threshold, evaluation, plots, saved model, auto-written reports
  evaluate.py    metrics, threshold selection, lift, plots
  predict.py     load model and score new customers
  eda.py         exploratory figures
app/streamlit_app.py   prediction web app
notebooks/churn_analysis.ipynb
tests/           unit tests (synthetic data with the same schema as the real dataset)
docs/            resume_and_github.md, viva_notes.md
```

## Methodology notes (why the numbers can be trusted)

- **No data leakage:** the test split is made first; imputing, scaling and encoding are fitted inside the pipeline on training folds only.
- **Model selection uses cross-validation on the training split**, never the test set. The decision threshold is also chosen from out-of-fold training predictions.
- **Accuracy is not the headline metric** because of the class imbalance; ROC-AUC, PR-AUC, recall and precision are reported.
- **Permutation importance** (drop in ROC-AUC when a feature is shuffled) is used for explanations. It is model-agnostic and works on the raw columns.
- **Limitations:** a single dataset from one company, no time dimension, and importance is not causation, so a retention A/B test would be needed to prove that an offer reduces churn.

## Dataset

IBM sample "Telco Customer Churn" data (7,043 customers, 21 columns), available on Kaggle. It is not included in this repo; see `data/README.md`.

## License

MIT
