# Viva / interview notes

**One-line pitch:** I predict which telecom customers will churn, compare three models, and explain the drivers so a
retention team can target the right customers.

**Why not just accuracy?** Only about 27% of customers churn, so a model that predicts "nobody churns" is about 73%
accurate and useless. I use ROC-AUC, PR-AUC, recall and precision.

**How did you handle imbalance?** Class weights in every model, plus choosing the decision threshold from out-of-fold
predictions instead of using 0.5. (SMOTE is an alternative; I avoided it to keep resampling out of the cross-validation risk area.)

**How did you avoid data leakage?** Train/test split first; imputer, scaler and encoder live inside the Pipeline and are
fitted only on training folds; the model and threshold are chosen without looking at the test set.

**Why those three models?** Logistic Regression is an interpretable baseline, Random Forest captures non-linear
interactions, Gradient Boosting is usually strongest on tabular data. If the simple model wins, that is a valid finding.

**What was dirty in the data?** `TotalCharges` is text with blank values for customers with tenure 0 (not yet billed). I set them to 0
because MonthlyCharges x tenure = 0.

**How do you explain the model?** Permutation importance: shuffle one feature on the test set and measure the drop in ROC-AUC.

**What does "top 20% captures X% of churners" mean?** Rank customers by predicted risk; if we only contact the top 20%,
we reach X% of all churners. Lift = X / 20%. This translates the model into business value.

**Limitations:** single company dataset, no time component, correlation is not causation (need an A/B test), threshold
should be set using real campaign cost versus customer lifetime value.

**Next steps:** cost-sensitive threshold using retention cost and customer value, SHAP explanations, time-based validation, model monitoring.
