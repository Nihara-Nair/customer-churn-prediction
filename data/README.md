# Customer Churn Prediction

A Machine Learning project that analyzes customer behavior, identifies churn patterns, and predicts the likelihood of customers leaving a telecommunications service. The project includes Exploratory Data Analysis (EDA), machine learning model comparison, performance evaluation, and an interactive Streamlit dashboard.

## Project Overview

Customer churn refers to the situation in which customers stop using a company's products or services. Predicting customer churn helps businesses identify customers who are more likely to leave and take appropriate retention measures.

This project uses the IBM Telco Customer Churn dataset to analyze customer information, identify factors influencing churn, and build machine learning models to predict customer churn probability.

## Features

- **Exploratory Data Analysis:** Analyzes customer information and identifies patterns associated with churn.
- **Data Preprocessing:** Handles numerical and categorical features for machine learning.
- **Machine Learning:** Trains and compares multiple classification algorithms.
- **Model Evaluation:** Evaluates models using ROC-AUC, precision, recall, and F1-score.
- **Churn Prediction:** Estimates the probability of a customer leaving the service.
- **Interactive Dashboard:** Displays customer statistics, EDA visualizations, model performance, and individual customer predictions using Streamlit.

## Dataset

This project uses the **Telco Customer Churn** dataset (IBM sample data, hosted on Kaggle):

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

1. Log in to Kaggle (free) and download the dataset.
2. Extract `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
3. Save it in the `data` folder as `telco_churn.csv` (rename it).

**Expected dataset:** 7,043 rows and 21 columns (customerID, 19 features, and the `Churn` target).

The CSV is git-ignored because the dataset is not redistributed with this repository.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and model evaluation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Interactive dashboard |

## Machine Learning Models

Three classification algorithms are trained and compared:

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier

The models undergo hyperparameter tuning and evaluation to identify a suitable model for churn prediction.

## Model Performance

The Gradient Boosting model was selected as the final model.

| Evaluation Metric | Score |
|---|---:|
| ROC-AUC | 0.844 |
| Recall | 0.733 |
| Precision | 0.547 |
| F1-Score | 0.626 |

*These scores represent the results from the current training run.*

## Exploratory Data Analysis – Key Findings

The analysis revealed the following customer churn patterns:

- Month-to-month contract customers showed a higher churn rate than customers with one-year or two-year contracts.
- Fiber optic internet customers had a higher churn rate than DSL customers.
- Customers using electronic checks showed a higher churn rate than customers using automatic payment methods.
- Customers with shorter tenure were more likely to leave the service.

These findings help identify customer segments that may require targeted retention strategies.

## Dashboard

The project includes an interactive Streamlit dashboard with four main sections:

### 1. Overview
Displays total customers, churned customers, retained customers, overall churn rate, and dataset information.

### 2. Exploratory Data Analysis
Provides visualizations to understand customer churn patterns and relationships between customer attributes and churn.

### 3. Model Performance
Displays model evaluation metrics and visualizations, including confusion matrix, ROC curve, precision-recall curve, and feature importance.

### 4. Customer Prediction
Allows users to enter customer details and obtain a churn prediction with an estimated churn probability and risk category.

## Project Structure

```text
customer-churn-prediction/
│
├── app/
│   └── streamlit_app.py
│
├── churn/
│   ├── config.py
│   ├── eda.py
│   ├── train.py
│   └── predict.py
│
├── data/
│   └── telco_churn.csv
│
├── models/
│
├── reports/
│   ├── figures/
│   ├── metrics.json
│   └── RESULTS.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

*The structure represents the main project components. Additional generated files may also be present.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Nihara-Nair/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset

Download the dataset from Kaggle and save it inside the `data` folder with the filename:

```text
telco_churn.csv
```

### 5. Run Exploratory Data Analysis

```bash
python -m churn.eda
```

### 6. Train the Models

```bash
python -m churn.train
```

This trains and evaluates the machine learning models and generates the model artifacts, evaluation metrics, and visualizations.

### 7. Launch the Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

Open the local URL displayed in the terminal, usually:

http://localhost:8501

## Future Improvements

- Deploy the application online for public access.
- Improve model performance through additional feature engineering and experimentation.
- Integrate customer-specific retention recommendations.
- Connect the application to a live customer database.

## Author

**Nihara Nair**

GitHub: https://github.com/Nihara-Nair

Project Repository: https://github.com/Nihara-Nair/customer-churn-prediction