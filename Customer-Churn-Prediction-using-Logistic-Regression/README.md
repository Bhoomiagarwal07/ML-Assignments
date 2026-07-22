# Assignment 2 — Customer Churn Prediction using Logistic Regression

## 📌 Objective
Build a **Logistic Regression** model that predicts whether a telecom customer is likely to
**churn** (leave the company) based on their demographic information and service usage
patterns.

## 📊 Dataset
**Telco Customer Churn Dataset** (7043 rows, 21 columns)
Source: [Kaggle — blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

*(The dataset is not uploaded to this repo per assignment instructions — use the Kaggle link above, or the notebook loads it automatically from a public mirror.)*

## 🛠️ Libraries Used
- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `scikit-learn` — train/test split, feature scaling, Logistic Regression, evaluation metrics
- `matplotlib` / `seaborn` — visualization (confusion matrix heatmap)

## 🔍 Methodology
1. **Data Understanding** — loaded the dataset, inspected structure, identified numerical
   features (`tenure`, `MonthlyCharges`, `TotalCharges`), 16 categorical features, and the
   target variable (`Churn`). Checked class balance (~73% no-churn / 27% churn).
2. **Data Preprocessing** — discovered and fixed a hidden data quality issue (`TotalCharges`
   stored as text with 11 blank values), filled missing values with the median, dropped the
   non-predictive `customerID` column, binary-encoded 2-category columns, one-hot encoded
   multi-category columns, split 80/20 with stratification, and scaled numerical features.
3. **Model Development** — trained a `LogisticRegression` model on the training set and
   generated churn predictions (both class labels and probabilities) on the test set.
4. **Model Evaluation** — evaluated using Accuracy, Precision, Recall, and F1-Score, and
   visualized results with a confusion matrix heatmap.

## 📈 Results

| Metric | Value |
|--------|-------|
| Accuracy  | ≈ 80.6% |
| Precision | ≈ 65.7% |
| Recall    | ≈ 55.9% |
| F1-Score  | ≈ 60.4% |

**Key finding:** `Contract` type and `tenure` are the strongest predictors of churn —
month-to-month customers with short tenure churn far more often than those on long-term
contracts. Fiber-optic internet customers also show elevated churn risk.

## ✅ Conclusion
This project built a Logistic Regression model to predict customer churn using demographic
and service-usage data from a telecom provider. The model achieved an accuracy of about
80.6%, with a precision of 65.7%, recall of 55.9%, and F1-score of 60.4%. The most important
factors influencing churn were **contract type** and **tenure** — customers on month-to-month
contracts and those with shorter tenure were far more likely to leave, while customers with
fiber-optic internet also showed elevated churn risk. These results suggest that contract
length and early customer experience are key retention levers for the business. One key
limitation of Logistic Regression here is that it assumes a **linear relationship between
each feature and the log-odds of churn**, and cannot automatically capture more complex,
non-linear interactions between features (for example, how tenure and contract type jointly
affect churn risk) without manually engineering interaction terms — a limitation that
tree-based models like Random Forest or Gradient Boosting can often handle more naturally.

## 📂 Files
- `Assignment-2.ipynb` — full notebook with code, outputs, and visualizations
