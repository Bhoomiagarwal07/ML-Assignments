# Assignment 5 — Employee Attrition Prediction using Decision Tree and Random Forest

## 📌 Objective
Build and compare **Decision Tree** and **Random Forest** classification models to predict
whether an employee is likely to leave the organization (**attrition**), based on
demographic, professional, and work-related attributes.

## 📊 Dataset
**IBM HR Analytics Employee Attrition & Performance Dataset** (1470 rows, 35 columns)
Source: [Kaggle — pavansubhasht/ibm-hr-analytics-attrition-dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

*(The dataset is not uploaded to this repo per assignment instructions — use the Kaggle link above, or the notebook loads it automatically from a public mirror hosted directly by IBM.)*

## 🛠️ Libraries Used
- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `scikit-learn` — train/test split, Decision Tree, Random Forest, evaluation metrics
- `matplotlib` / `seaborn` — visualization (confusion matrices, feature importance plot)

## 🔍 Methodology
1. **Data Understanding** — loaded the dataset, identified numerical/categorical features and
   the target (`Attrition`), and checked class balance (~84% stay / 16% leave — significantly
   imbalanced).
2. **Data Preprocessing** — checked for missing values (none found), removed 4 non-predictive
   columns (`EmployeeCount`, `Over18`, `StandardHours` — all constant; `EmployeeNumber` — a
   unique ID), encoded the target and one-hot encoded all categorical features, and split
   80/20 with stratification.
3. **Model Development** — trained two models on the identical training set: a
   `DecisionTreeClassifier` and a `RandomForestClassifier` with 100 estimators.
4. **Model Evaluation & Comparison** — evaluated both models using Accuracy, Precision,
   Recall, and F1-Score, visualized confusion matrices for both, and generated a feature
   importance plot for the Random Forest model.
5. **Bonus** — tuned the Decision Tree's `max_depth` hyperparameter across several values to
   observe its effect on accuracy and F1-score.

## 📈 Results

| Metric | Decision Tree | Random Forest |
|--------|---------------|----------------|
| Accuracy  | ≈ 76.5% | ≈ 83.3% |
| Precision | ≈ 31.0% | ≈ 41.7% |
| Recall    | ≈ 38.3% | ≈ 10.6% |
| F1-Score  | ≈ 34.3% | ≈ 16.9% |

## 🔬 Model Comparison
Random Forest achieved **higher overall accuracy**, but Decision Tree achieved **notably
higher recall** on the minority "leaves" class. This is a genuinely important finding: on this
imbalanced dataset (~84%/16%), Random Forest's higher accuracy comes partly from being more
conservative and predicting "stays" more often — which boosts accuracy but causes it to miss
more actual leavers. This demonstrates why accuracy alone can be misleading for imbalanced
classification problems like attrition prediction, where recall on the minority class often
matters more from a business standpoint.

`MonthlyIncome`, `Age`, `TotalWorkingYears`, `DailyRate`, and `OverTime` emerged as the top
predictors of attrition according to the Random Forest's feature importance scores.

**Bonus tuning finding:** Limiting the Decision Tree's `max_depth` to 10 improved F1-score
from ~0.343 (unlimited depth) to ~0.370, while very shallow trees (`max_depth=3` or `5`)
achieved higher accuracy but notably lower F1-scores — illustrating the classic
bias-variance tradeoff in tree depth tuning.

## ✅ Conclusion
This project built and compared Decision Tree and Random Forest classifiers to predict
employee attrition using the IBM HR Analytics dataset. Random Forest achieved higher overall
accuracy (83.3% vs. 76.5%), but Decision Tree achieved notably higher recall for the minority
"leaves" class (38.3% vs. 10.6%), showing that the better-looking model on accuracy alone is
not necessarily the better model for this specific business problem, where catching actual
attrition risk matters more than overall correctness. Random Forest generally outperforms a
single Decision Tree because it trains many trees on random subsets of data and features and
combines their votes, which reduces the variance and overfitting that a single tree is prone
to, producing more stable and generalizable predictions. One key limitation of Decision Trees
is that they are **prone to overfitting** on training data, especially when grown to full
depth, since a single tree can memorize noise and outliers rather than learning general
patterns. One key limitation of Random Forest is that it is **much less interpretable** than a
single Decision Tree — while a decision tree's rules can be read and explained step-by-step,
a forest of 100 trees voting together is effectively a "black box," making it harder to explain
individual predictions to non-technical stakeholders such as HR managers, even though feature
importance scores can still offer a partial explanation of which factors matter most overall.

## 📂 Files
- `Assignment-5.ipynb` — full notebook with code, outputs, and visualizations
