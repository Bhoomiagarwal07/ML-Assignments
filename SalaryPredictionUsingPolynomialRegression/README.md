# Assignment 3 — Salary Prediction using Polynomial Regression

## 📌 Objective
Build a **Polynomial Regression** model to predict an employee's salary based on their
position level, since the relationship between position level and salary is clearly
**non-linear** (salaries stay flat at junior levels and rise sharply at senior/executive
levels).

## 📊 Dataset
**Position Salaries Dataset** (10 rows, 3 columns — `Position`, `Level`, `Salary`)
Source: [Kaggle — akram24/position-salaries](https://www.kaggle.com/datasets/akram24/position-salaries)

*(The dataset is not uploaded to this repo per assignment instructions — use the Kaggle link above, or the notebook loads it automatically from a public mirror.)*

## 🛠️ Libraries Used
- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `scikit-learn` — train/test split, `PolynomialFeatures`, Linear Regression, evaluation metrics
- `matplotlib` — visualization (original scatter plot + polynomial regression curve)

## 🔍 Methodology
1. **Data Understanding** — loaded the dataset, displayed all 10 rows, identified `Level` as
   the input feature and `Salary` as the target, and visualized the raw scatter plot to
   confirm the non-linear relationship.
2. **Data Preprocessing** — checked for missing values (none found), selected `Level` as X and
   `Salary` as y, and split 80/20 into training and testing sets (8 train / 2 test rows, given
   the dataset's small size).
3. **Model Development** — transformed `Level` into polynomial features up to **degree 3**
   using `PolynomialFeatures`, then trained a `LinearRegression` model on the transformed
   features and predicted salaries for the test set.
4. **Model Evaluation** — evaluated using MAE, MSE, and R² score, and visualized the fit with
   a scatter plot overlaid with the smooth polynomial regression curve.

## 📈 Results

| Metric | Value |
|--------|-------|
| MAE    | ≈ $70,635 |
| MSE    | ≈ 6.26 × 10⁹ |
| R² Score | ≈ 0.88 |

**Key finding:** A degree-3 polynomial curve captures the accelerating, non-linear jump in
salary at senior levels far better than a straight line could. Note: with only 10 total data
points, the 2-row test set means these metrics are indicative rather than statistically robust.

## ✅ Conclusion
This project built a Polynomial Regression model (degree 3) to predict employee salaries
from their position level, since the relationship between the two is clearly non-linear —
salaries stay relatively flat at junior levels and rise sharply at senior/executive levels.
The model achieved an R² score of approximately 0.88 on the test set, with a mean absolute
error of about $70,600, reasonably capturing the steep upward curve at higher levels despite
the very small dataset. The key difference between Linear and Polynomial Regression is that
Linear Regression fits a single straight line (`y = b0 + b1*x`), assuming a constant rate of
change, while Polynomial Regression fits a curved line by adding higher-degree terms
(`x²`, `x³`, ...), allowing it to model rates of change that themselves increase or decrease
across the range of the input. The clear advantage of Polynomial Regression for this dataset
is its ability to capture the accelerating, non-linear jump in salary at senior levels — a
pattern that a simple straight-line model would systematically underestimate at the high end
and overestimate at the low end.

## 📂 Files
- `Assignment-3.ipynb` — full notebook with code, outputs, and visualizations
