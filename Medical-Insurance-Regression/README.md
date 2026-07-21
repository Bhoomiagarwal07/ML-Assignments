Assignment 1 — Medical Insurance Cost Prediction using Multiple Linear Regression
📌 Objective

Build a Multiple Linear Regression model that predicts a person's medical insurance charges using their personal and health-related information — age, sex, BMI, number of children, smoking status, and region.

📊 Dataset

Medical Cost Personal Insurance Dataset (1338 rows, 7 columns) Source: Kaggle — mirichoi0218/insurance

(The dataset is not uploaded to this repo per assignment instructions — use the Kaggle link above, or the notebook loads it automatically from a public mirror.)

🛠️ Libraries Used
pandas — data loading and manipulation
numpy — numerical operations
scikit-learn — train/test split, Linear Regression model, evaluation metrics
matplotlib — visualization (Actual vs Predicted plot)
🔍 Methodology
Data Understanding — loaded the dataset, inspected structure, identified numerical features (age, bmi, children), categorical features (sex, smoker, region), and the target variable (charges).
Data Preprocessing — checked for missing values (none found), encoded sex and smoker as binary (0/1), one-hot encoded region, and split the data 80/20 into training and testing sets.
Model Development — trained a LinearRegression model from scikit-learn on the training set and generated predictions on the held-out test set.
Model Evaluation — evaluated using MAE, MSE, RMSE, and R² score, and visualized performance with an Actual vs Predicted scatter plot.
📈 Results
Metric	Value
MAE	≈ $4,181
MSE	≈ 33,596,916
RMSE	≈ $5,796
R² Score	≈ 0.78

Key finding: smoker status is by far the strongest predictor of insurance charges, followed by bmi and age. sex and region have comparatively minor effects.

✅ Conclusion

This project built a Multiple Linear Regression model to predict medical insurance charges using age, sex, BMI, number of children, smoking status, and region. The model achieved an R² score of approximately 0.78, meaning it explains about 78% of the variation in charges, with a mean absolute error of roughly $4,180. The most influential factor by far was smoking status, followed by BMI and age, while sex and region had much smaller effects. These findings align with real-world expectations, since smoking and high BMI are well-documented health risk factors that insurers price heavily into premiums. One key limitation of Linear Regression here is that it assumes a purely linear, additive relationship between features and charges, but in reality, factors like smoking and BMI likely interact (e.g., a high BMI matters far more for a smoker than a non-smoker), which a simple linear model cannot capture without manually adding interaction terms or using a non-linear model instead.

📂 Files
Assignment-1.ipynb — full notebook with code, outputs, and visualizations