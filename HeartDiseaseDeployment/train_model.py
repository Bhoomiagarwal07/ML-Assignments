"""
Trains a Random Forest classifier to predict heart disease risk and saves it as model.pkl.
Run this once locally to regenerate the model file if needed:
    python train_model.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Task 1: Data Understanding and Preprocessing ---

# 1. Load the dataset using Pandas
df = pd.read_csv("heart.csv")

# 2. Display the first five records
print("First 5 records:")
print(df.head())

# 3. Identify numerical features and the target variable
# All 13 input columns are numerical (age, sex, cp, trestbps, chol, fbs, restecg,
# thalach, exang, oldpeak, slope, ca, thal). The target variable is 'target'
# (1 = heart disease present, 0 = no heart disease).
feature_columns = [col for col in df.columns if col != "target"]
print("\nNumerical features:", feature_columns)
print("Target variable: target (1 = heart disease, 0 = no heart disease)")

# 4. Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 5. Split the dataset into 80% training and 20% testing
X = df[feature_columns]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# --- Task 2: Model Development ---

# Build a Random Forest classifier (chosen for its strong out-of-the-box performance
# and robustness to the mix of binary/continuous clinical features in this dataset)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate using Accuracy Score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Save the trained model using Joblib, together with the exact feature column order
# it was trained on, so app.py can reliably reconstruct matching input at inference time.
joblib.dump({"model": model, "feature_columns": feature_columns}, "model.pkl")
print("\nModel saved to model.pkl")
