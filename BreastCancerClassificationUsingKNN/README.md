# Assignment 4 — Breast Cancer Classification using K-Nearest Neighbors (KNN)

## 📌 Objective
Build a **K-Nearest Neighbors (KNN)** classification model to predict whether a breast tumor
is **Malignant (M)** or **Benign (B)** based on diagnostic measurements computed from
digitized images of breast mass cell nuclei.

## 📊 Dataset
**Breast Cancer Wisconsin Diagnostic Dataset** (569 rows, 33 columns)
Source: [Kaggle — uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

*(The dataset is not uploaded to this repo per assignment instructions — use the Kaggle link above, or the notebook loads it automatically from a public mirror.)*

## 🛠️ Libraries Used
- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `scikit-learn` — train/test split, feature scaling, KNN classifier, evaluation metrics
- `matplotlib` / `seaborn` — visualization (confusion matrix heatmap)

## 🔍 Methodology
1. **Data Understanding** — loaded the dataset, identified 30 numerical diagnostic features
   and the target variable (`diagnosis`), and checked class balance (~63% Benign / 37%
   Malignant).
2. **Data Preprocessing** — checked for missing values (found the classic empty
   `Unnamed: 32` artifact column), removed the non-predictive `id` column and the empty
   column, encoded `diagnosis` (M→1, B→0), split 80/20 with stratification, and standardized
   all features using `StandardScaler`.
3. **Model Development** — trained a `KNeighborsClassifier` with **K = 5** and predicted
   diagnoses on the test set.
4. **Model Evaluation** — evaluated using Accuracy, Precision, Recall, and F1-Score,
   visualized results with a confusion matrix heatmap, and directly compared scaled vs
   unscaled feature performance to demonstrate why scaling matters for KNN.

## 📈 Results

| Metric | Value (Scaled Features) | Value (Unscaled Features) |
|--------|--------------------------|------------------------------|
| Accuracy  | ≈ 95.6% | ≈ 91.2% |
| Precision | ≈ 97.4% | — |
| Recall    | ≈ 90.5% | — |
| F1-Score  | ≈ 93.8% | ≈ 86.8% |

**Key finding:** Feature scaling made a clear, measurable difference in KNN performance,
confirming that unscaled features with larger numeric ranges (like `area_mean`) distort
distance-based predictions.

## ✅ Conclusion
This project built a K-Nearest Neighbors (K=5) classifier to predict whether a breast tumor is
malignant or benign using 30 diagnostic measurements derived from digitized cell nuclei images.
After removing the non-predictive `id` column and the empty `Unnamed: 32` column, encoding the
target variable, and standardizing all features, the model achieved 95.6% accuracy, 97.4%
precision, 90.5% recall, and an F1-score of 93.8% on the test set. Feature scaling proved
essential for KNN specifically, since the algorithm relies on distance calculations between
points — without scaling, features with naturally larger numeric ranges (like `area_mean`)
would dominate the distance metric and distort predictions, and indeed accuracy dropped
noticeably when scaling was removed. One key limitation of KNN is that it is a **computationally
expensive, "lazy" learner**: since it stores the entire training dataset and calculates
distances to all training points for every single prediction, it becomes slow and memory-heavy
on larger datasets, unlike models that learn a compact set of parameters during training (such
as Logistic Regression).

## 📂 Files
- `Assignment-4.ipynb` — full notebook with code, outputs, and visualizations
