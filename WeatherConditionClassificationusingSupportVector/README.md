# Assignment 6 — Weather Condition Classification using SVM and Open-Meteo API

## 📌 Objective
Build a **Support Vector Machine (SVM)** classifier with an RBF kernel to predict whether the
weather at a given hour is **Warm** or **Cool**, using live meteorological data fetched
directly from the free Open-Meteo API.

## 📊 Data Source
**Open-Meteo Weather API** (free, no API key required)
Documentation: [https://open-meteo.com/](https://open-meteo.com/)

This notebook pulls **live, real-time hourly weather data** (temperature, relative humidity,
surface pressure, wind speed) for a 7-day forecast window via a simple HTTP GET request — no
dataset file is used or needed.

## 🛠️ Libraries Used
- `requests` — calling the Open-Meteo API
- `pandas` — converting the JSON response into a DataFrame and manipulating it
- `numpy` — numerical operations
- `scikit-learn` — train/test split, feature scaling, SVM classifier, evaluation metrics
- `matplotlib` / `seaborn` — visualization (confusion matrix heatmap)

## 🔍 Methodology
1. **Data Collection & Understanding** — fetched 7 days of hourly weather data (168 rows) for
   New Delhi via the Open-Meteo API, converted the JSON response into a Pandas DataFrame, and
   created the `Weather_Class` target column (`Warm` if temperature ≥ 25°C, else `Cool`).
2. **Data Preprocessing** — checked for missing values (Open-Meteo's forecast data is
   spatially/temporally complete, so none were expected), dropped the non-numerical `Time`
   column, encoded the target (Warm→1, Cool→0), split 80/20 with stratification, and
   standardized all features using `StandardScaler`.
3. **Model Development** — trained an `SVC` classifier with an **RBF kernel** and predicted
   weather class on the test set.
4. **Model Evaluation** — evaluated using Accuracy, Precision, Recall, and F1-Score, and
   visualized results with a confusion matrix heatmap.

## 📈 Results
⚠️ Because this assignment uses **live API data**, exact results will differ depending on the
date and location you run the notebook for. Fill in your own run's numbers here before
submitting, for example:

| Metric | Value |
|--------|-------|
| Accuracy  |0.9706 |
| Precision | 0.9444|
| Recall    | 1.0000 |
| F1-Score  | 0.9714 |

**Key finding:** Since `temperature_2m` is used to construct the `Weather_Class` label, the
SVM has a fairly strong signal to work with — accuracy is generally expected to be high, with
most misclassifications occurring near the 25°C decision boundary.

## ✅ Conclusion
This project built a Support Vector Machine (SVM) classifier with an RBF kernel to classify
hourly weather observations as Warm or Cool, using live data fetched directly from the
Open-Meteo API for temperature, relative humidity, surface pressure, and wind speed. Feature
scaling was essential for this task, since SVM (particularly with an RBF kernel) relies on
distance-based calculations to find the optimal separating boundary between classes — without
standardization, a feature like surface pressure (values near 1000) would numerically dominate
a feature like temperature (values near 20-40), even though both are equally meaningful to the
classification. One key advantage of SVM is its ability to model complex, non-linear decision
boundaries via the kernel trick, without needing to manually engineer polynomial or
interaction features. One key limitation of SVM is that it does not scale well to very large
datasets, since training time grows significantly with the number of samples, and it also
requires careful tuning of hyperparameters like the kernel type, `C` (regularization
strength), and `gamma` to achieve good performance.


## 📂 Files
- `Assignment-6.ipynb` — full notebook with code, live API integration, and visualizations
