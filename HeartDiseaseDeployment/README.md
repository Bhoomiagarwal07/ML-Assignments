# HeartDiseaseDeployment — End-to-End ML Model Deployment (Flask + Render)

## 📌 Objective
Build a machine learning model that predicts whether a patient is at risk of heart disease
based on clinical parameters, expose it via a Flask REST API, and deploy it as a live,
publicly accessible web service using Render.

## 🔗 Live Demo
**Render URL:** _[fill in after deploying — see deployment steps below]_

Example request:
```bash
curl -X POST https://your-app-name.onrender.com/predict \\
  -H "Content-Type: application/json" \\
  -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'
```
Example response:
```json
{"prediction": "Heart Disease Detected"}
```

⚠️ Render's free tier spins down after inactivity — the first request after idle time may
take 30-60 seconds to "wake up" the service.

## 📊 Dataset
**Heart Disease Prediction Dataset** (303 rows, 14 columns — the classic Cleveland Heart
Disease dataset)
Source: [Kaggle — johnsmith88/heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

`heart.csv` is included directly in this repo (a small, standard benchmark dataset commonly
redistributed for educational use).

## 🛠️ Libraries Used
- `pandas` — data loading and manipulation
- `scikit-learn` — Random Forest classifier, train/test split, evaluation
- `joblib` — model serialization
- `flask` — REST API and web form
- `gunicorn` — production WSGI server (used by Render)

## 🔍 Methodology

### Task 1: Data Understanding and Preprocessing
- Loaded `heart.csv` using Pandas, displayed the first 5 records
- Identified 13 numerical clinical features (age, sex, cp, trestbps, chol, fbs, restecg,
  thalach, exang, oldpeak, slope, ca, thal) and the target variable (`target`: 1 = heart
  disease, 0 = no heart disease)
- Confirmed zero missing values across all columns
- Split into 80% training (242 rows) / 20% testing (61 rows), stratified by target

### Task 2: Model Development
- Trained a `RandomForestClassifier` (200 estimators) on the training set
- Achieved a test accuracy of **81.97%**
- Saved the trained model (plus its expected feature column order) using `joblib` → `model.pkl`

### Task 3: API Development
- Built a Flask REST API (`app.py`) that:
  - Loads the trained model at startup
  - Accepts patient clinical details as JSON via `POST /predict`
  - Returns a JSON response in the exact required format: `{"prediction": "Heart Disease Detected"}`
    or `{"prediction": "No Heart Disease Detected"}`
  - Also includes a simple HTML form at `/` for browser-based testing, and a `/health`
    endpoint for uptime monitoring

### Task 4: GitHub and Cloud Deployment
- Published the complete source code, trained model, and deployment files to a public GitHub repository
- Deployed the Flask application on Render (see steps below)

## 🚀 How to Deploy on Render

1. Push this folder to GitHub (see repository structure below)
2. Sign up at [render.com](https://render.com)
3. **New + → Web Service** → connect your GitHub repo
4. If this project lives in a subfolder of a larger repo, set **Root Directory** to
   `HeartDiseaseDeployment`
5. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** Free
6. Click **Create Web Service** and wait for the build to finish
7. Copy your live URL and paste it into this README (replacing the placeholder above)

**Note on build speed:** `requirements.txt` intentionally does not pin exact package
versions, and a `runtime.txt` specifies Python 3.11.9 — this ensures Render installs
prebuilt wheels rather than compiling packages like scikit-learn from source, which can
otherwise cause builds to take 15+ minutes or hang entirely.

## 📂 Repository Structure
```
HeartDiseaseDeployment/
├── app.py                 # Flask REST API
├── model.pkl              # Trained Random Forest model
├── requirements.txt        # Python dependencies (unpinned for fast Render builds)
├── runtime.txt              # Pins Python version for wheel compatibility
├── Procfile                  # Tells Render how to start the app
├── README.md
├── train_model.py           # Reproducible training script
├── heart.csv                 # Dataset
├── templates/
│   └── index.html            # Optional HTML form
└── static/
    └── style.css               # Form styling
```

## 📈 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 81.97% |

## ✅ Conclusion
This project trained a Random Forest classifier to predict heart disease risk from 13
clinical parameters, achieving a test accuracy of 81.97% on the classic Cleveland Heart
Disease dataset. The model was serialized with Joblib and served through a Flask REST API
that accepts patient data as JSON and returns a clear, human-readable prediction. The main
challenge during deployment was ensuring dependency compatibility between the local
development environment and Render's build environment — pinning exact package versions
initially caused Render to fall back to compiling scikit-learn from source, dramatically
slowing the build; switching to unpinned dependencies alongside an explicit Python runtime
version resolved this by allowing pip to install prebuilt wheels instead. This project
highlights why MLOps practices matter in real machine learning work: a model's value is
realized only once it can be reliably reproduced, packaged, and served in a live environment,
and issues like dependency management, environment consistency, and deployment automation are
just as critical to a model's real-world success as the accuracy of the model itself.

## 📝 Submission Details
- **GitHub Repository:** _[fill in your repo URL]_
- **Render Deployment URL:** _[fill in your live URL]_
