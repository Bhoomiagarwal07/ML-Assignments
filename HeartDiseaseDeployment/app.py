"""
Flask REST API: Heart Disease Risk Predictor
-----------------------------------------------
Loads a pre-trained Random Forest model and serves predictions two ways:
  1. A simple HTML form at "/" for humans to use in a browser
  2. A JSON REST API endpoint at "/predict" for programmatic access
"""

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model + the exact feature column order it was trained on.
# Loaded once at startup (not per-request) so predictions are fast.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
feature_columns = model_bundle["feature_columns"]


def build_feature_row(data):
    """
    Converts raw input (dict) into a single-row DataFrame that exactly matches
    the feature columns/order the model was trained on.
    """
    row = {col: data[col] for col in feature_columns}
    return pd.DataFrame([row])[feature_columns]


@app.route("/", methods=["GET"])
def home():
    """Renders the HTML form for browser-based use."""
    return render_template("index.html", prediction=None, error=None)


@app.route("/", methods=["POST"])
def predict_form():
    """Handles the HTML form submission and renders the result back on the page."""
    try:
        data = {col: float(request.form[col]) for col in feature_columns}
        features = build_feature_row(data)
        pred = int(model.predict(features)[0])
        prediction = "Heart Disease Detected" if pred == 1 else "No Heart Disease Detected"
        return render_template("index.html", prediction=prediction, error=None)
    except Exception as e:
        return render_template("index.html", prediction=None, error=str(e))


@app.route("/predict", methods=["POST"])
def predict_api():
    """
    JSON REST API endpoint.

    Example request body:
    {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }

    Example response:
    {
        "prediction": "Heart Disease Detected"
    }
    """
    try:
        data = request.get_json(force=True)
        features = build_feature_row(data)
        pred = int(model.predict(features)[0])
        result = "Heart Disease Detected" if pred == 1 else "No Heart Disease Detected"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    """Simple health check endpoint — useful for Render's health monitoring."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # For local testing only. Render uses gunicorn (see Procfile) in production.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
