"""
app.py

Flask REST API for the Loan Approval Checker project.

Serves:
  - the dashboard page (Step 2): GET /
  - the application form (Step 3): GET /apply
  - model info endpoints, read from model/model_metadata.json
  - a /api/predict endpoint that loads model/loan_svc_model.pkl and returns
    an approval decision for user-submitted applicant details.

Run `python train_model.py` once before starting this server, so that
model/loan_svc_model.pkl and model/model_metadata.json exist.
"""

import json
import logging
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "loan_svc_model.pkl"
METADATA_PATH = BASE_DIR / "model" / "model_metadata.json"
LOG_PATH = BASE_DIR / "app.log"

NUMERIC_FEATURES = ["Age", "Person Income", "Loan Amount", "Loan interest Rate", "Credit Score"]
CATEGORICAL_FEATURES = ["Home Onwership", "Previous Loan"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger("loan_app")

app = Flask(__name__)

# --- Load model + metadata once at startup ---
_model = None
_metadata = None
_load_error = None

try:
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        _metadata = json.load(f)
    logger.info("Model and metadata loaded successfully from %s", MODEL_PATH)
except FileNotFoundError as exc:
    _load_error = str(exc)
    logger.warning("Model not loaded: %s", exc)


def model_is_ready() -> bool:
    """Return True if both the trained pipeline and its metadata loaded successfully."""
    return _model is not None and _metadata is not None


# ---------- Pages ----------

@app.route("/")
def dashboard():
    """Render the Step 2 dashboard page (model info, metrics, samples, decision boundary)."""
    return render_template("dashboard.html")


@app.route("/apply")
def apply_form():
    """Render the Step 3 loan application form page."""
    return render_template("index.html")


# ---------- Model info API (Step 2) ----------

@app.route("/api/model/info")
def model_info():
    """Return algorithm, kernel, class weight, row counts, and support vector count."""
    if not model_is_ready():
        return jsonify({"error": "Model not found. Run train_model.py first."}), 404
    info = {
        "algorithm": _metadata["algorithm"],
        "kernel": _metadata["kernel"],
        "class_weight": _metadata["class_weight"],
        "model_file": _metadata["model_file"],
        "trained_rows": _metadata["trained_rows"],
        "test_rows": _metadata["test_rows"],
        "support_vectors_count": _metadata["support_vectors_count"],
        "target": _metadata["target"],
    }
    return jsonify(info)


@app.route("/api/model/features")
def model_features():
    """Return the original feature columns and their one-hot encoded expansion."""
    if not model_is_ready():
        return jsonify({"error": "Model not found. Run train_model.py first."}), 404
    return jsonify(
        {
            "original_columns_used": _metadata["original_columns_used"],
            "encoded_features": _metadata["features"],
        }
    )


@app.route("/api/model/samples")
def model_samples():
    """Return a handful of raw (pre-encoding) rows from the training data."""
    if not model_is_ready():
        return jsonify({"error": "Model not found. Run train_model.py first."}), 404
    return jsonify(_metadata["samples"])


@app.route("/api/model/metrics")
def model_metrics():
    """Return accuracy/precision/recall/F1 and the confusion matrix on the test set."""
    if not model_is_ready():
        return jsonify({"error": "Model not found. Run train_model.py first."}), 404
    return jsonify(
        {
            "metrics": _metadata["metrics"],
            "confusion_matrix": _metadata["confusion_matrix"],
        }
    )


@app.route("/api/model/decision_boundary")
def model_decision_boundary():
    """Return a 2D PCA projection of the test set, for the dashboard's scatter chart."""
    if not model_is_ready():
        return jsonify({"error": "Model not found. Run train_model.py first."}), 404
    return jsonify(_metadata["decision_boundary_projection"])


# ---------- Prediction API (Step 3) ----------

@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict loan approval for a single applicant submitted from the form.

    Expects a JSON body with Age, Person Income, Loan Amount, Loan interest
    Rate, Credit Score, Home Onwership, and Previous Loan. Encodes the input
    the same way training did, then returns approved/not approved.

    Note: Loan Status 1 = approved, 0 = rejected, per the source dataset's
    documented column description (Kaggle "Loan Approval Classification Data").
    """
    if not model_is_ready():
        logger.warning("Predict called but model is not ready.")
        return jsonify({"error": "Model not found. Run train_model.py first."}), 404

    payload = request.get_json(silent=True) or {}
    logger.info("Predict request received: %s", payload)

    # Validate required fields are present
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [field for field in required if field not in payload]
    if missing:
        logger.warning("Predict request missing fields: %s", missing)
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        row = {feat: float(payload[feat]) for feat in NUMERIC_FEATURES}
    except (TypeError, ValueError):
        logger.warning("Predict request had non-numeric values: %s", payload)
        return jsonify({"error": "Numeric fields must be valid numbers."}), 400

    for feat in CATEGORICAL_FEATURES:
        row[feat] = payload[feat]

    input_df = pd.DataFrame([row])

    # One-hot encode categoricals the same way training did, then align
    # columns to exactly what the pipeline was fit on (fills missing
    # dummy columns with 0, drops anything unexpected).
    numeric_part = input_df[NUMERIC_FEATURES]
    dummies_part = pd.get_dummies(input_df[CATEGORICAL_FEATURES])
    x = pd.concat([numeric_part, dummies_part], axis=1)
    x = x.reindex(columns=_metadata["features"], fill_value=0)

    prediction = _model.predict(x)[0]
    approved = bool(int(prediction) == 1)

    result = {"approved": approved, "raw_prediction": int(prediction)}

    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(x)[0]
        classes = list(_model.classes_)
        idx_approved = classes.index(1) if 1 in classes else None
        if idx_approved is not None:
            result["approval_probability"] = round(float(proba[idx_approved]), 4)

    logger.info("Predict result: approved=%s raw_prediction=%s", approved, int(prediction))
    return jsonify(result)


if __name__ == "__main__":
    if not model_is_ready():
        print(f"WARNING: model not loaded ({_load_error}). Run train_model.py first.")
    app.run(debug=True, port=5000)
