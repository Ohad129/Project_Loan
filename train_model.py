"""
train_model.py

Trains the loan approval SVC pipeline (StandardScaler + SVC) on p_data.csv,
evaluates it, and saves two artifacts:

  model/loan_svc_model.pkl   -> the fitted sklearn Pipeline (scaler + classifier)
  model/model_metadata.json  -> everything the dashboard / API need to display:
                                 feature list, sample rows, metrics, confusion
                                 matrix, and a 2D PCA projection of the decision
                                 boundary (for the "SVM Decision Boundary" chart
                                 shown in the assignment PDF).

Run this once before starting app.py. Re-run any time p_data.csv changes.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "p_data.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "loan_svc_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

NUMERIC_FEATURES = [
    "Age",
    "Person Income",
    "Loan Amount",
    "Loan interest Rate",
    "Credit Score",
]
CATEGORICAL_FEATURES = ["Home Onwership", "Previous Loan"]
TARGET = "Loan Status"


def build_features(df: pd.DataFrame):
    """Split df into X (7 original columns, one-hot expanded) and y."""
    numeric_x = df[NUMERIC_FEATURES]
    dummies_x = pd.get_dummies(df[CATEGORICAL_FEATURES])
    x = pd.concat([numeric_x, dummies_x], axis=1)
    y = df[TARGET]
    return x, y


def main():
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    x, y = build_features(df)
    feature_columns = list(x.columns)  # exact column order the model expects

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", class_weight="balanced", probability=True)),
        ]
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()
    print("Metrics:", metrics)
    print("Confusion matrix:", cm)

    # --- Save the fitted pipeline ---
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Saved trained pipeline to {MODEL_PATH}")

    # --- Sample rows for the dashboard table (raw, pre-encoding, easier to read) ---
    display_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    samples = df[display_cols].head(6).to_dict(orient="records")

    # --- 2D PCA projection of the scaled test set, for the decision-boundary chart ---
    scaler = pipeline.named_steps["scaler"]
    x_test_scaled = scaler.transform(x_test)
    pca = PCA(n_components=2, random_state=42)
    x_test_2d = pca.fit_transform(x_test_scaled)
    projection = [
        {
            "x": round(float(px), 3),
            "y": round(float(py), 3),
            "label": int(label),
        }
        for (px, py), label in zip(x_test_2d, y_test.to_numpy())
    ]
    # keep the payload light for the browser
    if len(projection) > 400:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(projection), size=400, replace=False)
        projection = [projection[i] for i in idx]

    metadata = {
        "algorithm": "SVC (Support Vector Classifier)",
        "kernel": pipeline.named_steps["classifier"].kernel,
        "class_weight": "balanced",
        "model_file": MODEL_PATH.name,
        "trained_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "support_vectors_count": int(pipeline.named_steps["classifier"].support_vectors_.shape[0]),
        "features": feature_columns,
        "original_columns_used": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "target": TARGET,
        "metrics": metrics,
        "confusion_matrix": {
            # Loan Status: 1 = approved, 0 = rejected (per source dataset documentation)
            "labels": ["Rejected (0)", "Approved (1)"],
            "matrix": cm,
        },
        "samples": samples,
        "decision_boundary_projection": projection,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
