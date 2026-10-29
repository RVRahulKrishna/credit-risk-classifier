# import necessary libraries
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Constant paths for feature names and paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "model"
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "german_credit_raw.csv"
TEST_DATA_PATH = ROOT_DIR / "test_data.csv"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Constants for feature names
TARGET_COL = "loan_default"  # 1 = applicant defaulted (bad credit), 0 = repaid

# Constants for numeric features
NUMERIC_FEATURES = [
    "months_loan_duration",
    "amount",
    "installment_rate",
    "residence_history",
    "age",
    "existing_credits",
    "dependents",
]

# Constants for categorical features and required columns
CATEGORICAL_FEATURES = [
    "checking_balance",
    "credit_history",
    "purpose",
    "savings_balance",
    "employment_length",
    "personal_status",
    "other_debtors",
    "property",
    "installment_plan",
    "housing",
    "telephone",
    "foreign_worker",
    "job",
]

# Constants for all feature columns and required columns
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLS = FEATURE_COLS + [TARGET_COL]

# Display name file stem of the saved pipeline in model/
MODEL_FILES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}

# Constants for metric order
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

def compute_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """Return the six assignment metrics (positive class = 1 = default)."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

def positive_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Probability of the positive class (falls back to decision_function)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)

def evaluate_pipeline(model, X: pd.DataFrame, y) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Score a *fitted* pipeline on (X, y) without refitting anything."""
    y_pred = model.predict(X)
    y_score = positive_scores(model, X)
    return compute_metrics(y, y_pred, y_score), y_pred, y_score, confusion_matrix(y, y_pred)