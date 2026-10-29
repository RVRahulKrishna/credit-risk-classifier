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