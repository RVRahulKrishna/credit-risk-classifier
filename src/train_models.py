# import necessary libraries
from __future__ import annotations

import json
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.credit_utils import (
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_FILES,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
    TARGET_COL,
    TEST_DATA_PATH,
    evaluate_pipeline,
)

SEED = 2026
TEST_SIZE = 0.20

# Load and prepare the data
def load_and_prepare() -> pd.DataFrame:
    raw = pd.read_csv(RAW_DATA_PATH)
    df = raw.rename(columns={"default": TARGET_COL}).copy()
    # Original coding: 1 = good credit, 2 = bad credit  ->  0 / 1 (1 = default)
    df[TARGET_COL] = (df[TARGET_COL] == 2).astype(int)
    # Keep only the schema columns, in a fixed order.
    return df[FEATURE_COLS + [TARGET_COL]]

# Preprocessor function to scale numeric features and one-hot encode categorical features
def make_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    num_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", num_step, NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

if __name__ == "__main__":
    main()
