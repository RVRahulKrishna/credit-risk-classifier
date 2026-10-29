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

# Dictionary that returns in-scope models and their hyperparameter grids
def model_zoo() -> dict[str, tuple[Pipeline, dict]]:
    """name -> (pipeline, hyper-parameter grid)."""
    return {
        "Logistic Regression": (
            Pipeline(
                [
                    ("prep", make_preprocessor()),
                    ("clf", LogisticRegression(max_iter=5000, random_state=SEED)),
                ]
            ),
            {
                "clf__C": [0.01, 0.1, 1, 10],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "Decision Tree": (
            Pipeline(
                [
                    ("prep", make_preprocessor(scale_numeric=False)),
                    ("clf", DecisionTreeClassifier(random_state=SEED)),
                ]
            ),
            {
                "clf__max_depth": [3, 4, 5, 6, 8],
                "clf__min_samples_leaf": [5, 10, 20],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "KNN": (
            Pipeline(
                [
                    ("prep", make_preprocessor()),
                    ("clf", KNeighborsClassifier()),
                ]
            ),
            {
                "clf__n_neighbors": [5, 9, 15, 25, 35],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            },
        ),
        "Naive Bayes": (
            Pipeline(
                [
                    ("prep", make_preprocessor()),
                    ("clf", GaussianNB()),
                ]
            ),
            {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-2, 1e-1]},
        ),
        "Support Vector Machine": (
            Pipeline(
                [
                    ("prep", make_preprocessor()),
                    (
                        "clf",
                        SVC(kernel="rbf", probability=True, random_state=SEED),
                    ),
                ]
            ),
            {
                "clf__C": [0.5, 1, 5, 10],
                "clf__gamma": ["scale", 0.01, 0.005],
                "clf__class_weight": [None, "balanced"],
            },
        ),
    }

# Main function to load data, train models, and save metrics
def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_and_prepare()
    print(f"Loaded {df.shape[0]} rows x {len(FEATURE_COLS)} features | "
          f"default rate = {df[TARGET_COL].mean():.1%}")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET_COL], random_state=SEED
    )
    test_df.to_csv(TEST_DATA_PATH, index=False)
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows -> {TEST_DATA_PATH.name}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    report: dict[str, dict] = {}

    for name, (pipe, grid) in model_zoo().items():
        t0 = time.time()
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        metrics, *_ = evaluate_pipeline(best, X_test, y_test)
        joblib.dump(best, MODEL_DIR / f"{MODEL_FILES[name]}.joblib", compress=3)

        report[name] = {
            "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_auc_train": round(float(search.best_score_), 4),
        }
        print(f"[{name:<22}] cv-AUC={search.best_score_:.3f} | "
              + " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
              + f" | {time.time() - t0:.1f}s")

    with open(METRICS_PATH, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nSaved models to {MODEL_DIR}/ and metrics to {METRICS_PATH.name}")

# Main function call
if __name__ == "__main__":
    main()
