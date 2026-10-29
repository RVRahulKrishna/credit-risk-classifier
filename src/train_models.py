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