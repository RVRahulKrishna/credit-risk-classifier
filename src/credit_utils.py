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