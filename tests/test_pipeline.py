# import necessary libraries
import joblib
import pandas as pd

from src.credit_utils import (FEATURE_COLS, MODEL_DIR, MODEL_FILES, TARGET_COL,
                              TEST_DATA_PATH, evaluate_pipeline, validate_upload)

# Test functions to validate the pipeline and models
def test_test_data_is_valid():
    df = pd.read_csv(TEST_DATA_PATH)
    assert len(df) >= 100 and len(FEATURE_COLS) >= 12
    assert validate_upload(df) == []

# Test function to ensure all saved models score above 0.5 AUC without retraining
def test_all_saved_models_score_without_retraining():
    df = pd.read_csv(TEST_DATA_PATH)
    for name, stem in MODEL_FILES.items():
        model = joblib.load(MODEL_DIR / f"{stem}.joblib")
        metrics, *_ = evaluate_pipeline(model, df[FEATURE_COLS], df[TARGET_COL])
        assert 0.5 < metrics["AUC"] <= 1.0, name

# Test function to check if validation catches missing columns in the test data
def test_validation_catches_missing_column():
    df = pd.read_csv(TEST_DATA_PATH).drop(columns=["age"])
    assert any("age" in p for p in validate_upload(df))

# Main execution block
if __name__ == "__main__":
    test_test_data_is_valid(); test_all_saved_models_score_without_retraining()
    test_validation_catches_missing_column(); print("all tests passed")
