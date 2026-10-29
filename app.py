# import necessary libraries
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, roc_curve

from src.credit_utils import (
    FEATURE_COLS,
    METRIC_ORDER,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_FILES,
    REQUIRED_COLS,
    TARGET_COL,
    TEST_DATA_PATH,
    evaluate_pipeline,
    validate_upload,
)

# Streamlit app configuration
st.set_page_config(page_title="Credit Default Risk | Model Explorer", page_icon="🏦", layout="wide")

# Class labels for the target variable
CLASS_LABELS = ["Repaid (0)", "Default (1)"]

# Function to load a model from the MODEL_DIR based on the display name
@st.cache_resource(show_spinner=False)
def load_model(display_name: str):
    return joblib.load(MODEL_DIR / f"{MODEL_FILES[display_name]}.joblib")

# Function to load the training report (metrics) from METRICS_PATH
@st.cache_data(show_spinner=False)
def load_training_report() -> dict:
    with open(METRICS_PATH) as fh:
        return json.load(fh)

# Function to load the test dataset from TEST_DATA_PATH
@st.cache_data(show_spinner=False)
def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    import io

    return pd.read_csv(io.BytesIO(raw))

# Plotting
# Function to plot the confusion matrix using seaborn heatmap
def plot_confusion(cm, title: str):
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="crest", cbar=False, ax=ax,
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, linewidths=1, linecolor="white",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig

# Function to plot the ROC curve for multiple models
def plot_roc(curves: dict[str, tuple]):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for label, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig

# Sidebar controls
st.sidebar.header("⚙️ Controls")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES), index=0)

uploaded = st.sidebar.file_uploader("Upload test CSV", type="csv",
                                    help="Must contain all 20 feature columns plus `loan_default`.")
use_sample = st.sidebar.checkbox("Use bundled test_data.csv", value=uploaded is None,
                                 disabled=uploaded is not None)

if TEST_DATA_PATH.exists():
    st.sidebar.download_button("⬇️ Download sample test_data.csv", TEST_DATA_PATH.read_bytes(),
                               file_name="test_data.csv", mime="text/csv")

with st.sidebar.expander("Expected columns"):
    st.code(", ".join(REQUIRED_COLS), language="text")

# Main app
st.title("🏦 Credit Default Risk — Classifier Explorer")
st.caption("UCI Statlog (German Credit): predict whether a loan applicant will default. "
           "Five pre-trained scikit-learn pipelines are evaluated on the data you upload — "
           "no retraining happens here.")

# Load the test data from either the uploaded file or the bundled sample
df = None
if uploaded is not None:
    df = read_csv_bytes(uploaded.getvalue())
    source = f"uploaded file `{uploaded.name}`"
elif use_sample and TEST_DATA_PATH.exists():
    df = pd.read_csv(TEST_DATA_PATH)
    source = "bundled `test_data.csv`"

tab_eval, tab_compare, tab_about = st.tabs(["📊 Evaluate selected model", "🏆 Compare all models", "ℹ️ About"])

if df is None:
    with tab_eval:
        st.info("Upload a test CSV in the sidebar (or tick *Use bundled test_data.csv*) to begin.")
else:
    problems = validate_upload(df)
    if problems:
        with tab_eval:
            for p in problems:
                st.error(p)
            st.stop()

    X, y = df[FEATURE_COLS], df[TARGET_COL].astype(int)

    # Evaluate the selected model on the uploaded or sample data
    with tab_eval:
        st.subheader(f"{model_name}")
        st.write(f"Scored **{len(df):,} rows** from {source} — "
                 f"default rate in this file: **{y.mean():.1%}**.")
        with st.expander("Preview uploaded data"):
            st.dataframe(df.head(20), width="stretch")

        model = load_model(model_name)
        metrics, y_pred, y_score, cm = evaluate_pipeline(model, X, y)

        cols = st.columns(6)
        for col, key in zip(cols, METRIC_ORDER):
            col.metric(key, f"{metrics[key]:.3f}")

        left, right = st.columns(2)
        with left:
            st.markdown("**Confusion matrix**")
            st.pyplot(plot_confusion(cm, model_name), clear_figure=True)
        with right:
            st.markdown("**ROC curve**")
            fpr, tpr, _ = roc_curve(y, y_score)
            st.pyplot(plot_roc({model_name: (fpr, tpr, metrics["AUC"])}), clear_figure=True)

        st.markdown("**Classification report**")
        rep = classification_report(y, y_pred, target_names=CLASS_LABELS, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")

        tn, fp, fn, tp = cm.ravel()
        st.caption(f"Reading the matrix: {tp} defaulters caught, {fn} missed (false negatives), "
                   f"{fp} good customers wrongly flagged, {tn} good customers correctly approved.")

        out = df.copy()
        out["predicted_default"] = y_pred
        out["default_probability"] = y_score.round(4)
        st.download_button("⬇️ Download predictions", out.to_csv(index=False).encode(),
                           file_name=f"predictions_{MODEL_FILES[model_name]}.csv", mime="text/csv")

    # Compare all models on the same data
    with tab_compare:
        st.subheader("All five models on the same data")
        rows, curves = [], {}
        for name in MODEL_FILES:
            m, _, score, _ = evaluate_pipeline(load_model(name), X, y)
            rows.append({"ML Model Name": name, **m})
            fpr, tpr, _ = roc_curve(y, score)
            curves[name] = (fpr, tpr, m["AUC"])
        table = pd.DataFrame(rows).set_index("ML Model Name")[METRIC_ORDER]
        st.dataframe(
            table.style.format("{:.3f}").highlight_max(axis=0, color="#cfe8cf"),
            width="stretch",
        )
        st.caption("Green = best value per metric.")
        st.pyplot(plot_roc(curves), clear_figure=True)

# About the Project
with tab_about:
    st.markdown(
        """
**Task** – binary classification: will a loan applicant default (`loan_default = 1`)?

**Data** – UCI Statlog (German Credit Data): 1,000 applicants, 20 features
(7 numeric, 13 categorical). Positive class (default) is 30 % of rows.
The bundled `test_data.csv` is the stratified 20 % hold-out (200 rows) that none of
the models saw during training or hyper-parameter search.

**Pipelines** – every model file contains the full preprocessing
(standard scaling + one-hot encoding) followed by the classifier, so any CSV with the
raw columns can be scored directly.
        """
    )
    st.markdown("**Tuned hyper-parameters & cross-validated AUC (training split only)**")
    rep = load_training_report()
    st.dataframe(
        pd.DataFrame(
            {
                "Best parameters": {k: str(v["best_params"]) for k, v in rep.items()},
                "5-fold CV AUC": {k: v["cv_auc_train"] for k, v in rep.items()},
            }
        ),
        width="stretch",
    )
