# Credit Default Risk Classifier (German Credit, 5 ML models + Streamlit app)

An end-to-end machine-learning project: five classifiers trained on the UCI Statlog
*German Credit* data, evaluated with six metrics, and served through an interactive
Streamlit app that scores an uploaded test CSV **without retraining**.

---

## 1. Problem statement

A lender must decide whether to approve a loan. Given an applicant's financial history and
personal circumstances, predict whether the applicant will **default** (`loan_default = 1`)
or **repay** (`0`). This is a binary classification problem with an imbalanced target
(30 % defaults). Missing a defaulter is usually costlier than rejecting a good customer, so
besides accuracy the project reports AUC, precision, recall, F1 and MCC, which describe
performance on the minority (default) class far better than accuracy alone.

## 2. Dataset description

| Property | Value |
| --- | --- |
| Source | UCI Machine Learning Repository - *Statlog (German Credit Data)* (Prof. Hans Hofmann, Univ. Hamburg). The CSV in `data/german_credit_raw.csv` is a labelled-column copy of the UCI file hosted in the public `stedy/Machine-Learning-with-R-datasets` GitHub repo. |
| Instances | 1,000 (minimum required: 500) |
| Features | 20 (minimum required: 12): 7 numeric + 13 categorical |
| Target | `loan_default` - 1 = bad credit / default (300 rows, 30 %), 0 = good credit (700 rows). Re-coded from the original 1/2 labels. |
| Missing values | None |
| Numeric features | `months_loan_duration`, `amount`, `installment_rate`, `residence_history`, `age`, `existing_credits`, `dependents` |
| Categorical features | `checking_balance`, `credit_history`, `purpose`, `savings_balance`, `employment_length`, `personal_status`, `other_debtors`, `property`, `installment_plan`, `housing`, `telephone`, `foreign_worker`, `job` |

**Methodology**

- Stratified **80 / 20 split** (`random_state=2026`): 800 training rows, 200 hold-out rows. The hold-out is saved as `test_data.csv` (140 repaid / 60 default) and is the file to upload in the app.
- Every model is a single scikit-learn `Pipeline`: `ColumnTransformer` (standard-scale numeric columns, one-hot encode categoricals, `handle_unknown="ignore"`) followed by the classifier. Preprocessing therefore travels inside each `.joblib` file.
- Hyper-parameters tuned with `GridSearchCV` (5-fold stratified CV, scoring = ROC-AUC) on the **training split only**. The test split is used once, for the final report.
- Positive class for precision / recall / F1 / MCC is **default (1)**.

## 3. Necessary links

- **GitHub repository:** `https://github.com/RVRahulKrishna/credit-risk-classifier`
- **Live Streamlit app:** `https://credit-risk-ml-classifier.streamlit.app/`

**Repository layout**

```
credit-risk-classifier/
|-- app.py                  # Streamlit application
|-- requirements.txt
|-- README.md
|-- test_data.csv           # 200-row hold-out (features + loan_default) for the app
|-- data/german_credit_raw.csv
|-- model/                  # saved pipelines + metrics.json
|   |-- logistic_regression.joblib  decision_tree.joblib  knn.joblib
|   |-- naive_bayes.joblib          svm.joblib             metrics.json
|-- src/
|   |-- credit_utils.py     # schema, paths, metric helpers (shared by training + app)
|   |-- train_models.py     # data prep, split, tuning, evaluation, model saving
|-- tests/test_pipeline.py  # smoke tests
```

## 4. Models used

All five models were trained on the same split and evaluated on the same 200-row hold-out set.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.785 | 0.815 | 0.673 | 0.550 | 0.606 | 0.464 |
| Decision Tree | 0.690 | 0.751 | 0.489 | 0.767 | 0.597 | 0.389 |
| KNN | 0.725 | 0.784 | 0.692 | 0.150 | 0.247 | 0.226 |
| Naive Bayes | 0.735 | 0.780 | 0.552 | 0.617 | 0.583 | 0.391 |
| Support Vector Machine | 0.710 | 0.826 | 0.510 | 0.817 | 0.628 | 0.441 |

Selected hyper-parameters (from CV on the training split):

| Model | Best parameters | CV AUC (train) |
| --- | --- | --- |
| Logistic Regression | `{'C': 0.1, 'class_weight': None}` | 0.779 |
| Decision Tree | `{'class_weight': 'balanced', 'max_depth': 8, 'min_samples_leaf': 20}` | 0.713 |
| KNN | `{'n_neighbors': 35, 'p': 1, 'weights': 'distance'}` | 0.758 |
| Naive Bayes | `{'var_smoothing': 0.1}` | 0.786 |
| Support Vector Machine | `{'C': 1, 'class_weight': 'balanced', 'gamma': 0.01}` | 0.782 |

## 5. Observations on model performance

Confusion-matrix counts on the hold-out set (positive = default; 60 defaulters, 140 good):

| Model | TP | FN | FP | TN |
| --- | --- | --- | --- | --- |
| Logistic Regression | 33 | 27 | 16 | 124 |
| Decision Tree | 46 | 14 | 48 | 92 |
| KNN | 9 | 51 | 4 | 136 |
| Naive Bayes | 37 | 23 | 30 | 110 |
| Support Vector Machine | 49 | 11 | 47 | 93 |

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Highest accuracy (0.785), precision (0.673) and MCC (0.464), with the second-best AUC (0.815). It catches 33 of 60 defaulters and wrongly flags only 16 good customers, giving the most balanced error profile. Its CV AUC (0.779) is close to its test AUC, so it generalises well; strong regularisation (C = 0.1) suited the many one-hot columns. The default 0.5 threshold is conservative for the minority class (recall 0.55). |
| Decision Tree | Weakest ranker (AUC 0.751, CV AUC 0.713) and lowest accuracy (0.690) of the five. `class_weight="balanced"` pushed recall up to 0.767 (46/60 caught), but at the cost of 48 false alarms (precision 0.489). A single tree is high-variance on 800 rows; its value is interpretability (a loan officer can read the rules), not raw accuracy. |
| KNN | Looks acceptable on accuracy (0.725) but that is misleading: recall is only 0.150 (9 of 60 defaulters found) and MCC 0.226, the worst of all models. With 20 features (61 after one-hot encoding) distance becomes less informative and the 70 / 30 imbalance lets the majority class dominate each neighbourhood (best k = 35, the edge of the searched grid). It is the model that would approve the most bad loans. |
| Naive Bayes | Fast and competitive in cross-validation (best CV AUC 0.786) but weaker on the hold-out (AUC 0.780, MCC 0.391). The independence assumption is violated by correlated financial attributes and by one-hot columns being treated as Gaussians, so probabilities are poorly calibrated; recall (0.617) is better than Logistic Regression at the price of more false positives (30). |
| Support Vector Machine | Best AUC (0.826) and best F1 (0.628), and the highest recall of all (0.817, 49 of 60 defaulters caught) thanks to `class_weight="balanced"`. The price is lower precision (0.510) and accuracy (0.710) because 47 good customers are flagged. Best choice when missing a defaulter is much costlier than rejecting a good applicant. Slowest to tune. |
| **Overall, Winner for your dataset?** | **Logistic Regression** is the best all-round model: top accuracy, precision and MCC (the metric that is most robust to class imbalance) and second-best AUC, while remaining simple, fast and interpretable. **SVM** is the runner-up and the better pick if recall on defaulters is the priority (best AUC, F1 and recall). **KNN** is clearly the worst fit. Caveat: the hold-out has only 200 rows (60 positives), so differences of about 0.01-0.02 AUC between LR, SVM, NB and KNN are within sampling noise. |

## 6. How to run locally

```bash
git clone https://github.com/RVRahulKrishna/credit-risk-classifier.git
cd credit-risk-classifier

# for windows
python -m venv .venv && source .venv/Scripts/activate
python -m pip install -r requirements.txt

# (optional) retrain everything - reproducible, regenerates model/ and test_data.csv
python -m src.train_models

# (optional) smoke tests
python -m tests.test_pipeline

# launch the streamlit app
python -m streamlit run app.py
```

Then upload `test_data.csv` in the sidebar (or tick *Use bundled test_data.csv*).
