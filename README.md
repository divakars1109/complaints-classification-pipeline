# Complaints Dispute Prediction

A triage model for consumer complaints that catches **71% of complaints that go on to become formal disputes**, letting a support/compliance team route high-risk cases to specialist review before they escalate.

## Problem & Business Context

Meridian Financial Group (a fictional lender used for this assignment) is seeing a sustained rise in customer complaints across mortgages, credit cards, loans, and credit reporting. A subset of these complaints escalate into formal disputes with the CFPB, which carry regulatory, reputational, and customer-churn risk. This project uses the CFPB Consumer Complaint Database (~321K records) to build a model that flags which incoming complaints are most likely to escalate, so limited review capacity can be pointed at the highest-risk cases first.

## Key Results

Two models were trained and compared on a held-out test set, using F1 on the disputed class as the primary metric (accuracy is misleading here — the target is ~80/20 imbalanced) and ROC-AUC as a secondary ranking metric.

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Disputed F1 | 0.36 | 0.36 |
| Disputed Recall | 0.66 | **0.71** |
| Disputed Precision | 0.25 | 0.25 |
| ROC-AUC | 0.6093 | **0.6123** |
| True Positives (disputes caught) | 8,388 | **9,025** |
| False Negatives (missed disputes) | 4,411 | **3,774** |

**Random Forest was selected for deployment.** Both models tie on F1, but Random Forest catches 637 more true disputes at the same precision, with better recall and ROC-AUC — the better trade-off when missed disputes (false negatives) are costlier than extra manual reviews (false positives).

Feature importance shows `Company response to consumer` as the single strongest driver of escalation, followed by submission channel (Web complaints escalate more than assisted channels like Phone) and response timing (`days_to_respond`, `Timely response?`).

## How It Works

The pipeline follows five stages: **EDA** quantifies escalation patterns across products, channels, and states to ground the modeling in real business risk; **feature engineering** parses dates into a `days_to_respond` field, drops leakage-prone columns, and backfills missing required fields; **model training** fits both a Logistic Regression baseline and a Random Forest on preprocessed mixed-type data, with hyperparameter tuning done on a stratified subsample for speed before refitting on the full training set; **evaluation** compares both models on F1, recall, precision, and ROC-AUC for the disputed class; and **deployment packaging** serializes the winning pipeline (preprocessing + classifier together) so it can be reloaded and run on new data without retraining.

## Tech Stack

`numpy` `pandas` `scikit-learn` `joblib` — versions pinned in [68206_requirements.txt](68206_requirements.txt).

## Repo Structure

```
├── 68206_Complaints_Notebook.ipynb    # EDA, feature engineering, training, evaluation, insights (Q1-Q5)
├── 68206_Validation_Notebook.ipynb    # Reloads the saved pipeline and scores it on external data
├── feature_engineering.py             # Shared preprocessing used by both training and inference
├── 68206_Pipeline.pkl                 # Trained Random Forest pipeline (preprocessing + model)
├── 68206_requirements.txt             # Pinned dependencies
├── complaints_modeltesting100.csv     # 100-row sample for smoke-testing the saved pipeline
└── README.md
```

`complaints_training.csv` (122MB, the full CFPB extract used for training) is excluded from this repo — see below.

## How to Run It

Install dependencies:

```bash
pip install -r 68206_requirements.txt
```

Reload the saved pipeline and score new data, following the same approach as `68206_Validation_Notebook.ipynb`:

```python
import pickle
import pandas as pd
from feature_engineering import feature_engineering

with open("68206_Pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

df = pd.read_csv("complaints_modeltesting100.csv")
X = feature_engineering(df)
predictions = pipeline.predict(X)  # 1 = predicted to escalate to a dispute
```

To reproduce training from scratch, run `68206_Complaints_Notebook.ipynb` end to end. It expects a CFPB-schema extract at `complaints_training.csv` in the project root (not included here — see [Training data](#training-data)).

### Training data

`complaints_training.csv` exceeds GitHub's 100MB file size limit and isn't included. Obtain a comparable extract from the [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) and place it in the project root before re-running the training notebook.

## Limitations

This model is intentionally honest about being weak in absolute terms: **F1 ≈ 0.36 and precision ≈ 0.25** on the disputed class. Roughly 3 in 4 complaints it flags as high-risk will *not* actually end up disputed. This is expected given the feature set is purely categorical/metadata (product, channel, response type, state, timing) with **no NLP on the complaint narrative text**, which likely carries most of the remaining predictive signal.

Given this, the model should be used as a **triage and prioritization tool**, not an auto-decision system: it helps rank which complaints deserve earlier specialist attention, but every flagged case still needs human review, and it should never be used to automatically deny, delay, or deprioritize a complaint response.
