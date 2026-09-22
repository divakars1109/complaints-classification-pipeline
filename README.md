# Complaints project

Machine learning pipeline that predicts whether a consumer disputes a company's
response to a complaint, built on the CFPB Consumer Complaint Database schema
(fields include `Product`, `Submitted via`, `Company response to consumer`,
`Timely response?`, `State`, etc.).

## Contents

- `68206_Complaints_Notebook.ipynb` — main analysis, feature engineering, and model training notebook.
- `68206_Validation_Notebook.ipynb` — loads the saved pipeline and validates it against a held-out sample.
- `feature_engineering.py` — shared preprocessing function (date parsing, `days_to_respond` feature, leakage-column removal) used by both the training pipeline and inference.
- `68206_Pipeline.pkl` — trained scikit-learn pipeline (joblib-serialized).
- `complaints_modeltesting100.csv` — small sample of records for smoke-testing the pipeline.
- `68206_requirements.txt` — pinned dependencies.

## Setup

```bash
pip install -r 68206_requirements.txt
```

## Training data

`complaints_training.csv` is not included in this repository (122MB, exceeds
GitHub's file size limit). It follows the CFPB Consumer Complaint Database
schema — obtain a comparable extract and place it in the project root before
re-running the training notebook.

## Usage

Run `68206_Complaints_Notebook.ipynb` to reproduce training, or load the saved
pipeline directly:

```python
import joblib
import pandas as pd
from feature_engineering import feature_engineering

pipeline = joblib.load("68206_Pipeline.pkl")
X = feature_engineering(pd.read_csv("complaints_modeltesting100.csv"))
predictions = pipeline.predict(X)
```
