import pandas as pd
import numpy as np


def feature_engineering(X):
    """Prepare complaints data for training/inference with the saved pipeline."""
    X = X.copy()

    received_col = "Date received"
    sent_col = "Date sent to company"

    if received_col in X.columns:
        X[received_col] = pd.to_datetime(X[received_col], errors="coerce")
    if sent_col in X.columns:
        X[sent_col] = pd.to_datetime(X[sent_col], errors="coerce")

    if received_col in X.columns and sent_col in X.columns:
        X["days_to_respond"] = (X[sent_col] - X[received_col]).dt.days
    elif "days_to_respond" not in X.columns:
        X["days_to_respond"] = np.nan

    # Drop raw target to avoid leakage if present.
    X = X.drop(columns=["Consumer disputed?"], errors="ignore")

    # Drop raw date columns after feature creation.
    X = X.drop(columns=[received_col, sent_col], errors="ignore")

    # Ensure model-required columns exist even when missing at inference time.
    required_columns = [
        "Product",
        "Submitted via",
        "Company response to consumer",
        "Company public response",
        "Timely response?",
        "Tags",
        "Consumer consent provided?",
        "State",
        "days_to_respond",
    ]
    for column in required_columns:
        if column not in X.columns:
            X[column] = np.nan

    return X
