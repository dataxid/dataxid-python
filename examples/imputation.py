"""
DataXID SDK — Imputation

Demonstrates Model.impute(): replace NULL cells in a DataFrame with model
predictions while preserving every non-NULL value.

1. Single trial                 — fast default
2. Multiple trials + pick       — more stable cells via aggregation

Requirements:
    pip install dataxid scikit-learn

Usage:
    export DATAXID_API_KEY="dx_..."
    python examples/imputation.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

import dataxid

dataxid.enable_logging("info")

dataxid.api_key = os.environ.get("DATAXID_API_KEY", "")


def load_data(n_rows: int = 1000) -> pd.DataFrame:
    """Load Adult (Census Income) dataset from sklearn — no local file needed."""
    dataset = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    df = dataset.frame.head(n_rows).reset_index(drop=True)
    df = df.drop(columns=["fnlwgt", "education-num"], errors="ignore")
    return df


def inject_nulls(df: pd.DataFrame, columns: list[str], rate: float = 0.2) -> pd.DataFrame:
    """Randomly mask cells in the given columns to simulate dirty data."""
    rng = np.random.default_rng(42)
    out = df.copy()
    for col in columns:
        mask = rng.random(len(out)) < rate
        out.loc[mask, col] = np.nan
    return out


# ---------------------------------------------------------------------------
# Example 1 — single trial imputation
# ---------------------------------------------------------------------------

def example_single_trial(model: dataxid.Model, dirty: pd.DataFrame) -> None:
    print("\n--- Example 1: single trial ---")

    imputed = model.impute(dirty)

    remaining_nulls = imputed.isna().sum().sum()
    print(f"Input nulls    : {dirty.isna().sum().sum()}")
    print(f"Output nulls   : {remaining_nulls}  (expected 0)")
    print(imputed.head(3).to_string())


# ---------------------------------------------------------------------------
# Example 2 — multiple trials aggregated with mode
# ---------------------------------------------------------------------------

def example_multi_trial(model: dataxid.Model, dirty: pd.DataFrame) -> None:
    print("\n--- Example 2: multiple trials, pick='mode' ---")

    imputed = model.impute(dirty, trials=5, pick="mode")

    remaining_nulls = imputed.isna().sum().sum()
    print(f"Trials         : 5")
    print(f"Pick strategy  : mode  (most frequent value per cell)")
    print(f"Output nulls   : {remaining_nulls}  (expected 0)")
    print(imputed.head(3).to_string())


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"dataxid v{dataxid.__version__}")

    if not dataxid.api_key:
        print("\nNo DATAXID_API_KEY set — skipping API examples.")
        print("Set export DATAXID_API_KEY='dx_...' to run full examples.")
    else:
        df = load_data(n_rows=1000)
        dirty = inject_nulls(df, columns=["occupation", "workclass", "age"], rate=0.2)
        print(f"Dataset: {len(df)} rows x {len(df.columns)} cols")
        print(f"Nulls injected: {dirty.isna().sum().sum()} cells")

        model = dataxid.Model.create(
            data=df,
            config=dataxid.ModelConfig(max_epochs=20),
        )
        print(f"Model ID: {model.id}")

        try:
            example_single_trial(model, dirty)
            example_multi_trial(model, dirty)
        finally:
            model.delete()
            print(f"\nModel deleted: status={model.status}")
