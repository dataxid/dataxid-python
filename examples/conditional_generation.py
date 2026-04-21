"""
DataXID SDK — Conditional Generation

Demonstrates Model.generate(conditions=...): fix known column values and let
the model fill in the rest. Uses the Adult (Census Income) dataset from
sklearn — no local file needed.

1. Single-column condition    — every row has class == ">50K"
2. Multi-column condition     — combined demographic constraints

Requirements:
    pip install dataxid scikit-learn

Usage:
    export DATAXID_API_KEY="dx_..."
    python examples/conditional_generation.py
"""

import os

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


# ---------------------------------------------------------------------------
# Example 1 — fix a single column
# ---------------------------------------------------------------------------

def example_single_column(model: dataxid.Model) -> None:
    print("\n--- Example 1: condition on a single column ---")

    conditions = pd.DataFrame({"class": [">50K"] * 200})
    synthetic = model.generate(conditions=conditions)

    share = (synthetic["class"] == ">50K").mean()
    print(f"Rows            : {len(synthetic)}")
    print(f"class=='>50K'   : {share:.1%}  (expected 100%)")
    print(synthetic[["age", "education", "occupation", "class"]].head(3).to_string())


# ---------------------------------------------------------------------------
# Example 2 — fix multiple columns
# ---------------------------------------------------------------------------

def example_multi_column(model: dataxid.Model) -> None:
    print("\n--- Example 2: condition on multiple columns ---")

    conditions = pd.DataFrame({
        "sex": ["Female"] * 200,
        "class": [">50K"] * 200,
    })
    synthetic = model.generate(conditions=conditions)

    female_share = (synthetic["sex"] == "Female").mean()
    high_income_share = (synthetic["class"] == ">50K").mean()
    print(f"Rows                    : {len(synthetic)}")
    print(f"sex=='Female'           : {female_share:.1%}  (expected 100%)")
    print(f"class=='>50K'           : {high_income_share:.1%}  (expected 100%)")
    print(synthetic[["age", "sex", "occupation", "class"]].head(3).to_string())


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"dataxid v{dataxid.__version__}")

    if not dataxid.api_key:
        print("\nNo DATAXID_API_KEY set — skipping API examples.")
        print("Set export DATAXID_API_KEY='dx_...' to run full examples.")
    else:
        df = load_data(n_rows=1000)
        print(f"Dataset: {len(df)} rows x {len(df.columns)} cols")

        model = dataxid.Model.create(
            data=df,
            config=dataxid.ModelConfig(max_epochs=20),
        )
        print(f"Model ID: {model.id}")

        try:
            example_single_column(model)
            example_multi_column(model)
        finally:
            model.delete()
            print(f"\nModel deleted: status={model.status}")
