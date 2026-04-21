"""
DataXID SDK — Advanced Generation

Demonstrates v0.3 generation-time controls:

1. Distribution     — rebalance a single categorical column to a target ratio
2. Bias             — enforce statistical parity of a target across groups
3. Synthetic preset — bundle scalar tuning knobs, with explicit override

Each control can be passed independently or combined in a single
Model.generate() call.

Requirements:
    pip install dataxid scikit-learn

Usage:
    export DATAXID_API_KEY="dx_..."
    python examples/advanced_generation.py
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
# Example 1 — Distribution: rebalance a single categorical column
# ---------------------------------------------------------------------------

def example_distribution(model: dataxid.Model, baseline: float) -> None:
    print("\n--- Example 1: Distribution (rebalance) ---")

    distribution = dataxid.Distribution(
        column="class",
        probabilities={">50K": 0.5, "<=50K": 0.5},
    )
    synthetic = model.generate(n_samples=500, distribution=distribution)

    share = (synthetic["class"] == ">50K").mean()
    print(f"Baseline '>50K' share : {baseline:.1%}")
    print(f"Rebalanced share      : {share:.1%}  (target 50%)")


# ---------------------------------------------------------------------------
# Example 2 — Bias: statistical parity across sensitive groups
# ---------------------------------------------------------------------------

def example_bias(model: dataxid.Model) -> None:
    print("\n--- Example 2: Bias (statistical parity) ---")

    bias = dataxid.Bias(
        target="class",
        sensitive=["sex", "race"],
    )
    synthetic = model.generate(n_samples=1000, bias=bias)

    groups = synthetic.groupby(["sex", "race"])["class"].apply(
        lambda s: (s == ">50K").mean()
    )
    print("P(class='>50K' | sex, race):")
    print(groups.to_string())


# ---------------------------------------------------------------------------
# Example 3 — Synthetic preset with explicit override
# ---------------------------------------------------------------------------

def example_preset(model: dataxid.Model) -> None:
    print("\n--- Example 3: Synthetic preset + override ---")

    preset = dataxid.Synthetic(
        diversity=0.5,
        rare_cutoff=0.95,
        rare_strategy="sample",
    )

    synthetic = model.generate(n_samples=200, synthetic=preset)
    print(f"Preset run    : {len(synthetic)} rows, diversity=0.5 (from preset)")

    synthetic = model.generate(n_samples=200, synthetic=preset, diversity=0.9)
    print(f"Override run  : {len(synthetic)} rows, diversity=0.9 (kwarg wins)")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"dataxid v{dataxid.__version__}")

    if not dataxid.api_key:
        print("\nNo DATAXID_API_KEY set — skipping API examples.")
        print("Set export DATAXID_API_KEY='dx_...' to run full examples.")
    else:
        df = load_data(n_rows=1000)
        print(f"Dataset: {len(df)} rows x {len(df.columns)} cols")

        baseline_high = (df["class"] == ">50K").mean()

        model = dataxid.Model.create(
            data=df,
            config=dataxid.ModelConfig(max_epochs=20),
        )
        print(f"Model ID: {model.id}")

        try:
            example_distribution(model, baseline=baseline_high)
            example_bias(model)
            example_preset(model)
        finally:
            model.delete()
            print(f"\nModel deleted: status={model.status}")
