"""
Dataxid SDK — Quickstart

Demonstrates:
1. synthesize()     — one-liner, simplest usage
2. Model.create()   — full control with ModelConfig
3. Error handling   — catching SDK exceptions

Requirements:
    pip install dataxid scikit-learn

Usage:
    export DATAXID_API_KEY="dx_..."
    python examples/quickstart.py
"""

import os

import pandas as pd
from sklearn.datasets import fetch_openml

import dataxid

dataxid.enable_logging("info")

dataxid.api_key = os.environ.get("DATAXID_API_KEY", "")


def load_data(n_rows: int = 500) -> pd.DataFrame:
    """Load Adult (Census Income) dataset from sklearn — no local file needed."""
    dataset = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    df = dataset.frame.head(n_rows).reset_index(drop=True)
    df = df.drop(columns=["fnlwgt", "education-num"], errors="ignore")
    return df


# ---------------------------------------------------------------------------
# Example 1 — synthesize() one-liner
# ---------------------------------------------------------------------------

def example_one_liner(df: pd.DataFrame) -> None:
    print("\n--- Example 1: synthesize() ---")

    synthetic = dataxid.synthesize(data=df, n_samples=100)

    print(f"Original : {len(df)} rows x {len(df.columns)} cols")
    print(f"Synthetic: {len(synthetic)} rows x {len(synthetic.columns)} cols")
    print(synthetic.head(3).to_string())


# ---------------------------------------------------------------------------
# Example 2 — Model.create() with ModelConfig
# ---------------------------------------------------------------------------

def example_full_control(df: pd.DataFrame) -> None:
    print("\n--- Example 2: Model.create() with ModelConfig ---")

    model = dataxid.Model.create(
        data=df,
        config=dataxid.ModelConfig(
            embedding_dim=64,
            model_size="medium",
            max_epochs=20,
        ),
    )

    print(f"Model ID : {model.id}")
    print(f"Epochs   : {len(model.train_losses)}, early_stopped={model.stopped_early}")

    synthetic = model.generate(n_samples=100)
    print(f"Synthetic: {len(synthetic)} rows")

    model.delete()
    print(f"Status   : {model.status}")


# ---------------------------------------------------------------------------
# Example 3 — Error handling
# ---------------------------------------------------------------------------

def example_error_handling(df: pd.DataFrame) -> None:
    print("\n--- Example 3: Error handling ---")

    try:
        dataxid.synthesize(data=df, n_samples=10, api_key="dx_invalid_key")
    except dataxid.AuthenticationError as e:
        print(f"AuthenticationError (expected): {e}")
    except dataxid.QuotaExceededError as e:
        print(f"QuotaExceededError: {e}  upgrade_url={e.upgrade_url}")
    except dataxid.RateLimitError as e:
        print(f"RateLimitError: retry_after={e.retry_after}s")
    except dataxid.DataxidError as e:
        print(f"DataxidError [{type(e).__name__}]: {e}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"dataxid v{dataxid.__version__}")

    df = load_data(n_rows=500)
    print(f"Dataset: {len(df)} rows x {len(df.columns)} cols — {list(df.columns)}")

    example_error_handling(df)

    if not dataxid.api_key:
        print("\nNo DATAXID_API_KEY set — skipping API examples.")
        print("Set export DATAXID_API_KEY='dx_...' to run full examples.")
    else:
        example_one_liner(df)
        example_full_control(df)
