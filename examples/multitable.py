"""
DataXID SDK — Multi-Table Quickstart

Demonstrates synthesize_tables() with two related tables:
accounts (parent) and transactions (child with FK).

Requirements:
    pip install dataxid

Usage:
    export DATAXID_API_KEY="dx_..."
    python examples/multitable.py
"""

import os

import pandas as pd

import dataxid
from dataxid import Table

dataxid.enable_logging("info")

dataxid.api_key = os.environ.get("DATAXID_API_KEY", "")


def main() -> None:
    print(f"dataxid v{dataxid.__version__}\n")

    if not dataxid.api_key:
        print("No DATAXID_API_KEY set.")
        print("Set export DATAXID_API_KEY='dx_...' to run this example.")
        return

    df_accounts = pd.DataFrame({
        "account_id": range(1, 101),
        "district": [f"district_{i % 5}" for i in range(100)],
        "frequency": ["monthly", "weekly", "daily"] * 33 + ["monthly"],
        "open_date": pd.date_range("2020-01-01", periods=100, freq="10D"),
    })

    df_transactions = pd.DataFrame({
        "account_id": [i for i in range(1, 101) for _ in range(5)],
        "date": pd.date_range("2023-01-01", periods=500, freq="D"),
        "amount": [round(50 + i * 2.5, 2) for i in range(500)],
        "type": ["credit", "debit"] * 250,
    })

    print(f"Accounts:     {len(df_accounts)} rows")
    print(f"Transactions: {len(df_transactions)} rows")

    accounts = Table(df_accounts, primary_key="account_id")
    transactions = Table(df_transactions, foreign_keys={"account_id": accounts})

    synthetic = dataxid.synthesize_tables({
        "accounts": accounts,
        "transactions": transactions,
    })

    print(f"\nSynthetic accounts:     {len(synthetic['accounts'])} rows")
    print(f"Synthetic transactions: {len(synthetic['transactions'])} rows")

    print("\n--- Synthetic accounts (first 5) ---")
    print(synthetic["accounts"].head().to_string())

    print("\n--- Synthetic transactions (first 5) ---")
    print(synthetic["transactions"].head().to_string())


if __name__ == "__main__":
    main()
