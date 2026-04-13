# DataXID Python SDK

[![PyPI version](https://img.shields.io/pypi/v/dataxid)](https://pypi.org/project/dataxid/)
[![Python versions](https://img.shields.io/pypi/pyversions/dataxid)](https://pypi.org/project/dataxid/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/dataxid/dataxid-python/blob/main/LICENSE)

High-fidelity synthetic data generation for single-table, multi-table, and time series data.

## Installation

```bash
pip install dataxid
```

## Quick Start

```python
import dataxid
import pandas as pd

dataxid.api_key = "dx_..."
dataxid.enable_logging("info")  # optional: see training progress

df = pd.read_csv("customers.csv")
synthetic = dataxid.synthesize(data=df, n_samples=1000)
```

## Multi-Table & Time Series

Synthesize related tables with referential integrity. Child tables are
generated sequentially by default — preserving realistic per-entity
patterns like transaction counts, temporal ordering, and sequence lengths.

```python
from dataxid import Table

accounts = Table(pd.read_csv("accounts.csv"), primary_key="account_id")
transactions = Table(pd.read_csv("transactions.csv"),
                     foreign_keys={"account_id": accounts})

synthetic = dataxid.synthesize_tables({
    "accounts": accounts,
    "transactions": transactions,
})

synthetic["accounts"]       # synthetic accounts with auto-assigned PKs
synthetic["transactions"]   # sequential transactions per account, valid FKs
```

## How It Works

Dataxid is built on a **privacy-by-architecture** principle. Data encoding and decoding happen entirely on your machine; only abstract embeddings are shared with the API for model training. Raw data never leaves your environment.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | `64` | Embedding size (larger = more expressive) |
| `model_size` | `"medium"` | Model capacity: `"small"`, `"medium"`, `"large"` |
| `max_epochs` | `100` | Maximum training epochs |
| `batch_size` | `256` | Training batch size |
| `privacy_enabled` | `False` | Add noise to embeddings for privacy |
| `privacy_noise` | `0.1` | Noise scale (Gaussian std) |

```python
import dataxid
import pandas as pd

dataxid.api_key = "dx_..."

df = pd.read_csv("customers.csv")

model = dataxid.Model.create(
    data=df,
    config=dataxid.ModelConfig(
        embedding_dim=128,
        model_size="large",
        max_epochs=50,
    ),
)
synthetic = model.generate(n_samples=1000)
model.delete()
```

`synthesize_tables` handles orchestration automatically. Use `Model.create` for fine-grained control:

```python
model = dataxid.Model.create(
    data=transactions_df,
    parent=accounts_df,
    foreign_key="account_id",
)
synthetic = model.generate(parent=synthetic_accounts_df)
model.delete()
```

## Logging

```python
dataxid.enable_logging("info")   # see training progress, epoch stats
dataxid.enable_logging("debug")  # verbose: includes HTTP requests
dataxid.disable_logging()        # turn off (default state)
```

Or via environment variable (no code change needed):

```bash
DATAXID_LOG=info python my_script.py
```

## Error Handling

```python
import dataxid

try:
    synthetic = dataxid.synthesize(data=df)
except dataxid.AuthenticationError:
    print("Invalid API key")
except dataxid.QuotaExceededError as e:
    print(f"Quota exceeded. Upgrade: {e.upgrade_url}")
except dataxid.RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after}s")
except dataxid.DataxidError as e:
    print(f"Error: {e}")
```

## Links

- [Documentation](https://docs.dataxid.com)
- [API Reference](https://docs.dataxid.com/docs/api-reference)
- [GitHub](https://github.com/dataxid/dataxid-python)
