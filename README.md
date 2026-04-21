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

df = pd.read_csv("data.csv")
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

Per-table generation controls are passed as dicts keyed by table name:

```python
from dataxid import Synthetic, Distribution

accounts_preset = Synthetic(n=1000)
transactions_preset = Synthetic(n=5000, seed=42)

country_distribution = Distribution(
    column="country",
    probabilities={"US": 0.6, "UK": 0.4},
)

synthetic = dataxid.synthesize_tables(
    tables={"accounts": accounts, "transactions": transactions},
    synthetic={
        "accounts": accounts_preset,
        "transactions": transactions_preset,
    },
    distribution={"accounts": country_distribution},
)
```

## Iterative workflow

When you want to train once and generate many times — for example,
running several sampling strategies against the same model — split the
call into `Model.create` and `model.generate`:

```python
model = dataxid.Model.create(data=df)
synthetic_a = model.generate(n_samples=1000, diversity=0.8)
synthetic_b = model.generate(n_samples=1000, diversity=1.2)
model.delete()
```

## How It Works

DataXID is built on a **privacy-by-architecture** principle. Data encoding and decoding happen entirely on your machine; only abstract embeddings are shared with the API for model training. Raw data never leaves your environment.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | `64` | Embedding size (larger = more expressive) |
| `model_size` | `"medium"` | Model capacity: `"small"`, `"medium"`, `"large"` |
| `max_epochs` | `100` | Maximum training epochs |
| `batch_size` | `256` | Training batch size |
| `privacy` | `Privacy()` | Privacy config (see below) |

```python
config = dataxid.ModelConfig(
    embedding_dim=128,
    model_size="large",
    max_epochs=50,
    privacy=dataxid.Privacy(enabled=True, noise=0.2),
)
model = dataxid.Model.create(data=df, config=config)
```

A plain dict is also accepted for quick experiments: `config={"embedding_dim": 128}`.

### Privacy

Privacy settings are grouped under a dedicated `Privacy` config:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `False` | Add Gaussian noise to embeddings before they leave the machine |
| `noise` | `0.1` | Noise scale (Gaussian std) when `enabled=True` |
| `protect_rare` | `True` | Hide rare categorical values behind a `<protected>` token |
| `rare_strategy` | `"mask"` | How protected values appear: `"mask"` or `"sample"` |

## Advanced Generation

### Rebalance category distributions

Override the natural distribution of a categorical column:

```python
model = dataxid.Model.create(data=df)

distribution = dataxid.Distribution(
    column="gender",
    probabilities={"M": 0.5, "F": 0.5},
)
synthetic = model.generate(n_samples=1000, distribution=distribution)
```

### Bias correction

Reduce statistical parity gaps across sensitive attributes:

```python
bias = dataxid.Bias(
    target="income",
    sensitive=["gender", "race"],
)
synthetic = model.generate(n_samples=1000, bias=bias)
```

### Conditional generation

Fix known values and let the model complete the rest:

```python
conditions = pd.DataFrame({"income": [">50K"] * 1000})

synthetic = model.generate(conditions=conditions)
```

### Impute missing values

Fill `NaN` cells with model predictions; non-NULL cells are preserved:

```python
model = dataxid.Model.create(data=df)
filled = model.impute(df, trials=3, pick="mode")
```

### Tuning presets

Bundle generation-time knobs into a reusable preset:

```python
preset = dataxid.Synthetic(
    n=1000,
    seed=42,
    diversity=0.8,
    rare_cutoff=0.95,
)
synthetic = model.generate(synthetic=preset)
```

Direct keyword arguments override preset values:

```python
synthetic = model.generate(synthetic=preset, diversity=1.2)  # diversity=1.2 wins
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
- [Examples](examples/quickstart.py)
