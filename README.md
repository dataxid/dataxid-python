# Dataxid Python SDK

[![PyPI version](https://img.shields.io/pypi/v/dataxid)](https://pypi.org/project/dataxid/)
[![Python versions](https://img.shields.io/pypi/pyversions/dataxid)](https://pypi.org/project/dataxid/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/dataxid/dataxid-python/blob/main/LICENSE)

Privacy-preserving synthetic data generation, built on a privacy-by-architecture principle. Your raw data never leaves your machine — only abstract embeddings are shared with the API.

## Installation

```bash
pip install dataxid
```

## Quick Start

```python
import dataxid
import pandas as pd

dataxid.api_key = "dx_..."

df = pd.read_csv("data.csv")
synthetic = dataxid.synthesize(data=df, n_samples=1000)
```

## Full Control

```python
import dataxid
import pandas as pd

dataxid.api_key = "dx_..."

df = pd.read_csv("data.csv")

model = dataxid.Model.create(data=df)
synthetic = model.generate(n_samples=1000)
model.delete()
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
model = dataxid.Model.create(
    data=df,
    config=dataxid.ModelConfig(
        embedding_dim=128,
        model_size="large",
        max_epochs=50,
    ),
)
```

Plain dict also works for quick experiments:

```python
model = dataxid.Model.create(
    data=df,
    config={"embedding_dim": 128, "max_epochs": 50},
)
```

## Links

- [Documentation](https://docs.dataxid.com)
- [API Reference](https://docs.dataxid.com/docs/api-reference)
- [GitHub](https://github.com/dataxid/dataxid-python)
- [Examples](examples/quickstart.py)
