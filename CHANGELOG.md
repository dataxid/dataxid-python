# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-04-21

This release introduces typed dataclasses for structured generation
configuration, exposes new generation-time controls (distribution, bias,
conditions, imputation, presets), and consolidates the `Privacy` config
under a dedicated sub-config.

Two small breaking changes; code that does not use `seed_data` or the
top-level `privacy_enabled` / `privacy_noise` fields upgrades without
changes.

### Breaking Changes

1. **`Model.generate(seed_data=df)` → `Model.generate(conditions=df)`**
   The `seed_data` keyword is renamed to `conditions` for clarity.
   `synthesize()` and `synthesize_tables()` are renamed identically.
2. **Privacy config is now nested under a `Privacy` dataclass**
   Top-level `privacy_enabled` and `privacy_noise` on `ModelConfig`
   have moved into a dedicated `Privacy` sub-config.

### Added

**New dataclasses (top-level exports)**:

- `dataxid.Distribution` — override natural category distributions at generation time
- `dataxid.Bias` — statistical parity correction across sensitive attributes
- `dataxid.Synthetic` — reusable bundle of generation-time tuning parameters
- `dataxid.Privacy` — typed privacy config (nested in `ModelConfig`)

**New `Model.generate()` parameters**:

- `distribution: Distribution | None` — rebalance category outputs (single-table; per-table dict in multi-table)
- `bias: Bias | None` — statistical parity correction
- `conditions: pd.DataFrame | None` — fix known values (replaces `seed_data`)
- `diversity: float = 1.0` — sampling temperature
- `rare_cutoff: float = 1.0` — nucleus-style truncation for rare categories
- `rare_strategy: "mask" | "sample" | None` — per-call override for rare-category handling
- `seed: int | None` — exposes deterministic generation
- `synthetic: Synthetic | None` — apply a preset; direct keyword arguments override preset values

**New imputation API** — single dedicated method:

```python
model.impute(df_with_nulls)

# or with multiple trials for more stable cells:
model.impute(df_with_nulls, trials=5, pick="mode")
```

- `trials` (default `1`) — number of independent generation passes
- `pick` — aggregation across trials: `"mode"` (default), `"mean"`, `"median"`, `"all"`, or a custom callable

**Privacy controls** (defaults preserve v0.2.0 behavior):

- `Privacy.protect_rare` (default `True`) — hide rare categorical values
- `Privacy.rare_strategy` (default `"mask"`) — whether protected values appear as `<protected>` or are re-sampled

### Changed

- Rare-category output may include the `<protected>` sentinel when `rare_strategy="mask"` (default).

### Notes

- The v0.2.0 public surface — `synthesize(data, n_samples=...)`, `Model.create()`, `Model.generate(n_samples=..., parent=...)`, `Table` / `synthesize_tables`, and the error hierarchy — is **preserved**.
- See the updated examples in `README.md` and `examples/` for end-to-end usage of the new parameters.

## [0.2.0] - 2026-04-13

### Added

- **Multi-table synthesis**: `Table` class and `dataxid.synthesize_tables()` for generating related tables with referential integrity
- **Sequential / time-series generation**: automatic per-entity sequence modeling when foreign keys are present (`sequential`, `sequence_by`)
- **Context-aware generation**: `parent`, `parent_key`, `foreign_key` parameters on `synthesize()` and `Model.create()`
- **Datetime auto-detection**: heuristic that detects datetime columns by name patterns and value sampling
- **SDK logging**: `dataxid.enable_logging()` / `disable_logging()` and `DATAXID_LOG` env var (default off, sensitive headers masked)
- Multi-table quickstart example (`examples/multitable.py`)

## [0.1.0] - 2026-03-10

### Added

- `dataxid.synthesize()` — one-call synthetic data generation
- `dataxid.Model.create()` — full control over training and generation
- `dataxid.ModelConfig` — typed configuration with IDE autocomplete
- Privacy-preserving encoder (raw data never leaves your machine)
- Split-learning: only 64-float embeddings cross the API boundary
- Frozen encoder mode for faster training (~10x) and lower egress (~150x)
- HTTP training with server-side orchestration (early stopping, LR scheduling)
- Privacy noise injection for embeddings (`privacy_enabled`, `privacy_noise`)
- Structured error hierarchy (`DataxidError` → `AuthenticationError`, `RateLimitError`, etc.)
- Automatic retry with exponential backoff and `Retry-After` support
- `DATAXID_API_KEY` environment variable support

[0.3.0]: https://github.com/dataxid/dataxid-python/releases/tag/v0.3.0
[0.2.0]: https://github.com/dataxid/dataxid-python/releases/tag/v0.2.0
[0.1.0]: https://github.com/dataxid/dataxid-python/releases/tag/v0.1.0
