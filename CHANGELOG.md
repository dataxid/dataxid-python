# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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

[0.1.0]: https://github.com/dataxid/dataxid-python/releases/tag/v0.1.0
