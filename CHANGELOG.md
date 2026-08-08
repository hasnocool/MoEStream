# Changelog

All notable changes to this project will be documented in this file.

The project follows Semantic Versioning.

## [0.2.0-alpha.1] - 2026-08-07

### Added

- Qwen3 MoE adapter module.
- Strict parsing and validation for Qwen3 MoE configuration metadata.
- Correctness-oriented softmax/top-k routing reference helper with optional top-k probability normalization.
- Unit tests for official Qwen3-30B-A3B configuration dimensions, invalid configurations, route ordering, normalization, and deterministic fixture tie handling.

### Changed

- Package version advanced to `0.2.0-alpha.1` for the first real model-adapter development milestone.
- Roadmap now separates deterministic routing fixtures from authoritative router-weight tensor execution and reference-runtime parity.

### Known limitations

- Router logits are supplied to the reference helper; MoEStream does not yet decode or execute the Qwen3 router weight tensor.
- Deterministic fixture tie ordering is not claimed to reproduce PyTorch `topk` tie behavior until verified against the reference runtime.
- Tokenization, expert tensor decoding, and token generation are not yet implemented.

## [0.1.1] - 2026-08-07

### Added

- Bounded asynchronous expert byte-range reads.
- Coalescing for concurrent requests targeting the same expert.
- Configurable expert I/O concurrency limits.
- LRU cache eviction.
- Cache metrics for hits, misses, coalesced waits, evictions, and bytes read.
- Unit tests for range validation, concurrent loads, and LRU behavior.
- First-adapter target analysis selecting Qwen3-30B-A3B and defining the correctness/benchmark plan.

### Changed

- Runtime configuration now exposes expert I/O concurrency.
- Expert loading no longer reads entire expert-bank files before slicing ranges.
- CLI status output derives its version directly from Cargo package metadata.

## [0.1.0] - 2026-08-07

### Added

- Initial Rust project scaffold.
- Generic `ModelAdapter` abstraction for sparse model families.
- Thread-safe asynchronous expert cache prototype.
- Runtime skeleton for authoritative expert acquisition.
- OpenAI-compatible HTTP API scaffold with health/model endpoints.
- Architecture, roadmap, CI, governance, and licensing documentation.
