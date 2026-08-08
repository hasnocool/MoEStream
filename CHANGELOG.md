# Changelog

All notable changes to this project will be documented in this file.

The project follows Semantic Versioning.

## [0.1.1] - 2026-08-07

### Added

- Bounded asynchronous expert byte-range reads.
- Coalescing for concurrent requests targeting the same expert.
- Configurable expert I/O concurrency limits.
- LRU cache eviction.
- Cache metrics for hits, misses, coalesced waits, evictions, and bytes read.
- Unit tests for range validation, concurrent loads, and LRU behavior.

### Changed

- Runtime configuration now exposes expert I/O concurrency.
- Expert loading no longer reads entire expert-bank files before slicing ranges.

## [0.1.0] - 2026-08-07

### Added

- Initial Rust project scaffold.
- Generic `ModelAdapter` abstraction for sparse model families.
- Thread-safe asynchronous expert cache prototype.
- Runtime skeleton for authoritative expert acquisition.
- OpenAI-compatible HTTP API scaffold with health/model endpoints.
- Architecture, roadmap, CI, governance, and licensing documentation.
