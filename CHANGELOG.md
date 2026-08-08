# Changelog

All notable changes to this project will be documented in this file.

The project follows Semantic Versioning.

## [0.2.0-alpha.8] - 2026-08-07

### Added

- Recorded PyTorch 2.10.0 CPU BF16 router parity fixture containing BF16 weight/hidden/logit bit patterns, float32 softmax outputs, top-k indices, and selected weights before/after normalization.
- Python fixture generator using the reference `torch.nn.functional.linear` → float32 softmax → `torch.topk` sequence.
- Rust integration test that reconstructs the BF16 safetensors router through the real checkpoint inventory/loader and compares against the recorded PyTorch outputs.
- Router parity documentation defining the current numerical contract and the remaining real-checkpoint promotion gate.

### Changed

- Package version advanced to `0.2.0-alpha.8`.
- Router parity requirements now distinguish portable non-tied routing parity from tied `topk` ordering, which is treated as reference-environment-specific rather than a model invariant.

### Known limitations

- The recorded fixture is synthetic and CPU-based; at least one captured hidden state from the real Qwen3-30B-A3B checkpoint is still required before production-authoritative promotion.
- Floating-point and tied top-k behavior may vary by PyTorch release/platform, so parity is defined with explicit fixture environments and tolerances.
- Tokenizer support, expert tensor decoding, CPU end-to-end execution, and token/logit parity remain pending.

## [0.2.0-alpha.7] - 2026-08-07

### Added

- Qwen3 router tensor discovery for `model.layers.{layer}.mlp.gate.weight`.
- BF16 and F32 router-weight decoding from indexed safetensors byte spans.
- Router tensor shape and byte-length validation against configured expert count and hidden size.
- Correctness-oriented f32 router matrix-vector execution using decoded checkpoint weights.
- Async routing entrypoint that isolates CPU matrix-vector work on Tokio's blocking pool before applying softmax/top-k selection.
- Unit tests for F32 loading/logits, BF16 decoding, route ordering, and normalized route weights.

### Changed

- Package version advanced to `0.2.0-alpha.7`.
- The roadmap now distinguishes checkpoint router decoding/reference execution from the later parity gate required before the router path is considered production-authoritative.

### Known limitations

- The f32 matrix-vector reference path does not yet claim bit-for-bit parity with PyTorch BF16 linear execution or exact `torch.topk` tie behavior.
- F16/quantized router tensors are not yet supported by this bring-up path; the official Qwen3-30B-A3B checkpoint uses BF16 router weights.
- Tokenizer support, expert tensor decoding, CPU end-to-end execution, and token/logit parity remain pending.

## [0.2.0-alpha.6] - 2026-08-07

### Added

- Qwen3-specific checkpoint preparation that discovers every expert's `gate_proj`, `up_proj`, and `down_proj` tensor triplet.
- Validation that expert projection dtypes match and that tensor shapes agree with Qwen3 hidden/MoE intermediate dimensions.
- Deterministic expert-bank packing in layer-major/expert-major order with `gate_proj`, `up_proj`, then `down_proj` payload order.
- Bounded 1 MiB asynchronous source-range copy buffers so checkpoint shards and complete expert records are never materialized in RAM.
- Automatic expert-bank manifest generation with validated `(layer, expert)` offsets and lengths.
- Temporary-file publication for expert-bank and manifest outputs so incomplete conversions are not exposed as final artifacts.
- Tests covering expert discovery, shape rejection, packed byte ordering, bank offsets, bank size, and generated manifest validation.

### Changed

- Package version advanced to `0.2.0-alpha.6`.
- Qwen3 checkpoint preparation now progresses from header/index inventory to actual runtime expert-bank generation.

### Known limitations

- Generated bank manifests do not yet embed a SHA-256 for the newly packed bank; integrity hashing can be run through the existing manifest verification path after a digest is supplied.
- Router tensor execution, tokenizer support, expert tensor decoding, and CPU inference remain pending.

## [0.2.0-alpha.5] - 2026-08-07

### Added

- Parser and validator for sharded `model.safetensors.index.json` checkpoint indexes.
- Safe relative `.safetensors` shard-path validation with traversal rejection.
- Header-only checkpoint inventory that cross-checks every weight-map entry against the declared shard and rejects unindexed tensors in referenced shards.
- Tensor resolution from logical tensor name to source shard, dtype/shape metadata, and absolute source-file byte span.
- Unit tests for multi-shard inventory, path traversal, stale shard mappings, and partial indexes.

### Changed

- Package version advanced to `0.2.0-alpha.5`.
- Qwen3 checkpoint preparation can now discover the full sharded tensor inventory without loading tensor payloads into RAM.

### Known limitations

- Qwen3 `gate_proj`, `up_proj`, and `down_proj` tensor payloads are not yet repacked into contiguous expert records.
- Automatic expert-bank manifest generation remains pending the repacking stage.

## [0.2.0-alpha.4] - 2026-08-07

### Added

- Asynchronous safetensors prefix/header reader that does not read tensor payload bytes.
- Structured tensor metadata for dtype, shape, and data-relative offsets.
- Absolute source-file tensor span resolution for later checkpoint conversion.
- Header-size limits and checked offset arithmetic for untrusted or corrupted checkpoint metadata.
- Validation that tensor spans remain in bounds and cover the safetensors data section without holes or trailing unindexed bytes.
- Tests covering metadata extraction, absolute span resolution, malformed JSON, oversized headers, out-of-bounds tensor spans, layout holes, and trailing data.

### Changed

- Package version advanced to `0.2.0-alpha.4`.
- Checkpoint preparation is now explicitly split into metadata discovery, sharded-index inventory, and contiguous expert-bank repacking.

### Design decision

- Qwen3 experts are represented upstream by separate `gate_proj`, `up_proj`, and `down_proj` tensors. The first converter will repack those spans into one contiguous MoEStream expert record so the hot runtime can preserve one logical expert read instead of issuing multiple source-file reads.

### Known limitations

- Sharded `model.safetensors.index.json` inventory is not yet implemented.
- No tensor payload is decoded or converted by this release.

## [0.2.0-alpha.3] - 2026-08-07

### Added

- Full SHA-256 verification for manifest source files that declare a digest.
- 1 MiB streaming hash buffers so verification does not materialize model files in RAM.
- Isolation of synchronous file reads and CPU hashing on Tokio's blocking pool so runtime worker threads remain non-blocking.
- Corruption-detection and successful-verification unit tests using deterministic source fixtures.

### Changed

- Package version advanced to `0.2.0-alpha.3`.
- Hash verification first performs asynchronous source type/size validation before starting expensive content hashing.

### Known limitations

- Source files without a declared SHA-256 digest are size-checked but intentionally skipped by content verification.
- Manifests are not yet generated automatically from upstream checkpoints.

## [0.2.0-alpha.2] - 2026-08-07

### Added

- Versioned expert-bank manifest schema.
- Generic `ExpertIndex` mapping `(layer, expert)` identities to validated byte-range locations.
- Manifest validation for safe relative source paths, declared file sizes, duplicate expert IDs, unknown file references, byte-range overflow, and out-of-bounds spans.
- Optional declared SHA-256 metadata with strict hexadecimal-format validation.
- Asynchronous manifest loading and source-file size verification.
- Unit tests for index construction, duplicate rejection, path traversal rejection, out-of-range spans, and async file-size verification.

### Changed

- Package version advanced to `0.2.0-alpha.2`.
- Roadmap now separates manifest/index validation from future checkpoint conversion and full SHA-256 content verification.

### Known limitations

- Declared SHA-256 values are validated syntactically but are not yet computed against source file contents.
- Manifests are not yet generated automatically from upstream checkpoints.

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
- Unit tests for range validation, concurrent access, and LRU behavior.
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
