# Agent Instructions

## Project goal

Build a correctness-first, storage-aware inference runtime for sparse MoE models on consumer hardware.

## Required workflow

Before changing code:

1. Confirm the change supports the project goal.
2. Inspect existing architecture and avoid duplicate functionality.
3. Identify downstream impacts on runtime, adapters, tests, docs and benchmarks.
4. Prefer measured improvements over speculative complexity.

For every meaningful change:

- preserve non-blocking async behavior;
- use thread-safe synchronization;
- keep model routing authoritative;
- add or update tests;
- update `README.md`, `TODO.md`, `CHANGELOG.md`, and architecture documentation when behavior or milestones change;
- follow Semantic Versioning and keep `Cargo.toml` version aligned with release documentation;
- document performance claims with reproducible benchmark conditions.

Do not silently trade output correctness for speed. Any quantization, pruning, approximate routing, speculative execution, or caching tradeoff must be explicit and measurable.
