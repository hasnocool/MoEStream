# MoEStream

**MoEStream** is an experimental local-inference runtime for running sparse Mixture-of-Experts (MoE) language models on consumer hardware by treating fast NVMe storage as an extension of the model's expert working set.

The project is inspired by the architectural lessons of projects such as Deltafin, WASTE, llama.cpp, and other storage-aware local inference experiments, while aiming for a **model-adapter architecture rather than a single-model runtime**.

> Status: **0.2.0-alpha.6 / Qwen3 adapter + contiguous expert-bank conversion.** This repository does not yet execute a production LLM.

## Goal

Make sparse models that are larger than system RAM practically usable on laptops and small PCs without pretending that NVMe is RAM or weakening model correctness.

```text
                 OpenAI-compatible API
                          │
                     MoEStream
                          │
                Model Adapter Layer
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
    Kimi-style         Qwen MoE         DeepSeek MoE
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │
                 Authoritative Router
                          │
                 Expert Working Set
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
       RAM               NVMe              GPU
    hot cache          cold store        hot cache
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
                       compute
```

## Design principles

1. **Correctness first.** Prefetch prediction may schedule I/O but never replace authoritative model routing.
2. **Sparse models first.** Dense models require most weights per token and usually gain less from storage streaming.
3. **Non-blocking orchestration.** Disk and network operations are async; expensive blocking work must be isolated from executor threads.
4. **Model-specific semantics behind adapters.** Routing, tensor layout, tokenizer, attention variants, and expert storage formats belong in model adapters/providers.
5. **Storage is a tier, not fake RAM.** The runtime explicitly schedules NVMe reads, cache admission, residency, and eviction.
6. **Measure everything.** Cache hit rate, bytes/token, read amplification, I/O latency, route predictability, and tokens/sec are first-class metrics.
7. **OpenAI-compatible surface.** Local coding agents should be able to connect without custom client integrations.

## Why MoE models?

A dense 32B model may need most of its parameters for every generated token. A sparse 80B MoE might activate only a small subset per token. That makes a tiered expert store plausible:

```text
80B total parameters
████████████████████████████████████████

~5B active for a token
████
```

The resident model spine and hot experts can remain in RAM/GPU memory while colder experts live on NVMe and are prefetched before use.

## Planned runtime

```text
small draft model ── speculative proposals ──┐
                                             ▼
input → resident spine → authoritative router → target experts
                              │                    │
                              │              cache lookup
                              │                    │
                              │          ┌─────────┴─────────┐
                              │          │                   │
                              └────► prefetch            cache hit
                                         │                   │
                                        NVMe                RAM/GPU
                                         │                   │
                                         └─────────┬─────────┘
                                                   ▼
                                                compute
                                                   │
                                                verify
                                                   │
                                                 output
```

## Current implementation

The Rust runtime foundation contains:

- a thread-safe `ModelAdapter` boundary;
- authoritative expert IDs and storage locations;
- bounded asynchronous expert byte-range reads;
- request coalescing so concurrent misses for the same expert perform one physical read;
- configurable disk-I/O concurrency using a semaphore;
- LRU host-RAM expert eviction;
- runtime cache metrics for hits, misses, coalescing, evictions, and physical bytes read;
- tests for range validation, concurrent access, and LRU behavior;
- Qwen3 MoE config parsing and structural validation;
- a correctness-oriented Qwen3 softmax/top-k routing fixture helper;
- a versioned expert-bank manifest/index that maps `(layer, expert)` IDs to validated source-file byte ranges;
- safe relative-path checks, duplicate-ID/range validation, asynchronous declared-size verification, and SHA-256 content verification for expert-bank files;
- an asynchronous safetensors header parser that discovers dtype, shape, and exact source-file tensor spans without reading weight payloads;
- safetensors layout validation for bounded, contiguous tensor data sections;
- sharded `model.safetensors.index.json` parsing with safe shard-path validation;
- header-only checkpoint inventory that cross-checks weight-map entries against actual shard contents and resolves tensors to absolute source spans;
- Qwen3-specific discovery and validation of every expert's `gate_proj`, `up_proj`, and `down_proj` tensor triplet;
- bounded asynchronous repacking of those three source spans into deterministic contiguous expert-bank records;
- automatic validated expert-bank manifest generation with layer/expert offsets and lengths;
- temporary-file publication so partially written banks/manifests are not exposed as completed output;
- an OpenAI-compatible API skeleton;
- `/health` and `/v1/models` endpoints;
- CI and project governance documentation.

The current cache deliberately uses ordinary asynchronous file seek/read operations. Direct I/O, io_uring-specific paths, mmap experiments, prefetch cancellation, and accelerator residency should only be added when benchmarks demonstrate that they improve the intended hardware targets.

Expert-bank integrity verification streams source files through a 1 MiB hashing buffer, so full model files are never materialized in RAM. Sources without declared hashes remain size-validated but are skipped by content verification.

Checkpoint discovery reads only the safetensors prefix and JSON headers. Sharded checkpoint inventory adds the index JSON and cross-checks every referenced shard header without touching tensor payload bytes, so even very large Qwen3 checkpoints can be structurally inventoried with bounded memory.

For Qwen3, the preparation layer now validates expected expert tensor names, dtypes, and shapes, then copies `gate_proj`, `up_proj`, and `down_proj` in deterministic layer/expert order into one contiguous expert record. Copying uses a bounded 1 MiB buffer and async seek/read/write operations, so neither source shards nor complete expert banks are materialized in RAM. A matching runtime manifest is generated automatically.

The Qwen3 routing helper operates on already-computed router logits. It is not yet the authoritative tensor-level router; router weight decoding, exact reference parity, tokenizer support, and expert execution remain part of the 0.2 milestone.

## First model target

The first real adapter target is **Qwen3-30B-A3B**. It has 30.5B total parameters with roughly 3.3B activated, 128 experts, and 8 experts selected per token, making it a practical first test of NVMe-backed expert streaming on consumer hardware.

Newer Qwen3.5/Qwen3.6 MoE families exist, but their hybrid attention/DeltaNet and shared-expert/multimodal architecture adds unnecessary complexity to the first parity bring-up. They remain later compatibility targets.

See [`docs/FIRST-MODEL-TARGET.md`](docs/FIRST-MODEL-TARGET.md) for the selection rationale, implementation phases, reference-comparison plan, and benchmark matrix.

## Run the scaffold

```bash
cargo run -- status
cargo run -- serve --address 127.0.0.1:8000
```

Then:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

## Repository roadmap

See [`TODO.md`](TODO.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

Kimi K3 remains a long-term extreme storage-streaming target, but it should follow correctness and performance validation on smaller sparse models.

## Non-goals for 0.x

- claiming that arbitrary GGUF models work without model-specific semantics;
- silently pruning experts or changing routing to improve benchmarks;
- hiding model-quality tradeoffs behind storage optimizations;
- blocking async executor threads with disk or compute work;
- optimizing only for headline tokens/sec while ignoring I/O volume and power use.

## License

MIT. Third-party model weights and runtimes retain their own licenses.
