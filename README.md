# MoEStream

**MoEStream** is an experimental local-inference runtime for running sparse Mixture-of-Experts (MoE) language models on consumer hardware by treating fast NVMe storage as an extension of the model's expert working set.

The project is inspired by the architectural lessons of projects such as Deltafin, WASTE, llama.cpp, and other storage-aware local inference experiments, while aiming for a **model-adapter architecture rather than a single-model runtime**.

> Status: **0.1.0 / architecture + runtime scaffold.** This repository does not yet execute a production LLM.

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

The initial Rust scaffold contains:

- a thread-safe `ModelAdapter` boundary;
- authoritative expert IDs and storage locations;
- an async expert cache prototype;
- a runtime object that loads only routed experts;
- an OpenAI-compatible API skeleton;
- `/health` and `/v1/models` endpoints;
- CI and project governance documentation.

The cache currently uses whole-file async reads before slicing the requested expert range. This is intentionally marked as a prototype; the storage milestone will replace it with bounded range reads/direct I/O experiments and measured cache policies.

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

The first meaningful inference milestone is a **small supported MoE model adapter** that is practical to benchmark on commodity NVMe hardware. Kimi K3 should not be the first implementation target because its scale makes iteration needlessly expensive.

## Non-goals for 0.x

- claiming that arbitrary GGUF models work without model-specific semantics;
- silently pruning experts or changing routing to improve benchmarks;
- hiding model-quality tradeoffs behind storage optimizations;
- blocking async executor threads with disk or compute work;
- optimizing only for headline tokens/sec while ignoring I/O volume and power use.

## License

MIT. Third-party model weights and runtimes retain their own licenses.
