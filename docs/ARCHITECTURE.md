# Architecture

## Objective

MoEStream separates **model semantics** from **storage/runtime policy** so multiple sparse model families can share expert caching, prefetch, metrics, and API infrastructure.

## Layers

### 1. API and request scheduler

Accepts OpenAI-compatible requests, owns admission control, conversation state, cancellation, and streaming output.

### 2. Model adapter

A model adapter is authoritative for:

- tokenizer/chat template;
- layer graph;
- expert count and activation count;
- router semantics;
- expert tensor layout and dtype;
- attention/recurrent state semantics;
- tensor execution provider integration.

Storage hints are advisory only. The adapter's authoritative route must decide which experts participate.

### 3. Expert scheduler

The scheduler will combine:

- current authoritative route;
- predicted next-layer route;
- cache residency snapshot;
- I/O queue depth;
- available RAM/VRAM;
- measured expert reuse probability;
- cancellation of losing prefetches.

Authoritative expert loads already use bounded I/O concurrency. Predictive prefetch cancellation remains a separate scheduling milestone so speculative reads cannot accidentally affect correctness.

### 4. Tiered cache

```text
L0  accelerator memory    fastest / smallest
L1  host RAM              fast / moderate
L2  local NVMe            large / latency-sensitive
L3  remote object store   optional / cold bootstrap only
```

The runtime should promote/demote explicit expert objects rather than relying on accidental OS swapping.

The current host cache implements:

- asynchronous byte-range reads using seek + exact read;
- explicit file-bound validation before allocation/read completion;
- a semaphore limiting concurrent physical reads;
- per-expert in-flight request coalescing;
- LRU eviction by expert identity;
- counters for logical hits/misses, coalesced waits, evictions, and physical bytes read.

These primitives establish the baseline for later direct-I/O, io_uring, mmap, and accelerator-residency experiments. Alternative storage paths should be benchmarked against this baseline before adoption.

### 5. Execution provider

Compute backends are intentionally separate from storage scheduling. Planned providers:

- CPU: AVX2/AVX-512 where available;
- Vulkan for broad laptop GPU support;
- CUDA for NVIDIA;
- Metal for Apple Silicon.

## Concurrency rules

- Async request paths must not perform blocking disk or network operations.
- CPU-bound tensor work that cannot be asynchronous must execute in a dedicated compute pool/provider, not on Tokio worker threads.
- Shared cache metadata must use bounded, thread-safe synchronization.
- Physical expert reads must be limited by explicit concurrency controls.
- Prefetch tasks must be cancellable and bounded by semaphores/queue depth.
- Duplicate misses for the same expert must coalesce into one in-flight physical load.
- Authoritative cache hits may update recency metadata but must never change routing decisions.

## Correctness rule

A predictor can say "expert 12 will probably be needed next." It may start reading expert 12. When the real router runs, only the router's result can select the expert. A wrong prefetch costs I/O; it must never alter model output.

## Metrics to capture

Already exposed by the cache foundation:

- cache hits and misses;
- coalesced concurrent waits;
- host-cache evictions;
- physical expert bytes read.

Planned end-to-end metrics:

- tokens/sec and time-to-first-token;
- prefill vs decode time;
- bytes read per generated token;
- expert cache hit/miss ratio by tier;
- prefetch hit accuracy;
- cancelled prefetch bytes;
- average/p95/p99 expert read latency;
- NVMe queue depth;
- RAM and VRAM residency;
- CPU/GPU utilization;
- energy per token when platform telemetry is available.

## First adapter target criteria

Prefer a model that is:

- truly sparse MoE;
- small enough to iterate on a 16 GB laptop;
- publicly documented;
- available in a legal redistributable format or downloadable by the user;
- representative of larger routing behavior;
- already supported by a reference runtime for correctness comparison.
