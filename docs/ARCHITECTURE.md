# Architecture

## Objective

MoEStream separates **model semantics** from **storage/runtime policy** so multiple sparse model families can share checkpoint discovery, expert indexing, caching, prefetch, metrics, and API infrastructure.

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

### 3. Checkpoint discovery and preparation

MoEStream must inspect and prepare large upstream checkpoints without materializing complete model shards or expert tensors in RAM.

The safetensors metadata reader therefore:

- reads the 8-byte little-endian JSON-header length asynchronously;
- caps accepted header size before allocation;
- reads only the JSON header, never tensor payload bytes during discovery;
- records dtype, shape, and data-relative `[begin, end)` offsets;
- converts data-relative offsets into checked absolute source-file spans;
- rejects malformed JSON, reversed ranges, out-of-bounds tensors, holes, and trailing unindexed data.

The sharded checkpoint inventory parses `model.safetensors.index.json`, validates safe relative shard paths, loads each referenced safetensors header asynchronously, and cross-checks the index against actual shard contents. A logical tensor can then be resolved to a source shard plus an absolute byte span without reading its payload.

For canonical Hugging Face Qwen3-MoE checkpoints, an expert uses three source tensors:

```text
model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight
model.layers.{layer}.mlp.experts.{expert}.up_proj.weight
model.layers.{layer}.mlp.experts.{expert}.down_proj.weight
```

The Qwen3 preparation stage validates each projection's dtype and expected `[out_features, in_features]` shape, then copies the exact source bytes through a bounded asynchronous buffer in deterministic `gate_proj → up_proj → down_proj` order. It never decodes, transposes, quantizes, or otherwise changes tensor payload bytes.

Each logical expert becomes one contiguous bank record. The generated manifest keeps projection-level subranges containing name, dtype, shape, relative offset, and length. This preserves one logical expert read for runtime storage scheduling while retaining enough information for later tensor decoding.

Preparation writes `.partial` bank/manifest files, hashes the finished bank with synchronous reading and CPU hashing isolated on Tokio's blocking pool, validates the generated manifest, and only then renames the files into their published names. Failed preparations remove partial output.

### 4. Expert manifest and index

The manifest is the trust boundary between model conversion and runtime expert loading. It maps logical `(layer, expert)` identities onto explicit byte ranges in declared source files.

The current schema validates before index construction:

- schema version and model identity;
- source files referenced by stable manifest keys;
- paths that are relative to the configured model root and contain no traversal components;
- non-zero declared source sizes;
- optional SHA-256 metadata syntax;
- unique `(layer, expert)` identities;
- non-zero expert spans;
- checked offset/length arithmetic;
- spans bounded by each source file's declared size;
- optional expert tensor subranges with unique names, non-zero lengths, bounded offsets, and exact contiguous coverage of the expert record.

The resulting `ExpertIndex` resolves an `ExpertId` to the existing `ExpertLocation` structure consumed by the cache and retains its `ManifestExpert` tensor metadata for future decoding. Filesystem type/size checks use asynchronous Tokio APIs.

For sources with declared SHA-256 digests, `ExpertIndex::verify_declared_hashes()` performs full content verification. The method size-checks sources first, then uses the reusable `sha256_file()` helper, which isolates synchronous file reads and SHA-256 CPU work with `tokio::task::spawn_blocking`. Hashing uses a reusable 1 MiB buffer, so source files are streamed rather than loaded wholesale into RAM.

### 5. Expert scheduler

The scheduler will combine:

- current authoritative route;
- predicted next-layer route;
- cache residency snapshot;
- I/O queue depth;
- available RAM/VRAM;
- measured expert reuse probability;
- cancellation of losing prefetches.

Authoritative expert loads already use bounded I/O concurrency. Predictive prefetch cancellation remains a separate scheduling milestone so speculative reads cannot accidentally affect correctness.

### 6. Tiered cache

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

### 7. Execution provider

Compute backends are intentionally separate from storage scheduling. Planned providers:

- CPU: AVX2/AVX-512 where available;
- Vulkan for broad laptop GPU support;
- CUDA for NVIDIA;
- Metal for Apple Silicon.

## Concurrency rules

- Async request paths must not perform blocking disk or network operations.
- Checkpoint metadata discovery must use asynchronous file I/O and must not read tensor payloads unnecessarily.
- Checkpoint payload repacking must use bounded asynchronous range reads/writes rather than whole-tensor allocations.
- CPU-bound tensor work that cannot be asynchronous must execute in a dedicated compute pool/provider, not on Tokio worker threads.
- Shared cache metadata must use bounded, thread-safe synchronization.
- Physical expert reads must be limited by explicit concurrency controls.
- Manifest and source-file filesystem checks must use asynchronous file APIs.
- Full-file integrity hashing must run outside Tokio worker threads and use bounded-memory streaming buffers.
- Prefetch tasks must be cancellable and bounded by semaphores/queue depth.
- Duplicate misses for the same expert must coalesce into one in-flight physical load.
- Authoritative cache hits may update recency metadata but must never change routing decisions.

## Correctness rule

A predictor can say "expert 12 will probably be needed next." It may start reading expert 12. When the real router runs, only the router's result can select the expert. A wrong prefetch costs I/O; it must never alter model output.

A manifest may say where expert 12 resides, but it must never decide that expert 12 participates in the token. Routing identity and storage location remain separate concerns.

A declared integrity hash is also not advisory: when present and verification is requested, a mismatch is a hard failure before the source is trusted.

Checkpoint conversion may rearrange tensor byte spans into a storage-optimized expert record, but the default preparation path must preserve each source tensor's dtype, shape, element order, and bytes exactly. Quantization, transposition, or any numerical conversion must be an explicit future mode with separate correctness validation.

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
