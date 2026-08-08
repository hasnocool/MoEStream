# TODO

## 0.1.x — Foundation

- [x] Create Rust workspace/runtime scaffold.
- [x] Define model adapter boundary.
- [x] Add thread-safe expert cache prototype.
- [x] Add OpenAI-compatible server skeleton.
- [x] Add CI, changelog, architecture and governance docs.
- [x] Add unit tests for cache range validation and concurrent access.
- [x] Replace whole-file prototype reads with bounded asynchronous range reads.
- [x] Coalesce concurrent requests for the same expert.
- [ ] Add bounded prefetch concurrency and cancellation.
- [x] Add LRU/ARC-style measured cache policy.
- [x] Add structured runtime metrics.

## 0.2.0 — First real model adapter

- [x] Survey practical 10B–100B sparse MoE models and choose Qwen3-30B-A3B as the reference target.
- [x] Parse and validate Qwen3 MoE configuration metadata.
- [x] Add a deterministic softmax/top-k routing reference helper for fixtures.
- [x] Add a versioned expert-bank manifest/index mapping `(layer, expert)` to validated byte ranges.
- [x] Validate manifest source paths, declared file sizes, duplicate expert IDs, and range bounds.
- [x] Verify declared source SHA-256 hashes against file contents without blocking Tokio worker threads.
- [x] Read and validate safetensors headers asynchronously without loading tensor payloads.
- [x] Resolve safetensors data-relative tensor offsets into absolute source-file spans.
- [x] Parse sharded `model.safetensors.index.json` files and inventory all checkpoint tensors.
- [x] Generate contiguous MoEStream expert-bank records from Qwen3 `gate_proj`, `up_proj`, and `down_proj` tensors.
- [x] Generate manifests automatically from supported Qwen3 safetensors checkpoints.
- [x] Load and decode Qwen3 `mlp.gate.weight` router tensors from BF16/F32 safetensors spans.
- [x] Add non-blocking reference router matrix-vector execution using the decoded tensor.
- [ ] Verify exact PyTorch `topk` tie behavior, router dtype/accumulation semantics, and router output parity.
- [ ] Promote the verified router path to authoritative production routing.
- [ ] Implement tokenizer/chat template.
- [ ] Implement expert tensor decoding.
- [ ] Add CPU execution provider.
- [ ] Compare token IDs/logits against a reference runtime.
- [ ] Benchmark RAM, bytes/token, NVMe throughput and tokens/sec.

See [`docs/FIRST-MODEL-TARGET.md`](docs/FIRST-MODEL-TARGET.md) for the target rationale and verification plan.

## 0.3.0 — Predictive storage scheduling

- [ ] Record authoritative route traces.
- [ ] Implement next-layer expert prediction as scheduling-only hints.
- [ ] Cancel losing prefetches after authoritative routing.
- [ ] Add route-frequency based offline cache warming.
- [ ] Measure prediction accuracy and read amplification.

## 0.4.0 — Accelerator residency

- [ ] Add Vulkan provider/residency experiments.
- [ ] Add CUDA provider/residency experiments.
- [ ] Add Metal provider/residency experiments.
- [ ] Reserve explicit accelerator headroom.
- [ ] Read only cache misses when device experts remain resident.

## 0.5.0 — Speculative decoding

- [ ] Define draft-model interface.
- [ ] Add exact target verification transactions.
- [ ] Benchmark coding autocomplete and chat independently.
- [ ] Ensure target model remains authoritative for emitted tokens.

## Later

- [ ] Persistent conversation/KV state.
- [ ] Multi-model hot switching.
- [ ] Optional remote cold expert store.
- [ ] Power-aware scheduling for laptops/off-grid systems.
- [ ] OpencodeSmart backend integration.
