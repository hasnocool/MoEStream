# First Model Target: Qwen3-30B-A3B

Date selected: 2026-08-07

## Decision

MoEStream's first production-oriented model adapter target is **Qwen3-30B-A3B**.

Official model card: https://huggingface.co/Qwen/Qwen3-30B-A3B

Official config: https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json

Reference implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py

## Why this model

Qwen3-30B-A3B provides a useful balance between total model size and active compute:

- 30.5B total parameters;
- 3.3B activated parameters;
- 48 transformer layers;
- 128 experts;
- 8 experts activated per token;
- 32 query heads and 4 key/value heads;
- Apache-2.0 model license;
- existing Hugging Face Transformers support for `Qwen3MoeForCausalLM`.

This makes it a much better first storage-streaming target than models whose active parameter count is itself too large for rapid laptop iteration.

## Why not Qwen3.5/Qwen3.6 first

Newer Qwen3.5/Qwen3.6 MoE checkpoints are available, including A3B-class models. They are not the first implementation target because their text architecture adds substantially more moving parts: hybrid Gated DeltaNet/attention layers, shared-expert behavior, and a multimodal model family. Those are useful later compatibility targets, but they would make it harder to isolate storage, routing, and expert-kernel correctness during the first adapter bring-up.

MoEStream therefore intentionally starts with Qwen3-30B-A3B even though it is not the newest Qwen family. This is an engineering-scope decision, not a claim that Qwen3 is newer or stronger than Qwen3.5/Qwen3.6.

Reference: https://huggingface.co/docs/transformers/en/model_doc/qwen3_5_moe

## Why not Kimi K3 first

Kimi K3 remains an important long-term stress target, but its scale makes basic correctness iteration expensive. MoEStream should establish the adapter ABI, expert-bank format, route verification, CPU tensor path, and storage metrics on a smaller sparse model before attempting multi-terabyte-class expert corpora.

## Why not Mixtral 8x7B first

Mixtral 8x7B is well documented and permissively licensed, but it has approximately 45B total parameters and compute comparable to roughly a 14B dense model because two experts are selected per token. It is useful as a secondary compatibility target, but Qwen3-30B-A3B's approximately 3.3B activated parameter footprint is more appropriate for the first laptop benchmark cycle.

Reference: https://huggingface.co/docs/transformers/en/model_doc/mixtral

## Adapter implementation plan

### Phase A — metadata and tokenizer

1. Parse and validate the Qwen3 MoE config.
2. Load tokenizer metadata without requiring the complete expert bank in RAM.
3. Implement the exact chat template and tokenization behavior.
4. Add fixture tests against Hugging Face token IDs.

Configuration parsing and validation are implemented in `src/adapters/qwen3.rs` as of `0.2.0-alpha.1`.

### Phase B — authoritative routing

1. Implement Qwen3 MoE router tensor decoding.
2. Implement exact top-8 expert selection and normalized routing weights.
3. Record route decisions for deterministic fixture inputs.
4. Compare route IDs and weights against a reference Transformers run.

A correctness-oriented helper for already-computed router logits is implemented in `0.2.0-alpha.1`. It follows the reference sequence of float32 softmax, top-k selection, and optional selected-weight normalization. It deliberately uses deterministic expert-ID tie ordering for fixtures; production routing will not claim parity until exact reference-runtime tie behavior is tested.

### Phase C — expert storage format

1. Build an offline index mapping `(layer, expert)` to exact byte ranges.
2. Keep shared/resident tensors separate from streamed expert tensors.
3. Validate tensor dtype, shape, offset, length, and source-file identity.
4. Add checksums to generated expert-bank manifests.

### Phase D — CPU execution

1. Implement the minimum tensor operators required for Qwen3 MoE decode.
2. Keep CPU-bound tensor work outside Tokio executor threads.
3. Start with correctness-first scalar/reference kernels where necessary.
4. Add AVX2/AVX-512 optimized kernels only after reference parity exists.

### Phase E — verification

For fixed prompts and fixed model weights, compare:

- tokenizer IDs;
- per-layer routed expert IDs;
- routing weights;
- selected intermediate activations;
- final logits;
- generated token IDs.

No performance optimization is accepted if it changes authoritative routing or output beyond an explicitly documented numeric-tolerance boundary.

## First benchmark matrix

Capture at minimum:

- model storage bytes;
- resident RAM before generation;
- peak RAM;
- physical expert bytes read/token;
- cache hit ratio;
- coalesced expert requests;
- NVMe read throughput;
- p50/p95 expert read latency;
- prefill tokens/sec;
- decode tokens/sec;
- CPU utilization;
- energy/token where platform telemetry is available.

## Secondary targets

After Qwen3-30B-A3B is correct:

1. Qwen3.5/Qwen3.6 A3B-class MoE — newer hybrid/shared-expert compatibility target.
2. Mixtral 8x7B — mature reference architecture and top-2 routing.
3. Larger Qwen MoE variants — validates scaling with similar semantics.
4. Kimi K3 — extreme storage-streaming stress target.
5. Other sparse architectures only after their routing and license terms are independently verified.
