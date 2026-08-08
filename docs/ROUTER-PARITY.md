# Qwen3 Router Parity

## Purpose

MoEStream must not call its Qwen3 routing path authoritative until the decoded checkpoint router produces reference-compatible routing decisions.

The first parity layer is intentionally small and deterministic. It isolates router semantics from tokenizer, attention, expert execution, KV-cache, and storage scheduling so failures can be localized.

## Reference sequence

For Qwen3-30B-A3B, the reference routing sequence is:

1. Read `model.layers.{layer}.mlp.gate.weight`.
2. Apply the router linear projection to the layer hidden state.
3. Convert router logits to float32 for softmax.
4. Select `num_experts_per_tok` with `topk`.
5. If `norm_topk_prob` is enabled, divide selected probabilities by their selected-weight sum.
6. Use the selected expert IDs as the authoritative model route.

The official Qwen3-30B-A3B configuration uses BF16 weights, 128 routed experts, top-8 routing, and `norm_topk_prob=true`.

## Recorded fixture

`tests/fixtures/qwen3_router_pytorch_2_10_cpu.json` records a small BF16 reference case produced with PyTorch 2.10.0 CPU.

The fixture contains:

- BF16 router-weight bit patterns;
- BF16 hidden-state bit patterns;
- BF16 router-logit bit patterns;
- float32 router logits after conversion;
- float32 softmax probabilities;
- top-k expert indices;
- selected probabilities before normalization;
- selected probabilities after normalization.

The fixture can be regenerated with:

```bash
python scripts/generate_qwen3_router_fixture.py
```

The generator is intentionally offline tooling. PyTorch is not a runtime dependency of MoEStream.

## Current parity guarantee

The Rust integration test reconstructs the BF16 safetensors router, decodes it through the production checkpoint reader, replays the recorded hidden state, and checks:

- exact router logits for this fixture;
- identical non-tied top-k expert indices;
- selected pre-normalization weights within `1e-6`;
- selected normalized weights within `1e-6`.

This proves the current small BF16 CPU fixture agrees with its recorded PyTorch reference. It does **not** yet prove parity for every real Qwen3 layer or every hardware backend.

## Tie behavior is not a model contract

PyTorch does not guarantee bitwise-identical floating-point results across releases and platforms. Ordering among equal top-k values therefore cannot safely be treated as a portable model invariant.

MoEStream may keep deterministic behavior for its own fixtures, but it must not claim a universal PyTorch tie-order guarantee. Production parity should be judged on non-tied routes and on real-checkpoint fixtures where selected probabilities have a meaningful separation.

If an exact tie occurs in a reference comparison, report it separately instead of silently declaring one implementation wrong solely because equivalent experts were returned in a different order.

## Remaining promotion gate

Before the router is labeled production-authoritative:

1. Capture at least one hidden state from the real `Qwen/Qwen3-30B-A3B` checkpoint at a chosen layer.
2. Record the reference BF16 router logits, float32 softmax, top-k indices, and normalized weights.
3. Run that fixture through MoEStream's actual checkpoint inventory and router-weight loader.
4. Confirm the selected expert IDs match and numerical differences remain within documented tolerances.
5. Repeat on the initial CPU execution provider after its matrix kernels are introduced.

Only after those checks should runtime code rely on this router path for model-authoritative expert selection.
