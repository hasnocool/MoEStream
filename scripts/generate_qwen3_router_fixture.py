# scripts/generate_qwen3_router_fixture.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

SCHEMA_VERSION = 1


def bf16_bits(tensor: torch.Tensor) -> list[int]:
    tensor = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    return [int(value) for value in tensor.view(torch.uint16).reshape(-1).tolist()]


def generate_fixture() -> dict[str, object]:
    weights_f32 = torch.tensor(
        [
            [0.5, -1.0, 0.25, 2.0],
            [-0.75, 0.5, 1.5, -0.25],
            [1.0, 0.25, -0.5, 0.75],
            [-1.25, 1.0, 0.5, 0.5],
            [0.125, -0.375, 0.625, 1.125],
            [0.875, 0.75, -1.0, 0.375],
        ],
        dtype=torch.float32,
    )
    hidden_f32 = torch.tensor([0.75, -1.25, 0.5, 1.75], dtype=torch.float32)

    weights = weights_f32.to(torch.bfloat16)
    hidden = hidden_f32.to(torch.bfloat16)
    router_logits = torch.nn.functional.linear(hidden, weights)
    router_logits_f32 = router_logits.float()
    probabilities = torch.softmax(router_logits_f32, dim=-1)
    topk = 3
    topk_weights, topk_indices = torch.topk(probabilities, topk, dim=-1)
    normalized = topk_weights / topk_weights.sum()

    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "scripts/generate_qwen3_router_fixture.py",
        "pytorch_version": torch.__version__,
        "device": "cpu",
        "dtype": "bfloat16",
        "shape": list(weights.shape),
        "weights_bf16_bits": bf16_bits(weights),
        "hidden_bf16_bits": bf16_bits(hidden),
        "router_logits_bf16_bits": bf16_bits(router_logits),
        "router_logits_f32": [float(value) for value in router_logits_f32.tolist()],
        "softmax_f32": [float(value) for value in probabilities.tolist()],
        "topk": topk,
        "topk_indices": [int(value) for value in topk_indices.tolist()],
        "topk_weights_pre_norm": [float(value) for value in topk_weights.tolist()],
        "topk_weights_norm": [float(value) for value in normalized.tolist()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the deterministic PyTorch CPU Qwen3 router parity fixture."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/qwen3_router_pytorch_2_10_cpu.json"),
    )
    args = parser.parse_args()

    fixture = generate_fixture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} with PyTorch {torch.__version__}")


if __name__ == "__main__":
    main()
