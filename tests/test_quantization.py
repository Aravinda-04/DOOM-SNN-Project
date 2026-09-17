from __future__ import annotations

import torch

from quantization import dequantize_state_dict, quantize_state_dict
from diagnose import load_quantized_policy
from networks import build_model


def test_symmetric_quantization_produces_bounded_integer_tensors():
    state = {
        "weight": torch.tensor([-1.0, -0.2, 0.0, 0.7, 1.0]),
        "counter": torch.tensor(3, dtype=torch.int64),
    }

    quantized, scales = quantize_state_dict(state, bits=4)
    reconstructed = dequantize_state_dict(quantized, scales)

    assert quantized["weight"].dtype == torch.int8
    assert int(quantized["weight"].min()) >= -7
    assert int(quantized["weight"].max()) <= 7
    assert quantized["counter"].dtype == torch.int64
    assert torch.max(torch.abs(reconstructed["weight"] - state["weight"])) <= scales["weight"] / 2


def test_quantized_export_can_reconstruct_policy(tmp_path):
    source = build_model("snn", action_size=5)
    quantized, scales = quantize_state_dict(source.state_dict(), bits=8)
    path = tmp_path / "snn-int8.pt"
    torch.save(
        {
            "format": "symmetric_per_tensor",
            "bits": 8,
            "model": "snn",
            "quantized_state_dict": quantized,
            "scales": scales,
        },
        path,
    )

    restored, metadata, action_size = load_quantized_policy(
        path, "snn", torch.device("cpu")
    )

    assert action_size == 5
    assert metadata["bits"] == 8
    output, _ = restored(torch.rand(1, 1, 84, 84))
    assert output.shape == (1, 5)
