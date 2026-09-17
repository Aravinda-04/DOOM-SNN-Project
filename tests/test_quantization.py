from __future__ import annotations

import torch

from quantization import dequantize_state_dict, quantize_state_dict


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
