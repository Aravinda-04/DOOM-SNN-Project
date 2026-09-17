"""Per-tensor symmetric quantization utilities for hardware export."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def integer_dtype(bits: int) -> torch.dtype:
    if not 2 <= bits <= 32:
        raise ValueError("bits must be between 2 and 32.")
    if bits <= 8:
        return torch.int8
    if bits <= 16:
        return torch.int16
    return torch.int32


def quantize_state_dict(
    state_dict: Mapping[str, torch.Tensor], bits: int
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Quantize floating tensors and return integer tensors with their scales."""
    qmax = 2 ** (bits - 1) - 1
    dtype = integer_dtype(bits)
    quantized = {}
    scales = {}

    for name, tensor in state_dict.items():
        value = tensor.detach().cpu()
        if not value.is_floating_point():
            quantized[name] = value.clone()
            continue
        if not torch.isfinite(value).all():
            raise ValueError(f"Cannot quantize non-finite tensor: {name}")

        max_absolute = float(value.abs().max()) if value.numel() else 0.0
        scale = max_absolute / qmax if max_absolute else 1.0
        quantized[name] = (
            torch.round(value / scale).clamp(-qmax, qmax).to(dtype)
        )
        scales[name] = scale

    return quantized, scales


def dequantize_state_dict(
    quantized: Mapping[str, torch.Tensor], scales: Mapping[str, float]
) -> dict[str, torch.Tensor]:
    """Reconstruct float tensors for numerical validation in PyTorch."""
    return {
        name: tensor.to(torch.float32) * scales[name]
        if name in scales
        else tensor.clone()
        for name, tensor in quantized.items()
    }

