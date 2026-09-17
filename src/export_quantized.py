from __future__ import annotations

import argparse
from pathlib import Path

import torch

from checkpointing import checkpoint_action_size, load_checkpoint
from eval import find_default_checkpoint
from networks import MODEL_NAMES, build_model
from quantization import dequantize_state_dict, quantize_state_dict
from runtime_utils import resolve_device, set_random_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a checkpoint as symmetric per-tensor integer weights."
    )
    parser.add_argument("--model", choices=MODEL_NAMES, default="snn")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive.")

    set_random_seeds(args.seed)
    device = resolve_device(args.device)
    source = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else find_default_checkpoint(args.model)
    )
    output = args.output.expanduser().resolve()

    action_size = checkpoint_action_size(source)
    original_model = build_model(args.model, action_size=action_size).to(device)
    checkpoint = load_checkpoint(source, original_model, device)
    checkpoint_model = checkpoint.get("config", {}).get("model")
    if checkpoint_model and checkpoint_model != args.model:
        raise ValueError(
            f"Checkpoint contains model '{checkpoint_model}', not '{args.model}'."
        )
    original_model.eval()

    quantized_state, scales = quantize_state_dict(
        original_model.state_dict(), args.bits
    )
    reconstructed_model = build_model(args.model, action_size=action_size).to(device)
    reconstructed_model.load_state_dict(
        dequantize_state_dict(quantized_state, scales)
    )
    reconstructed_model.eval()

    samples = torch.rand(args.samples, 1, 84, 84, device=device)
    with torch.no_grad():
        original_output, _ = original_model(samples)
        reconstructed_output, _ = reconstructed_model(samples)
    absolute_error = (original_output - reconstructed_output).abs()
    mean_error = float(absolute_error.mean())
    max_error = float(absolute_error.max())

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "symmetric_per_tensor",
            "bits": args.bits,
            "model": args.model,
            "source_checkpoint": str(source),
            "quantized_state_dict": quantized_state,
            "scales": scales,
            "validation": {
                "samples": args.samples,
                "mean_absolute_error": mean_error,
                "max_absolute_error": max_error,
            },
        },
        output,
    )
    print(f"Exported {args.bits}-bit integer checkpoint: {output}")
    print(f"Mean absolute output error: {mean_error:.6g}")
    print(f"Maximum absolute output error: {max_error:.6g}")
    print("This is an integer interchange file, not a vendor-specific FPGA image.")


if __name__ == "__main__":
    main()
