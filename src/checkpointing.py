"""Save and load resumable and legacy model checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_FORMAT_VERSION = 3


def model_state_signature(model: torch.nn.Module) -> str:
    """Return a stable hash of state names, tensor shapes, and dtypes."""
    description = "\n".join(
        f"{name}|{tuple(tensor.shape)}|{tensor.dtype}"
        for name, tensor in model.state_dict().items()
    )
    return hashlib.sha256(description.encode("utf-8")).hexdigest()


def checkpoint_action_size(path: Path) -> int:
    """Infer the policy output width before constructing its network."""
    checkpoint_path = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = payload.get("model_state_dict", payload)
    for output_weight in ("fc2.weight", "fc.weight"):
        tensor = state_dict.get(output_weight) if isinstance(state_dict, dict) else None
        if tensor is not None and hasattr(tensor, "shape") and len(tensor.shape) == 2:
            return int(tensor.shape[0])
    raise ValueError(f"Cannot infer action size from checkpoint: {checkpoint_path}")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    episode: int,
    global_step: int,
    best_eval_mean: float,
    config: dict[str, Any],
    best_eval_completion: float | None = None,
    best_eval_success: float | None = None,
    best_eval_noninstant_success: float | None = None,
    best_eval_hard_success: float | None = None,
    best_eval_worst_direction: float | None = None,
    extra_state: dict[str, Any] | None = None,
) -> None:
    """Save enough state to evaluate or resume an experiment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_signature": model_state_signature(model),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode": episode,
        "global_step": global_step,
        "best_eval_mean": best_eval_mean,
        "config": config,
    }
    if best_eval_success is not None:
        payload["best_eval_success"] = best_eval_success
    if best_eval_completion is not None:
        payload["best_eval_completion"] = best_eval_completion
    if best_eval_noninstant_success is not None:
        payload["best_eval_noninstant_success"] = best_eval_noninstant_success
    if best_eval_hard_success is not None:
        payload["best_eval_hard_success"] = best_eval_hard_success
    if best_eval_worst_direction is not None:
        payload["best_eval_worst_direction"] = best_eval_worst_direction
    if extra_state:
        collisions = sorted(set(payload) & set(extra_state))
        if collisions:
            raise ValueError(
                f"extra_state cannot replace core checkpoint fields: {collisions}"
            )
        payload.update(extra_state)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    expected_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a structured checkpoint or a legacy raw state dictionary."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        saved_signature = payload.get("model_signature")
        expected_signature = model_state_signature(model)
        if saved_signature is not None and saved_signature != expected_signature:
            raise ValueError(
                "Cannot load checkpoint: model architecture signature differs "
                f"(saved {saved_signature}, expected {expected_signature})."
            )
        saved_config = payload.get("config", {})
        for key, expected in (expected_config or {}).items():
            if key in saved_config and saved_config[key] != expected:
                raise ValueError(
                    f"Cannot load checkpoint: saved {key}={saved_config[key]!r}, "
                    f"expected {expected!r}."
                )
        model.load_state_dict(payload["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        return payload

    # Older versions of this project saved model.state_dict() directly.
    model.load_state_dict(payload)
    return {
        "format_version": 0,
        "episode": -1,
        "global_step": 0,
        "best_eval_mean": -float("inf"),
        "config": {},
    }
