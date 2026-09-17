"""Save and load resumable and legacy model checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


CHECKPOINT_FORMAT_VERSION = 1


def checkpoint_action_size(path: Path) -> int:
    """Infer the policy output width before constructing its network."""
    checkpoint_path = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = payload.get("model_state_dict", payload)
    try:
        return int(state_dict["fc2.weight"].shape[0])
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(
            f"Cannot infer action size from checkpoint: {checkpoint_path}"
        ) from error


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    episode: int,
    global_step: int,
    best_eval_mean: float,
    config: dict[str, Any],
    best_eval_success: float | None = None,
    best_eval_noninstant_success: float | None = None,
    best_eval_hard_success: float | None = None,
) -> None:
    """Save enough state to evaluate or resume an experiment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode": episode,
        "global_step": global_step,
        "best_eval_mean": best_eval_mean,
        "config": config,
    }
    if best_eval_success is not None:
        payload["best_eval_success"] = best_eval_success
    if best_eval_noninstant_success is not None:
        payload["best_eval_noninstant_success"] = best_eval_noninstant_success
    if best_eval_hard_success is not None:
        payload["best_eval_hard_success"] = best_eval_hard_success
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load a structured checkpoint or a legacy raw state dictionary."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(payload, dict) and "model_state_dict" in payload:
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
