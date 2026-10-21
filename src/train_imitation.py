"""Sequence-based imitation pretraining for the canonical MAP01 CRSNN."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from agent_config import (
    ACTION_CATALOGUE_HASH,
    ACTION_NAMES,
    AGENT_CONTRACT,
    FRAME_STACK,
    PREPROCESSING_VERSION,
)
from checkpointing import load_checkpoint, save_checkpoint
from demonstrations import load_episode, sha256_file
from doom_level_env import DoomLevelEnvironment
from imitation_data import (
    DemonstrationSequenceDataset,
    action_class_counts,
    action_class_weights,
    successful_episode_splits,
)
from networks import build_model
from project_paths import LEVEL_CONFIG_PATH, PROJECT_ROOT, experiment_directories, make_run_id
from runtime_utils import resolve_device, set_random_seeds


DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "demonstrations" / "map01"


@dataclass(frozen=True)
class ImitationConfig:
    training_type: str
    model: str
    dataset_dir: str
    dataset_manifest_sha256: str
    agent_contract_hash: str
    preprocessing_version: int
    action_size: int
    action_names: tuple[str, ...]
    in_channels: int
    epochs: int
    batch_size: int
    sequence_length: int
    burn_in: int
    stride: int
    learning_rate: float
    sparsity_weight: float
    gradient_accumulation_steps: int
    validation_fraction: float
    balanced_sampling: bool
    class_weighting: bool
    sampling_minimum_support: int
    sampling_maximum_weight: float
    amp: bool
    seed: int
    run_id: str


def _detach_hidden(hidden: Optional[tuple]) -> Optional[tuple]:
    if hidden is None:
        return None
    return tuple(value.detach() for value in hidden)


def imitation_sequence_loss(
    model: torch.nn.Module,
    observations: torch.Tensor,
    actions: torch.Tensor,
    *,
    label_valid: Optional[torch.Tensor] = None,
    burn_in: int,
    class_weights: Optional[torch.Tensor],
    sparsity_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute recurrent behavior-cloning loss after no-gradient burn-in."""
    if observations.ndim != 5:
        raise ValueError("observations must have shape (B, T, C, H, W).")
    if actions.shape != observations.shape[:2]:
        raise ValueError("actions must have shape (B, T).")
    if label_valid is None:
        label_valid = torch.ones_like(actions, dtype=torch.bool)
    if label_valid.shape != actions.shape:
        raise ValueError("label_valid must have shape (B, T).")
    if not 0 <= burn_in < observations.shape[1]:
        raise ValueError("burn_in must be smaller than the sequence window.")

    hidden: Optional[tuple] = None
    # Keep recurrent context reconstruction in FP32 even when the caller wraps
    # the learning window in autocast. snntorch can return a detached FP16
    # hidden state from no-gradient burn-in that also disconnects subsequent
    # CRSNN outputs from autograd. The burn-in is small, carries no graph, and
    # only exists to establish temporal context, so FP32 is both safe and
    # inexpensive here. Exiting this block restores the caller's autocast state
    # for the gradient-bearing learning frames.
    with torch.no_grad(), torch.autocast(
        device_type=observations.device.type,
        enabled=False,
    ):
        for time_index in range(burn_in):
            _, _, hidden = model(observations[:, time_index], hidden)
    hidden = _detach_hidden(hidden)

    cross_entropy = torch.zeros((), device=observations.device)
    cross_entropy_weight = torch.zeros((), device=observations.device)
    total_spikes = torch.zeros((), device=observations.device)
    correct = torch.zeros((), device=observations.device)
    valid_examples = torch.zeros((), device=observations.device)
    learning_steps = observations.shape[1] - burn_in
    for time_index in range(burn_in, observations.shape[1]):
        logits, spikes, hidden = model(observations[:, time_index], hidden)
        targets = actions[:, time_index]
        mask = label_valid[:, time_index].bool()
        if mask.any():
            per_example_loss = F.cross_entropy(
                logits, targets, weight=class_weights, reduction="none"
            )
            cross_entropy = cross_entropy + per_example_loss[mask].sum()
            if class_weights is None:
                cross_entropy_weight = cross_entropy_weight + mask.sum()
            else:
                cross_entropy_weight = (
                    cross_entropy_weight + class_weights[targets[mask]].sum()
                )
            correct = correct + (
                logits.argmax(dim=1)[mask] == targets[mask]
            ).sum()
            valid_examples = valid_examples + mask.sum()
        total_spikes = total_spikes + spikes

    if valid_examples.item() == 0:
        raise ValueError("Sequence batch contains no valid learning labels.")
    if cross_entropy_weight.item() <= 0:
        raise ValueError("Valid learning labels have zero total class weight.")
    cross_entropy = cross_entropy / cross_entropy_weight
    opportunities = getattr(model, "spike_opportunities_per_sample", None)
    if opportunities is None:
        spike_rate = torch.zeros((), device=observations.device)
    elif opportunities <= 0:
        raise ValueError("spike_opportunities_per_sample must be positive.")
    else:
        spike_rate = total_spikes / (
            observations.shape[0] * learning_steps * opportunities
        )
    total_loss = cross_entropy + sparsity_weight * spike_rate
    metrics = {
        "cross_entropy": cross_entropy.detach(),
        "spike_rate": spike_rate.detach(),
        "correct": correct.detach(),
        "examples": valid_examples.detach(),
    }
    return total_loss, metrics


def evaluate_demonstrations(
    model: torch.nn.Module,
    dataset_dir: Path,
    entries: list[dict],
    device: torch.device,
) -> dict:
    """Evaluate complete held-out episodes while preserving recurrent state."""
    was_training = model.training
    model.eval()
    confusion = np.zeros(
        (AGENT_CONTRACT.action_size, AGENT_CONTRACT.action_size), dtype=np.int64
    )
    total_loss = 0.0
    total_examples = 0
    total_spike_rate = 0.0
    try:
        with torch.no_grad():
            for entry in entries:
                payload = load_episode(dataset_dir / entry["file"])
                observations = payload["observations"]
                actions = payload["actions"]
                label_valid = payload["label_valid"]
                hidden: Optional[tuple] = None
                for observation, target, is_valid in zip(
                    observations, actions, label_valid
                ):
                    tensor = torch.from_numpy(
                        observation.astype(np.float32) / 255.0
                    ).unsqueeze(0).to(device)
                    logits, spikes, hidden = model(tensor, hidden)
                    if not is_valid:
                        continue
                    target_tensor = torch.tensor([int(target)], device=device)
                    total_loss += float(
                        F.cross_entropy(logits, target_tensor).item()
                    )
                    prediction = int(logits.argmax(dim=1).item())
                    confusion[int(target), prediction] += 1
                    total_examples += 1
                    opportunities = getattr(
                        model, "spike_opportunities_per_sample", None
                    )
                    if opportunities:
                        total_spike_rate += float(spikes.item()) / opportunities
    finally:
        model.train(was_training)

    support = confusion.sum(axis=1)
    recalls: list[float | None] = []
    for action_index in range(AGENT_CONTRACT.action_size):
        if support[action_index] == 0:
            recalls.append(None)
        else:
            recalls.append(
                float(confusion[action_index, action_index] / support[action_index])
            )
    present_recalls = [value for value in recalls if value is not None]
    accuracy = float(np.trace(confusion) / max(total_examples, 1))
    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": accuracy,
        "macro_recall": float(np.mean(present_recalls)) if present_recalls else 0.0,
        "minimum_action_recall": (
            float(min(present_recalls)) if present_recalls else 0.0
        ),
        "per_action_recall": dict(zip(ACTION_NAMES, recalls)),
        "per_action_support": dict(
            zip(ACTION_NAMES, [int(value) for value in support])
        ),
        "confusion_matrix": confusion.tolist(),
        "mean_spike_rate": total_spike_rate / max(total_examples, 1),
        "examples": total_examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain CRSNN from MAP01 demonstrations.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--burn-in", type=int, default=4)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--sparsity-weight", type=float, default=1e-3)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Oversample recurrent windows containing supported rare actions.",
    )
    parser.add_argument(
        "--class-weighting",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Weight cross-entropy by action frequency. This cannot be combined "
            "with balanced sampling (default: disabled)."
        ),
    )
    parser.add_argument("--sampling-minimum-support", type=int, default=20)
    parser.add_argument("--sampling-maximum-weight", type=float, default=2.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA automatic mixed precision (default: enabled on CUDA).",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument(
        "--closed-loop-interval",
        type=int,
        default=5,
        help="Epochs between autonomous MAP01 checks; 0 disables them.",
    )
    parser.add_argument("--closed-loop-episodes", type=int, default=1)
    parser.add_argument(
        "--closed-loop-render",
        action="store_true",
        help="Open the ViZDoom window during closed-loop evaluation.",
    )
    parser.add_argument(
        "--closed-loop-delay",
        type=float,
        default=0.02,
        help="Seconds to pause per rendered evaluation decision (default: 0.02).",
    )
    parser.add_argument("--episode-timeout", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "stride": args.stride,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "checkpoint_interval": args.checkpoint_interval,
        "closed_loop_episodes": args.closed_loop_episodes,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.burn_in < 0:
        raise ValueError("--burn-in cannot be negative.")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.sparsity_weight < 0:
        raise ValueError("--sparsity-weight cannot be negative.")
    if args.sampling_minimum_support <= 0:
        raise ValueError("--sampling-minimum-support must be positive.")
    if args.sampling_maximum_weight < 1.0:
        raise ValueError("--sampling-maximum-weight must be at least 1.0.")
    if args.balanced_sampling and args.class_weighting:
        raise ValueError(
            "--balanced-sampling and --class-weighting cannot be combined; "
            "use only one imbalance correction strategy."
        )
    if args.closed_loop_interval < 0:
        raise ValueError("--closed-loop-interval cannot be negative.")
    if args.closed_loop_delay < 0:
        raise ValueError("--closed-loop-delay cannot be negative.")
    if args.episode_timeout is not None and args.episode_timeout <= 0:
        raise ValueError("--episode-timeout must be positive.")


def _closed_loop_evaluation(
    model: torch.nn.Module,
    device: torch.device,
    *,
    episodes: int,
    seed: int,
    episode_timeout: Optional[int],
    render: bool,
    render_delay: float,
) -> dict:
    from train_level import evaluate_policy

    return evaluate_policy(
        DoomLevelEnvironment,
        LEVEL_CONFIG_PATH,
        model,
        device,
        episodes=episodes,
        seeds=(seed,),
        episode_timeout=episode_timeout,
        progress=True,
        render=render,
        render_delay=render_delay,
    )


def main() -> None:
    args = parse_args()
    _validate_args(args)
    set_random_seeds(args.seed)
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    dataset_dir = args.dataset_dir.expanduser().resolve()
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Demonstration manifest not found: {manifest_path}. "
            "Record successful episodes with src/record_demonstrations.py first."
        )

    training_entries, validation_entries, manifest = successful_episode_splits(
        dataset_dir,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    training_dataset = DemonstrationSequenceDataset(
        dataset_dir,
        training_entries,
        sequence_length=args.sequence_length,
        burn_in=args.burn_in,
        stride=args.stride,
    )
    # Constructing this dataset verifies that held-out episodes can provide the
    # same recurrent window contract, even though evaluation uses whole episodes.
    validation_dataset = DemonstrationSequenceDataset(
        dataset_dir,
        validation_entries,
        sequence_length=args.sequence_length,
        burn_in=args.burn_in,
        stride=args.stride,
    )
    class_counts = action_class_counts(dataset_dir, training_entries)
    class_weights = (
        action_class_weights(dataset_dir, training_entries).to(device)
        if args.class_weighting
        else None
    )
    model = build_model(
        "crsnn",
        action_size=AGENT_CONTRACT.action_size,
        in_channels=FRAME_STACK,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loader_generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    sampling_weights = None
    if args.balanced_sampling:
        sampling_weights = training_dataset.balanced_sampling_weights(
            class_counts,
            minimum_support=args.sampling_minimum_support,
            maximum_weight=args.sampling_maximum_weight,
        )
        sampler = WeightedRandomSampler(
            sampling_weights,
            num_samples=len(training_dataset),
            replacement=True,
            generator=loader_generator,
        )
    train_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        generator=loader_generator,
        pin_memory=device.type == "cuda",
    )

    print(f"Device             : {device}")
    print(f"AMP                : {amp_enabled}")
    eligible_episodes = sum(
        bool(entry.get("completed") and entry.get("valid_action_contract"))
        for entry in manifest["episodes"]
    )
    print(f"Manifest episodes  : {len(manifest['episodes'])}")
    print(f"Eligible successes : {eligible_episodes}")
    print(f"Train/validation   : {len(training_entries)}/{len(validation_entries)}")
    print(f"Training windows   : {len(training_dataset)}")
    print(f"Validation windows : {len(validation_dataset)}")
    print(f"Balanced sampling  : {args.balanced_sampling}")
    print(
        "Loss weighting     : "
        + ("class weighted" if class_weights is not None else "unweighted")
    )
    if sampling_weights is not None:
        print(
            "Sampling weights   : "
            f"min={sampling_weights.min().item():.3f} "
            f"mean={sampling_weights.mean().item():.3f} "
            f"max={sampling_weights.max().item():.3f}"
        )
    if class_weights is not None:
        print(
            f"Action weights     : "
            f"{[round(v, 3) for v in class_weights.tolist()]}"
        )

    if args.validate_only:
        batch = next(iter(train_loader))
        observations = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        label_valid = batch["label_valid"].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            loss, _ = imitation_sequence_loss(
                model,
                observations,
                actions,
                label_valid=label_valid,
                burn_in=args.burn_in,
                class_weights=class_weights,
                sparsity_weight=args.sparsity_weight,
            )
        if not torch.isfinite(loss):
            raise RuntimeError("Validation batch produced a non-finite loss.")
        print(f"Validation passed with finite loss {loss.item():.6f}.")
        return

    run_id = args.run_id or f"imitation-{make_run_id(args.seed)}"
    model_dir, tensorboard_dir = experiment_directories("crsnn", run_id)
    if args.resume is None and (model_dir.exists() or tensorboard_dir.exists()):
        raise FileExistsError(
            f"Run '{run_id}' already exists; use another --run-id or --resume."
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    config = ImitationConfig(
        training_type="imitation",
        model="crsnn",
        dataset_dir=str(dataset_dir),
        dataset_manifest_sha256=sha256_file(manifest_path),
        agent_contract_hash=ACTION_CATALOGUE_HASH,
        preprocessing_version=PREPROCESSING_VERSION,
        action_size=AGENT_CONTRACT.action_size,
        action_names=ACTION_NAMES,
        in_channels=FRAME_STACK,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        burn_in=args.burn_in,
        stride=args.stride,
        learning_rate=args.learning_rate,
        sparsity_weight=args.sparsity_weight,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        validation_fraction=args.validation_fraction,
        balanced_sampling=args.balanced_sampling,
        class_weighting=args.class_weighting,
        sampling_minimum_support=args.sampling_minimum_support,
        sampling_maximum_weight=args.sampling_maximum_weight,
        amp=amp_enabled,
        seed=args.seed,
        run_id=run_id,
    )
    config_dict = asdict(config)
    start_epoch = 0
    global_step = 0
    best_accuracy = -math.inf
    best_macro_recall = -math.inf
    last_closed_loop: Optional[dict] = None
    if args.resume is not None:
        payload = load_checkpoint(
            args.resume,
            model,
            device,
            optimizer,
            expected_config={
                "training_type": "imitation",
                "model": "crsnn",
                "agent_contract_hash": ACTION_CATALOGUE_HASH,
                "preprocessing_version": PREPROCESSING_VERSION,
                "action_size": AGENT_CONTRACT.action_size,
                "in_channels": FRAME_STACK,
                "dataset_manifest_sha256": config.dataset_manifest_sha256,
                "batch_size": args.batch_size,
                "sequence_length": args.sequence_length,
                "burn_in": args.burn_in,
                "stride": args.stride,
                "learning_rate": args.learning_rate,
                "sparsity_weight": args.sparsity_weight,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "validation_fraction": args.validation_fraction,
                "balanced_sampling": args.balanced_sampling,
                "class_weighting": args.class_weighting,
                "sampling_minimum_support": args.sampling_minimum_support,
                "sampling_maximum_weight": args.sampling_maximum_weight,
                "amp": amp_enabled,
                "seed": args.seed,
                "run_id": run_id,
            },
        )
        start_epoch = int(payload.get("epoch", payload.get("episode", -1))) + 1
        global_step = int(payload.get("global_step", 0))
        best_accuracy = float(payload.get("best_eval_mean", -math.inf))
        best_macro_recall = float(payload.get("best_macro_recall", -math.inf))
        if "scaler_state_dict" in payload:
            scaler.load_state_dict(payload["scaler_state_dict"])
        if "data_loader_generator_state" in payload:
            loader_generator.set_state(
                payload["data_loader_generator_state"].cpu()
            )
        print(f"Resuming at epoch {start_epoch}, optimizer step {global_step}.")
    if start_epoch >= args.epochs:
        raise ValueError(
            f"Checkpoint already reached epoch {start_epoch}; increase --epochs."
        )

    (model_dir / "config.json").write_text(
        json.dumps(config_dict, indent=2), encoding="utf-8"
    )
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    last_completed_epoch = start_epoch - 1

    def save(path: Path, epoch: int, validation: dict) -> None:
        save_checkpoint(
            path,
            model,
            optimizer,
            episode=epoch,
            global_step=global_step,
            best_eval_mean=best_accuracy,
            best_eval_completion=(
                None if last_closed_loop is None
                else float(last_closed_loop["completion_rate"])
            ),
            config=config_dict,
            extra_state={
                "training_type": "imitation",
                "epoch": epoch,
                "best_macro_recall": best_macro_recall,
                "validation_metrics": validation,
                "closed_loop_metrics": last_closed_loop,
                "scaler_state_dict": scaler.state_dict(),
                "data_loader_generator_state": loader_generator.get_state(),
            },
        )

    try:
        for epoch in range(start_epoch, args.epochs):
            started_at = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            total_cross_entropy = 0.0
            total_spike_rate = 0.0
            total_correct = 0.0
            total_examples = 0.0

            for batch_index, batch in enumerate(train_loader):
                observations = batch["observations"].to(
                    device, non_blocking=device.type == "cuda"
                )
                actions = batch["actions"].to(
                    device, non_blocking=device.type == "cuda"
                )
                label_valid = batch["label_valid"].to(
                    device, non_blocking=device.type == "cuda"
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    loss, metrics = imitation_sequence_loss(
                        model,
                        observations,
                        actions,
                        label_valid=label_valid,
                        burn_in=args.burn_in,
                        class_weights=class_weights,
                        sparsity_weight=args.sparsity_weight,
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite imitation loss at epoch {epoch}, "
                        f"batch {batch_index}."
                    )
                group_start = (
                    batch_index // args.gradient_accumulation_steps
                ) * args.gradient_accumulation_steps
                group_size = min(
                    args.gradient_accumulation_steps,
                    len(train_loader) - group_start,
                )
                scaler.scale(loss / group_size).backward()
                should_step = (
                    (batch_index + 1) % args.gradient_accumulation_steps == 0
                    or batch_index + 1 == len(train_loader)
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                examples = float(metrics["examples"].item())
                total_loss += float(loss.detach().item()) * examples
                total_cross_entropy += (
                    float(metrics["cross_entropy"].item()) * examples
                )
                total_spike_rate += float(metrics["spike_rate"].item()) * examples
                total_correct += float(metrics["correct"].item())
                total_examples += examples

            train_metrics = {
                "loss": total_loss / max(total_examples, 1.0),
                "cross_entropy": total_cross_entropy / max(total_examples, 1.0),
                "accuracy": total_correct / max(total_examples, 1.0),
                "mean_spike_rate": total_spike_rate / max(total_examples, 1.0),
            }
            validation = evaluate_demonstrations(
                model, dataset_dir, validation_entries, device
            )
            if (
                args.closed_loop_interval > 0
                and (
                    (epoch + 1) % args.closed_loop_interval == 0
                    or epoch + 1 == args.epochs
                )
            ):
                print(
                    f"  -> Closed-loop MAP01 evaluation after epoch "
                    f"{epoch + 1}...",
                    flush=True,
                )
                last_closed_loop = _closed_loop_evaluation(
                    model,
                    device,
                    episodes=args.closed_loop_episodes,
                    seed=args.seed,
                    episode_timeout=args.episode_timeout,
                    render=args.closed_loop_render,
                    render_delay=args.closed_loop_delay,
                )
                print(
                    "     evaluation complete: "
                    f"completion={last_closed_loop['completion_rate']:.1%} "
                    f"reward={last_closed_loop['mean_reward']:+.2f} "
                    f"regions={last_closed_loop['mean_sectors_visited']:.1f}",
                    flush=True,
                )
                dominant_fraction = float(
                    last_closed_loop["dominant_action_fraction"]
                )
                if dominant_fraction >= 0.80:
                    dominant_action = max(
                        ACTION_NAMES,
                        key=lambda name: last_closed_loop[
                            f"action_fraction_{name}"
                        ],
                    )
                    print(
                        "     WARNING: action collapse detected: "
                        f"{dominant_action} accounts for "
                        f"{dominant_fraction:.1%} of evaluation decisions.",
                        flush=True,
                    )

            duration = time.perf_counter() - started_at
            print(
                f"[{epoch + 1:>3}/{args.epochs}] "
                f"train_acc={train_metrics['accuracy']:.1%} "
                f"val_acc={validation['accuracy']:.1%} "
                f"macro_recall={validation['macro_recall']:.1%} "
                f"loss={train_metrics['loss']:.4f} t={duration:.1f}s"
            )
            for name, value in train_metrics.items():
                writer.add_scalar(f"train/{name}", value, epoch)
            for name in (
                "loss",
                "accuracy",
                "macro_recall",
                "minimum_action_recall",
                "mean_spike_rate",
            ):
                writer.add_scalar(f"validation/{name}", validation[name], epoch)
            if last_closed_loop is not None:
                for name, value in last_closed_loop.items():
                    writer.add_scalar(f"closed_loop/{name}", value, epoch)

            is_best = (
                validation["accuracy"] > best_accuracy
                or (
                    validation["accuracy"] == best_accuracy
                    and validation["macro_recall"] > best_macro_recall
                )
            )
            if is_best:
                best_accuracy = float(validation["accuracy"])
                best_macro_recall = float(validation["macro_recall"])
                save(model_dir / "best.pth", epoch, validation)
            if (
                (epoch + 1) % args.checkpoint_interval == 0
                or epoch + 1 == args.epochs
            ):
                save(model_dir / "latest.pth", epoch, validation)
                writer.flush()
            last_completed_epoch = epoch
    except KeyboardInterrupt:
        interruption_metrics = {
            "interrupted": True,
            "last_completed_epoch": last_completed_epoch,
        }
        save(model_dir / "latest.pth", last_completed_epoch, interruption_metrics)
        writer.flush()
        print(f"Training interrupted; saved {model_dir / 'latest.pth'}.")
        raise
    finally:
        writer.close()

    print("Imitation training complete.")
    print(f"Best held-out accuracy    : {best_accuracy:.1%}")
    print(f"Best held-out macro recall: {best_macro_recall:.1%}")
    print(f"Best checkpoint           : {model_dir / 'best.pth'}")


if __name__ == "__main__":
    main()

