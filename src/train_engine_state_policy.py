"""Behavior cloning for the privileged recurrent MAP01 policy."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from engine_state_config import ENGINE_STATE_CONTRACT_HASH, FEATURE_INDEX, FEATURE_NAMES
from engine_state_data import (
    EngineStateSequenceDataset,
    episode_splits,
    target_counts,
)
from evaluate_engine_state_policy import evaluate_policy
from networks.engine_state_policy import (
    EngineStateGRUPolicy,
    EngineStateMLPPolicy,
    EngineStateSpikingPolicy,
)
from project_paths import MODEL_ROOT, PROJECT_ROOT, RUN_ROOT, make_run_id
from runtime_utils import resolve_device, set_random_seeds


DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "generated" / "map01-engine-state-v2"
DEFAULT_ROUTE_DATA = PROJECT_ROOT / "data" / "demonstrations" / "map01"


def _multiclass_weights(counts: torch.Tensor, maximum: float = 4.0) -> torch.Tensor:
    counts = counts.float()
    weights = torch.zeros_like(counts)
    present = counts > 0
    if present.any():
        reference = counts[present].max()
        weights[present] = torch.sqrt(reference / counts[present]).clamp(max=maximum)
        weights[present] /= (weights[present] * counts[present]).sum() / counts[present].sum()
    return weights


def _positive_weight(counts: torch.Tensor, maximum: float = 10.0) -> torch.Tensor:
    negative, positive = (float(value) for value in counts)
    if positive <= 0:
        return torch.tensor(1.0)
    return torch.tensor(min(math.sqrt(negative / positive), maximum))


def policy_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    *,
    locomotion_weights: torch.Tensor,
    turning_weights: torch.Tensor,
    attack_positive_weight: torch.Tensor,
    use_positive_weight: torch.Tensor,
    recovery_mask: torch.Tensor | None = None,
    recovery_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    locomotion = F.cross_entropy(
        outputs["locomotion"].reshape(-1, 5),
        targets[..., 0].reshape(-1),
        weight=locomotion_weights,
    )
    turning = F.cross_entropy(
        outputs["turning"].reshape(-1, 3),
        targets[..., 1].reshape(-1),
        weight=turning_weights,
    )
    attack = F.binary_cross_entropy_with_logits(
        outputs["attack"].reshape(-1),
        targets[..., 2].float().reshape(-1),
        pos_weight=attack_positive_weight,
    )
    use = F.binary_cross_entropy_with_logits(
        outputs["use"].reshape(-1),
        targets[..., 3].float().reshape(-1),
        pos_weight=use_positive_weight,
    )
    total = locomotion + turning + attack + use
    losses = {
        "locomotion": locomotion.detach(),
        "turning": turning.detach(),
        "attack": attack.detach(),
        "use": use.detach(),
    }
    if (
        recovery_mask is not None
        and recovery_loss_weight > 0
        and bool(recovery_mask.any())
    ):
        recovery = (
            F.cross_entropy(outputs["locomotion"][recovery_mask], targets[..., 0][recovery_mask])
            + F.cross_entropy(outputs["turning"][recovery_mask], targets[..., 1][recovery_mask])
            + F.binary_cross_entropy_with_logits(
                outputs["attack"][recovery_mask],
                targets[..., 2][recovery_mask].float(),
            )
            + F.binary_cross_entropy_with_logits(
                outputs["use"][recovery_mask],
                targets[..., 3][recovery_mask].float(),
            )
        )
        total = total + recovery_loss_weight * recovery
        losses["recovery"] = recovery.detach()
    return total, losses


def _update_confusions(
    confusions: dict[str, np.ndarray],
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
) -> None:
    predictions = {
        "locomotion": outputs["locomotion"].argmax(-1),
        "turning": outputs["turning"].argmax(-1),
        "attack": (outputs["attack"] >= 0).long(),
        "use": (outputs["use"] >= 0).long(),
    }
    columns = {"locomotion": 0, "turning": 1, "attack": 2, "use": 3}
    for name, prediction in predictions.items():
        truth = targets[..., columns[name]].reshape(-1).detach().cpu().numpy()
        predicted = prediction.reshape(-1).detach().cpu().numpy()
        np.add.at(confusions[name], (truth, predicted), 1)


def _metrics(confusions: dict[str, np.ndarray]) -> dict:
    result: dict[str, float | dict] = {}
    all_recalls: list[float] = []
    for name, confusion in confusions.items():
        support = confusion.sum(axis=1)
        recalls = [
            (float(confusion[i, i] / support[i]) if support[i] else None)
            for i in range(len(support))
        ]
        present = [value for value in recalls if value is not None]
        all_recalls.extend(present)
        result[f"{name}_accuracy"] = float(
            np.trace(confusion) / max(confusion.sum(), 1)
        )
        result[f"{name}_recall"] = recalls
        result[f"{name}_confusion"] = confusion.tolist()
    result["macro_recall"] = float(np.mean(all_recalls)) if all_recalls else 0.0
    result["attack_recall"] = float(result["attack_recall"][1] or 0.0)  # type: ignore[index]
    result["use_recall"] = float(result["use_recall"][1] or 0.0)  # type: ignore[index]
    return result


@torch.no_grad()
def evaluate_offline(
    model: EngineStateGRUPolicy,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    confusions = {
        "locomotion": np.zeros((5, 5), dtype=np.int64),
        "turning": np.zeros((3, 3), dtype=np.int64),
        "attack": np.zeros((2, 2), dtype=np.int64),
        "use": np.zeros((2, 2), dtype=np.int64),
    }
    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        outputs, _ = model(inputs)
        _update_confusions(confusions, outputs, targets)
    return _metrics(confusions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MAP01 engine-state policy.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--route-data", type=Path, default=DEFAULT_ROUTE_DATA)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--maximum-windows-per-episode", type=int, default=64)
    parser.add_argument(
        "--counterfactual-ammo-probability",
        type=float,
        default=0.35,
        help="Fraction of training windows relabeled for visible enemies with no ammo.",
    )
    parser.add_argument("--recovery-window-repeats", type=int, default=4)
    parser.add_argument("--recovery-loss-weight", type=float, default=4.0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--architecture", choices=("gru", "mlp", "snn"), default="mlp")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--closed-loop-interval", type=int, default=5)
    parser.add_argument("--closed-loop-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--closed-loop-render", action="store_true")
    parser.add_argument("--episode-timeout", type=int, default=None)
    parser.add_argument("--overfit-windows", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--minimum-unshielded-completion-rate",
        type=float,
        default=0.67,
        help="Required raw-policy completion rate across closed-loop seeds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in (
        "epochs", "batch_size", "sequence_length", "stride",
        "maximum_windows_per_episode", "hidden_size", "num_layers",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if not 0.0 <= args.counterfactual_ammo_probability <= 1.0:
        raise ValueError("--counterfactual-ammo-probability must be in [0, 1].")
    if args.recovery_window_repeats <= 0:
        raise ValueError("--recovery-window-repeats must be positive.")
    if args.recovery_loss_weight < 0:
        raise ValueError("--recovery-loss-weight cannot be negative.")
    if args.closed_loop_interval < 0 or args.overfit_windows < 0:
        raise ValueError("Intervals and overfit window count cannot be negative.")
    if not 0.0 <= args.minimum_unshielded_completion_rate <= 1.0:
        raise ValueError("--minimum-unshielded-completion-rate must be in [0, 1].")

    set_random_seeds(args.seed)
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    dataset_dir = args.dataset_dir.expanduser().resolve()
    training_entries, validation_entries, manifest = episode_splits(
        dataset_dir,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    training_dataset = EngineStateSequenceDataset(
        dataset_dir,
        training_entries,
        sequence_length=args.sequence_length,
        stride=args.stride,
        maximum_windows_per_episode=args.maximum_windows_per_episode,
        counterfactual_ammo_probability=args.counterfactual_ammo_probability,
        recovery_window_repeats=args.recovery_window_repeats,
    )
    validation_dataset = EngineStateSequenceDataset(
        dataset_dir,
        validation_entries,
        sequence_length=args.sequence_length,
        stride=args.stride,
        maximum_windows_per_episode=args.maximum_windows_per_episode,
        counterfactual_ammo_probability=0.0,
        recovery_window_repeats=1,
    )
    if args.overfit_windows:
        count = min(args.overfit_windows, len(training_dataset))
        training_dataset = Subset(training_dataset, range(count))
        validation_dataset = training_dataset

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    counts_source = (
        training_dataset.dataset
        if isinstance(training_dataset, Subset)
        else training_dataset
    )
    counts = target_counts(counts_source)
    locomotion_weights = _multiclass_weights(counts["locomotion"]).to(device)
    turning_weights = _multiclass_weights(counts["turning"]).to(device)
    attack_positive_weight = _positive_weight(counts["attack"]).to(device)
    use_positive_weight = _positive_weight(counts["use"], maximum=20.0).to(device)

    input_size = len(FEATURE_NAMES) * 2
    policy_classes = {
        "gru": EngineStateGRUPolicy,
        "mlp": EngineStateMLPPolicy,
        "snn": EngineStateSpikingPolicy,
    }
    policy_class = policy_classes[args.architecture]
    model = policy_class(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    print(f"Device             : {device}")
    print(f"AMP                : {amp_enabled}")
    print(f"Episodes           : {len(manifest['episodes'])}")
    print(f"Train/validation   : {len(training_entries)}/{len(validation_entries)}")
    print(f"Training windows   : {len(training_dataset)}")
    print(f"Validation windows : {len(validation_dataset)}")
    print(f"Target counts      : { {k: v.tolist() for k, v in counts.items()} }")
    print(
        "Positive weights   : "
        f"attack={attack_positive_weight.item():.2f} "
        f"use={use_positive_weight.item():.2f}"
    )

    if args.validate_only:
        batch = next(iter(train_loader))
        outputs, _ = model(batch["inputs"].to(device))
        loss, _ = policy_loss(
            outputs,
            batch["targets"].to(device),
            locomotion_weights=locomotion_weights,
            turning_weights=turning_weights,
            attack_positive_weight=attack_positive_weight,
            use_positive_weight=use_positive_weight,
        )
        loss.backward()
        print(f"Validation passed with differentiable loss {loss.item():.6f}.")
        return

    run_id = args.run_id or f"map01-engine-state-{make_run_id(args.seed)}"
    model_dir = MODEL_ROOT / "engine_state" / run_id
    run_dir = RUN_ROOT / "engine_state" / run_id
    if args.resume is None and (model_dir.exists() or run_dir.exists()):
        raise FileExistsError(f"Run already exists: {run_id}")
    model_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["dataset_dir"] = str(dataset_dir)
    config["route_data"] = str(args.route_data.expanduser().resolve())
    config["resume"] = None if args.resume is None else str(args.resume)
    config["engine_state_contract_hash"] = ENGINE_STATE_CONTRACT_HASH
    config["input_size"] = input_size
    (model_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    start_epoch = 0
    best_score = -math.inf
    best_completion_rate = 0.0
    if args.resume is not None:
        payload = torch.load(args.resume, map_location=device, weights_only=True)
        if payload.get("engine_state_contract_hash") != ENGINE_STATE_CONTRACT_HASH:
            raise ValueError("Resume checkpoint uses a different state contract.")
        if payload.get("architecture", "gru") != args.architecture:
            raise ValueError("Resume checkpoint uses a different architecture.")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate
        start_epoch = int(payload["epoch"]) + 1
        if args.resume.expanduser().resolve().parent == model_dir:
            best_score = float(payload.get("best_score", -math.inf))
            best_completion_rate = float(payload.get("best_completion_rate", 0.0))

    def save(path: Path, epoch: int, metrics: dict, score: float) -> None:
        torch.save(
            {
                "format_version": 1,
                "training_type": "engine_state_imitation",
                "architecture": args.architecture,
                "engine_state_contract_hash": ENGINE_STATE_CONTRACT_HASH,
                "input_size": input_size,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "epoch": epoch,
                "best_score": best_score,
                "best_completion_rate": best_completion_rate,
                "metrics": metrics,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            path,
        )

    for epoch in range(start_epoch, args.epochs):
        started = time.perf_counter()
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in train_loader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs, _ = model(inputs)
                loss, _ = policy_loss(
                    outputs,
                    targets,
                    locomotion_weights=locomotion_weights,
                    turning_weights=turning_weights,
                    attack_positive_weight=attack_positive_weight,
                    use_positive_weight=use_positive_weight,
                    recovery_mask=(
                        inputs[..., FEATURE_INDEX["stuck_fraction"]] > 0.15
                    ),
                    recovery_loss_weight=args.recovery_loss_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            batches += 1

        validation = evaluate_offline(model, validation_loader, device)
        score = float(validation["macro_recall"])
        duration = time.perf_counter() - started
        print(
            f"[{epoch + 1:3d}/{args.epochs}] loss={total_loss / max(batches, 1):.4f} "
            f"macro={score:.1%} attack={validation['attack_recall']:.1%} "
            f"use={validation['use_recall']:.1%} t={duration:.1f}s",
            flush=True,
        )

        closed_results: list[dict] = []
        if args.closed_loop_interval and (
            (epoch + 1) % args.closed_loop_interval == 0
            or epoch + 1 == args.epochs
        ):
            print(f"  -> closed-loop evaluation after epoch {epoch + 1}", flush=True)
            for closed_seed in args.closed_loop_seeds:
                closed = evaluate_policy(
                    model,
                    device=device,
                    route_data=args.route_data.expanduser().resolve(),
                    seed=closed_seed,
                    render=args.closed_loop_render,
                    render_delay=0.0,
                    episode_timeout=args.episode_timeout,
                    progress_interval=0,
                    use_control_shield=False,
                )
                closed_results.append(closed)
                print(
                    f"     seed={closed_seed} completed={closed['completed']} "
                    f"route={closed['route_progress']:.1%} kills={closed['kills']} "
                    f"regions={closed['regions']} terminal={closed['terminal_reason']}",
                    flush=True,
                )
            completion_rate = float(
                np.mean([result["completed"] for result in closed_results])
            )
            mean_progress = float(
                np.mean([result["route_progress"] for result in closed_results])
            )
            # Closed-loop behavior dominates checkpoint selection. Offline
            # recall only breaks ties between equally capable policies.
            closed_score = completion_rate * 100.0 + mean_progress + score * 0.01
            if closed_score > best_score:
                best_score = closed_score
                best_completion_rate = completion_rate
                checkpoint_metrics = dict(validation)
                checkpoint_metrics["closed_loop"] = closed_results
                checkpoint_metrics["completion_rate"] = completion_rate
                checkpoint_metrics["mean_progress"] = mean_progress
                save(model_dir / "best.pth", epoch, checkpoint_metrics, closed_score)
            print(
                f"     unshielded completion={completion_rate:.1%} "
                f"mean_route={mean_progress:.1%}",
                flush=True,
            )
        save(model_dir / "latest.pth", epoch, validation, score)

    print("Engine-state imitation training complete.")
    gate_passed = best_completion_rate >= args.minimum_unshielded_completion_rate
    print(f"Best unshielded completion : {best_completion_rate:.1%}")
    print(f"Autonomy gate              : {'PASSED' if gate_passed else 'FAILED'}")
    print(f"Best checkpoint   : {model_dir / 'best.pth'}")
    if not gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
