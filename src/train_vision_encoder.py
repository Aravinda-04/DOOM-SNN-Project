"""Multi-task training pipeline for the MAP01 Visual Perception Encoder."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from engine_state_config import FEATURE_INDEX
from networks.vision_encoder import VisionPerceptionEncoder
from project_paths import MODEL_ROOT, PROJECT_ROOT
from runtime_utils import resolve_device, set_random_seeds
from vision_dataset import VisionFrameDataset, vision_episode_splits


DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "generated" / "map01-vision-v1"
DEFAULT_CHECKPOINT_DIR = MODEL_ROOT / "vision_encoder"


def compute_perception_losses(
    outputs,
    features: torch.Tensor,
    validity: torch.Tensor,
    *,
    combat_weight: float = 1.0,
    nav_weight: float = 1.0,
    use_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute balanced multi-task losses with validity masking for invisible targets."""
    combat_raw = outputs.combat
    nav_raw = outputs.navigation
    inter_raw = outputs.interaction

    # Target extractions
    vis_target = features[:, FEATURE_INDEX["enemy_visible"]]
    cone_target = features[:, FEATURE_INDEX["enemy_in_firing_cone"]]
    scr_x_target = features[:, FEATURE_INDEX["enemy_screen_x"]]
    scr_area_target = features[:, FEATURE_INDEX["enemy_screen_area"]]
    dist_target = features[:, FEATURE_INDEX["enemy_distance"]]
    bearing_sin_target = features[:, FEATURE_INDEX["enemy_bearing_sin"]]
    bearing_cos_target = features[:, FEATURE_INDEX["enemy_bearing_cos"]]

    route_dist_target = features[:, FEATURE_INDEX["route_distance"]]
    route_sin_target = features[:, FEATURE_INDEX["route_bearing_sin"]]
    route_cos_target = features[:, FEATURE_INDEX["route_bearing_cos"]]
    err_sin_target = features[:, FEATURE_INDEX["route_heading_error_sin"]]
    err_cos_target = features[:, FEATURE_INDEX["route_heading_error_cos"]]

    use_target = features[:, FEATURE_INDEX["use_recommended"]]

    # 1. Combat visibility loss (BCE)
    loss_vis = F.binary_cross_entropy_with_logits(combat_raw[:, 0], vis_target)

    # 2. Combat attributes loss (masked to samples where enemy is actually visible)
    enemy_mask = vis_target > 0.5
    if enemy_mask.any():
        c_sub = combat_raw[enemy_mask]
        loss_scr_x = F.smooth_l1_loss(torch.sigmoid(c_sub[:, 1]), scr_x_target[enemy_mask])
        loss_scr_area = F.smooth_l1_loss(torch.sigmoid(c_sub[:, 2]), scr_area_target[enemy_mask])
        loss_dist = F.smooth_l1_loss(torch.clamp(c_sub[:, 3], 0.0, 1.0), dist_target[enemy_mask])

        # Cosine distance loss for bearing: 1.0 - cos(theta_pred - theta_gt)
        pred_bearing = F.normalize(c_sub[:, 4:6], p=2, dim=-1)
        gt_bearing = torch.stack([bearing_sin_target[enemy_mask], bearing_cos_target[enemy_mask]], dim=-1)
        cos_sim = (pred_bearing * gt_bearing).sum(dim=-1)
        loss_bearing = (1.0 - cos_sim).mean()

        loss_cone = F.binary_cross_entropy_with_logits(c_sub[:, 6], cone_target[enemy_mask])
        loss_combat = loss_vis + loss_scr_x + loss_scr_area + loss_dist + loss_bearing + loss_cone
    else:
        loss_combat = loss_vis

    # 3. Navigation losses
    loss_nav_dist = F.smooth_l1_loss(torch.clamp(nav_raw[:, 0], 0.0, 1.0), route_dist_target)

    pred_route_b = F.normalize(nav_raw[:, 1:3], p=2, dim=-1)
    gt_route_b = torch.stack([route_sin_target, route_cos_target], dim=-1)
    loss_route_b = (1.0 - (pred_route_b * gt_route_b).sum(dim=-1)).mean()

    pred_err = F.normalize(nav_raw[:, 3:5], p=2, dim=-1)
    gt_err = torch.stack([err_sin_target, err_cos_target], dim=-1)
    loss_err = (1.0 - (pred_err * gt_err).sum(dim=-1)).mean()

    loss_nav = loss_nav_dist + loss_route_b + loss_err

    # 4. Interaction loss (use_recommended BCE)
    loss_use = F.binary_cross_entropy_with_logits(inter_raw[:, 0], use_target)

    total_loss = (combat_weight * loss_combat) + (nav_weight * loss_nav) + (use_weight * loss_use)

    metrics = {
        "loss_total": float(total_loss.item()),
        "loss_combat": float(loss_combat.item()),
        "loss_nav": float(loss_nav.item()),
        "loss_use": float(loss_use.item()),
    }
    return total_loss, metrics


def train_vision_encoder(
    *,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    epochs: int = 25,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "auto",
    seed: int = 42,
) -> Path:
    set_random_seeds(seed)
    resolved_device = resolve_device(device)
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Splitting dataset from {dataset_dir}...")
    train_files, val_files = vision_episode_splits(dataset_dir, val_fraction=0.2, seed=seed)
    if not train_files:
        raise RuntimeError(f"No vision episodes found in {dataset_dir}. Run generate_vision_dataset.py first.")

    print(f"Loading datasets: {len(train_files)} train episodes, {len(val_files)} validation episodes...")
    train_dataset = VisionFrameDataset(dataset_dir, episode_files=train_files)
    val_dataset = VisionFrameDataset(dataset_dir, episode_files=val_files)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Initializing VisionPerceptionEncoder on {resolved_device}...")
    model = VisionPerceptionEncoder().to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_checkpoint = checkpoint_dir / "best_vision_encoder.pth"

    print(f"Starting multi-task training for {epochs} epochs ({len(train_dataset)} samples/epoch)...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            frames = batch["frames"].to(resolved_device)
            features = batch["features"].to(resolved_device)
            validity = batch["validity"].to(resolved_device)

            optimizer.zero_grad()
            outputs = model(frames)
            loss, _ = compute_perception_losses(outputs, features, validity)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"].to(resolved_device)
                features = batch["features"].to(resolved_device)
                validity = batch["validity"].to(resolved_device)
                outputs = model(frames)
                loss, _ = compute_perception_losses(outputs, features, validity)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses) if val_losses else avg_train
        lr = optimizer.param_groups[0]["lr"]

        improved = avg_val < best_val_loss
        if improved:
            best_val_loss = avg_val
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": avg_val,
                    "config": {
                        "in_channels": 4,
                        "embedding_dim": 256,
                    },
                },
                best_checkpoint,
            )

        marker = " (*)" if improved else ""
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {lr:.2e}{marker}")

    print(f"\nTraining completed! Best checkpoint saved to {best_checkpoint} (Val Loss: {best_val_loss:.4f})")
    return best_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-task Vision Perception Encoder.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_vision_encoder(
        dataset_dir=args.dataset_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
