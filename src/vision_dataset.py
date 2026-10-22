"""PyTorch Dataset implementations for offline MAP01 vision perception training."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from engine_state_config import FEATURE_INDEX, FEATURE_NAMES
from vision_data_schema import load_vision_manifest, load_vision_sequence


class VisionAugmentation:
    """Fast, vectorized visual augmentations for 4-frame grayscale stacks."""

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (4, 84, 84) in [0, 1]
        out = frames.clone()
        # 1. Random brightness scaling
        if torch.rand(1).item() > 0.3:
            scale = 0.85 + torch.rand(1).item() * 0.30
            out = out * scale

        # 2. Random contrast adjustment
        if torch.rand(1).item() > 0.4:
            mean = out.mean()
            contrast = 0.80 + torch.rand(1).item() * 0.40
            out = (out - mean) * contrast + mean

        # 3. Random Gaussian pixel noise
        if torch.rand(1).item() > 0.5:
            noise = torch.randn_like(out) * 0.02
            out = out + noise

        # 4. Random cutout (erase 12x12 square)
        if torch.rand(1).item() > 0.5:
            h, w = out.shape[1], out.shape[2]
            ch, cw = 12, 12
            top = torch.randint(0, h - ch, (1,)).item()
            left = torch.randint(0, w - cw, (1,)).item()
            out[:, top : top + ch, left : left + cw] = 0.0

        return torch.clamp(out, 0.0, 1.0)


class VisionFrameDataset(Dataset):
    """Dataset providing individual transitions (frame_stack, features, validity, targets).

    Optimized for multi-task feed-forward Visual Perception Encoder training.
    """

    def __init__(
        self,
        dataset_dir: Path,
        *,
        episode_files: Sequence[str] | None = None,
        transform=None,
        augment: bool = False,
    ):
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        manifest = load_vision_manifest(self.dataset_dir)
        available = {entry["file"]: entry for entry in manifest["episodes"]}

        if episode_files is None:
            self.selected_files = [entry["file"] for entry in manifest["episodes"]]
        else:
            self.selected_files = [f for f in episode_files if f in available]

        self.transform = transform
        self.augment = augment
        self._augmentor = VisionAugmentation() if augment else None
        self._samples: list[tuple[int, int]] = []  # (file_idx, step_idx)
        self._episodes: list[dict[str, np.ndarray]] = []

        for file_idx, filename in enumerate(self.selected_files):
            filepath = self.dataset_dir / filename
            data = load_vision_sequence(filepath)
            self._episodes.append(data)
            num_steps = len(data["frames"])
            for step_idx in range(num_steps):
                self._samples.append((file_idx, step_idx))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_idx, step_idx = self._samples[index]
        ep = self._episodes[file_idx]

        # Convert uint8 (4, 84, 84) to float32 [0.0, 1.0]
        frames = ep["frames"][step_idx].astype(np.float32) / 255.0
        features = ep["features"][step_idx].astype(np.float32)
        validity = ep["feature_valid"][step_idx].astype(np.bool_)
        targets = ep["targets"][step_idx].astype(np.int64)

        frames_tensor = torch.from_numpy(frames)
        if self._augmentor is not None:
            frames_tensor = self._augmentor(frames_tensor)

        sample = {
            "frames": frames_tensor,
            "features": torch.from_numpy(features),
            "validity": torch.from_numpy(validity),
            "targets": torch.from_numpy(targets),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class VisionSequenceDataset(Dataset):
    """Dataset providing continuous temporal sequence slices for recurrent training."""

    def __init__(
        self,
        dataset_dir: Path,
        sequence_length: int = 32,
        stride: int = 16,
        *,
        episode_files: Sequence[str] | None = None,
    ):
        if sequence_length <= 0 or stride <= 0:
            raise ValueError("sequence_length and stride must be positive.")
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)

        manifest = load_vision_manifest(self.dataset_dir)
        available = {entry["file"]: entry for entry in manifest["episodes"]}

        if episode_files is None:
            self.selected_files = [entry["file"] for entry in manifest["episodes"]]
        else:
            self.selected_files = [f for f in episode_files if f in available]

        self._slices: list[tuple[int, int, int]] = []  # (file_idx, start, end)
        self._episodes: list[dict[str, np.ndarray]] = []

        for file_idx, filename in enumerate(self.selected_files):
            filepath = self.dataset_dir / filename
            data = load_vision_sequence(filepath)
            self._episodes.append(data)
            num_steps = len(data["frames"])
            if num_steps < self.sequence_length:
                continue
            for start in range(0, num_steps - self.sequence_length + 1, self.stride):
                self._slices.append((file_idx, start, start + self.sequence_length))

    def __len__(self) -> int:
        return len(self._slices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_idx, start, end = self._slices[index]
        ep = self._episodes[file_idx]

        # Slices: (T, 4, 84, 84)
        frames = ep["frames"][start:end].astype(np.float32) / 255.0
        features = ep["features"][start:end].astype(np.float32)
        validity = ep["feature_valid"][start:end].astype(np.bool_)
        targets = ep["targets"][start:end].astype(np.int64)

        return {
            "frames": torch.from_numpy(frames),
            "features": torch.from_numpy(features),
            "validity": torch.from_numpy(validity),
            "targets": torch.from_numpy(targets),
        }


def vision_episode_splits(
    dataset_dir: Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Partition dataset episode filenames into train and validation sets."""
    manifest = load_vision_manifest(dataset_dir)
    filenames = sorted(entry["file"] for entry in manifest["episodes"])
    if not filenames:
        return [], []
    rng = np.random.default_rng(seed)
    shuffled = list(filenames)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    val_files = sorted(shuffled[:val_count])
    train_files = sorted(shuffled[val_count:])
    return train_files, val_files
