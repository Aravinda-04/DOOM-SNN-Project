"""Sequence datasets and split helpers for privileged MAP01 training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from engine_state_demonstrations import load_sequence, validate_manifest


def episode_splits(
    dataset_dir: Path,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    manifest = validate_manifest(dataset_dir)
    entries = list(manifest["episodes"])
    if len(entries) < 2:
        raise ValueError("At least two generated episodes are required.")
    random.Random(seed).shuffle(entries)
    validation_count = max(1, round(len(entries) * validation_fraction))
    validation_count = min(validation_count, len(entries) - 1)
    return entries[validation_count:], entries[:validation_count], manifest


class EngineStateSequenceDataset(Dataset):
    """Fixed recurrent windows with a cap preventing one timeout dominating."""

    def __init__(
        self,
        dataset_dir: Path,
        entries: Sequence[dict],
        *,
        sequence_length: int,
        stride: int,
        maximum_windows_per_episode: int = 64,
    ):
        if sequence_length <= 0 or stride <= 0:
            raise ValueError("sequence_length and stride must be positive.")
        if maximum_windows_per_episode <= 0:
            raise ValueError("maximum_windows_per_episode must be positive.")
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.sequence_length = int(sequence_length)
        self._episodes: list[dict] = []
        self._windows: list[tuple[int, int]] = []
        for entry in entries:
            payload = load_sequence(self.dataset_dir / entry["file"])
            length = len(payload["features"])
            if length < sequence_length:
                continue
            episode_index = len(self._episodes)
            self._episodes.append(payload)
            starts = np.arange(0, length - sequence_length + 1, stride)
            if len(starts) > maximum_windows_per_episode:
                selected = np.linspace(
                    0, len(starts) - 1, maximum_windows_per_episode
                ).round().astype(int)
                starts = starts[selected]
            self._windows.extend((episode_index, int(start)) for start in starts)
        if not self._windows:
            raise ValueError("No generated episode can provide a full sequence window.")

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, start = self._windows[index]
        stop = start + self.sequence_length
        payload = self._episodes[episode_index]
        features = payload["features"][start:stop]
        validity = payload["feature_valid"][start:stop].astype(np.float32)
        inputs = np.concatenate((features, validity), axis=1).astype(np.float32)
        return {
            "inputs": torch.from_numpy(inputs),
            "targets": torch.from_numpy(
                payload["targets"][start:stop].astype(np.int64)
            ),
        }


def target_counts(dataset: EngineStateSequenceDataset) -> dict[str, torch.Tensor]:
    locomotion = torch.zeros(5, dtype=torch.long)
    turning = torch.zeros(3, dtype=torch.long)
    attack = torch.zeros(2, dtype=torch.long)
    use = torch.zeros(2, dtype=torch.long)
    for episode in dataset._episodes:
        targets = torch.from_numpy(episode["targets"].astype(np.int64))
        locomotion += torch.bincount(targets[:, 0], minlength=5)
        turning += torch.bincount(targets[:, 1], minlength=3)
        attack += torch.bincount(targets[:, 2], minlength=2)
        use += torch.bincount(targets[:, 3], minlength=2)
    return {
        "locomotion": locomotion,
        "turning": turning,
        "attack": attack,
        "use": use,
    }
