"""Sequence datasets and split helpers for privileged MAP01 training."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from engine_state_config import FEATURE_INDEX
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
        counterfactual_ammo_probability: float = 0.0,
        recovery_window_repeats: int = 1,
    ):
        if sequence_length <= 0 or stride <= 0:
            raise ValueError("sequence_length and stride must be positive.")
        if maximum_windows_per_episode <= 0:
            raise ValueError("maximum_windows_per_episode must be positive.")
        if not 0.0 <= counterfactual_ammo_probability <= 1.0:
            raise ValueError("counterfactual_ammo_probability must be in [0, 1].")
        if recovery_window_repeats <= 0:
            raise ValueError("recovery_window_repeats must be positive.")
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.sequence_length = int(sequence_length)
        self.counterfactual_ammo_probability = float(
            counterfactual_ammo_probability
        )
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
            targets = payload["targets"]
            features = payload["features"]
            priority_mask = (
                (targets[:, 0] >= 2)
                | (targets[:, 2] == 1)
                | (targets[:, 3] == 1)
                | (features[:, FEATURE_INDEX["route_off_track"]] > 0.5)
                | (features[:, FEATURE_INDEX["stuck_fraction"]] > 0.15)
            )
            priority_steps = np.flatnonzero(priority_mask)
            priority_starts = np.unique(
                np.clip(
                    priority_steps - sequence_length // 2,
                    0,
                    length - sequence_length,
                )
            )
            if len(starts) + len(priority_starts) > maximum_windows_per_episode:
                priority_budget = min(
                    len(priority_starts), maximum_windows_per_episode // 2
                )
                uniform_budget = maximum_windows_per_episode - priority_budget
                uniform_indices = np.linspace(
                    0, len(starts) - 1, uniform_budget
                ).round().astype(int)
                selected_priority = (
                    np.linspace(0, len(priority_starts) - 1, priority_budget)
                    .round()
                    .astype(int)
                    if priority_budget
                    else np.asarray([], dtype=int)
                )
                starts = np.unique(
                    np.concatenate(
                        (starts[uniform_indices], priority_starts[selected_priority])
                    )
                )
            else:
                starts = np.unique(np.concatenate((starts, priority_starts)))
            self._windows.extend((episode_index, int(start)) for start in starts)
            if recovery_window_repeats > 1:
                stuck_steps = np.flatnonzero(
                    features[:, FEATURE_INDEX["stuck_fraction"]] > 0.15
                )
                recovery_starts = np.unique(
                    np.clip(
                        stuck_steps - sequence_length // 2,
                        0,
                        length - sequence_length,
                    )
                )
                if len(recovery_starts) > 16:
                    recovery_starts = recovery_starts[
                        np.linspace(0, len(recovery_starts) - 1, 16)
                        .round()
                        .astype(int)
                    ]
                for _ in range(recovery_window_repeats - 1):
                    self._windows.extend(
                        (episode_index, int(start)) for start in recovery_starts
                    )
        if not self._windows:
            raise ValueError("No generated episode can provide a full sequence window.")

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, start = self._windows[index]
        stop = start + self.sequence_length
        payload = self._episodes[episode_index]
        features = payload["features"][start:stop].copy()
        validity = payload["feature_valid"][start:stop].astype(np.float32)
        targets = payload["targets"][start:stop].astype(np.int64).copy()
        threshold = int(round(self.counterfactual_ammo_probability * 10_000))
        counterfactual = ((index * 2654435761 + 1013904223) % 10_000) < threshold
        if counterfactual:
            features[:, FEATURE_INDEX["ammo_bullets"]] = 0.0
            features[:, FEATURE_INDEX["ammo_shells"]] = 0.0
            enemy_visible = features[:, FEATURE_INDEX["enemy_visible"]] > 0.5
            use_recommended = features[:, FEATURE_INDEX["use_recommended"]] > 0.5
            affected = enemy_visible & ~use_recommended
            for timestep in np.flatnonzero(affected):
                sine = float(
                    features[timestep, FEATURE_INDEX["route_heading_error_sin"]]
                )
                cosine = float(
                    features[timestep, FEATURE_INDEX["route_heading_error_cos"]]
                )
                error = math.degrees(math.atan2(sine, cosine))
                targets[timestep, 0] = 0 if abs(error) > 12.0 else 1
                targets[timestep, 1] = 1 if error > 4.0 else 2 if error < -4.0 else 0
                targets[timestep, 2] = 0
        inputs = np.concatenate((features, validity), axis=1).astype(np.float32)
        return {
            "inputs": torch.from_numpy(inputs),
            "targets": torch.from_numpy(targets),
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
