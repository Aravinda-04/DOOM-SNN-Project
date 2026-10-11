"""Episode-safe sequence loading for CRSNN imitation training."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from agent_config import AGENT_CONTRACT
from demonstrations import load_episode, split_episode_entries, validate_manifest


def successful_episode_splits(
    dataset_dir: Path,
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[dict], list[dict], dict]:
    """Validate a dataset and split only complete, contract-valid episodes."""
    manifest = validate_manifest(dataset_dir)
    training, validation = split_episode_entries(
        manifest["episodes"],
        validation_fraction=validation_fraction,
        seed=seed,
        successful_only=True,
    )
    return training, validation, manifest


class DemonstrationSequenceDataset(Dataset):
    """Fixed recurrent windows indexed without splitting episode membership."""

    def __init__(
        self,
        dataset_dir: Path,
        entries: Sequence[dict],
        *,
        sequence_length: int,
        burn_in: int,
        stride: int | None = None,
    ):
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive.")
        if burn_in < 0:
            raise ValueError("burn_in cannot be negative.")
        self.dataset_dir = dataset_dir.expanduser().resolve()
        self.entries = [dict(entry) for entry in entries]
        self.sequence_length = sequence_length
        self.burn_in = burn_in
        self.window_length = sequence_length + burn_in
        self.stride = stride if stride is not None else sequence_length
        if self.stride <= 0:
            raise ValueError("stride must be positive.")

        self.windows: list[tuple[str, int, str]] = []
        for entry in self.entries:
            steps = int(entry["steps"])
            if steps < self.window_length:
                continue
            last_start = steps - self.window_length
            starts = list(range(0, last_start + 1, self.stride))
            if starts[-1] != last_start:
                starts.append(last_start)
            self.windows.extend(
                (entry["file"], start, entry["episode_id"])
                for start in starts
            )
        if not self.windows:
            raise ValueError(
                "No demonstration episode is long enough for burn_in + "
                "sequence_length."
            )

    @lru_cache(maxsize=8)
    def _load(self, filename: str) -> dict:
        return load_episode(self.dataset_dir / filename)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        filename, start, episode_id = self.windows[index]
        payload = self._load(filename)
        stop = start + self.window_length
        observations = (
            payload["observations"][start:stop].astype(np.float32) / 255.0
        )
        actions = payload["actions"][start:stop].astype(np.int64)
        if np.any(actions < 0):
            raise ValueError(
                f"Eligible episode {episode_id} unexpectedly has invalid labels."
            )
        return {
            "observations": torch.from_numpy(observations),
            "actions": torch.from_numpy(actions),
            "episode_id": episode_id,
            "start": start,
        }


def action_class_weights(dataset_dir: Path, entries: Sequence[dict]) -> torch.Tensor:
    """Return normalized inverse-frequency weights for actions that occur."""
    counts = np.zeros(AGENT_CONTRACT.action_size, dtype=np.int64)
    for entry in entries:
        payload = load_episode(dataset_dir / entry["file"])
        actions = payload["actions"]
        if np.any(actions < 0):
            raise ValueError(
                f"Episode {entry['episode_id']} has invalid action labels."
            )
        counts += np.bincount(actions, minlength=AGENT_CONTRACT.action_size)
    weights = np.zeros_like(counts, dtype=np.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / counts[present].astype(np.float32)
    if np.any(present):
        weights[present] /= weights[present].mean()
    return torch.from_numpy(weights)

