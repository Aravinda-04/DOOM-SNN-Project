"""Versioned storage helpers for recurrent MAP01 demonstrations."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from agent_config import (
    ACTION_CATALOGUE_HASH,
    AGENT_CONTRACT,
    FRAME_HEIGHT,
    FRAME_STACK,
    FRAME_WIDTH,
)


DEMONSTRATION_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
REQUIRED_ARRAYS = (
    "observations",
    "actions",
    "rewards",
    "raw_rewards",
    "dones",
    "positions",
    "game_times",
    "metadata_json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observation_to_uint8(observation: np.ndarray) -> np.ndarray:
    """Encode one normalized model observation without changing its shape."""
    array = np.asarray(observation)
    expected = (FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH)
    if array.shape != expected:
        raise ValueError(f"Expected observation shape {expected}, got {array.shape}.")
    if array.dtype == np.uint8:
        return np.array(array, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError("Observation contains NaN or infinite values.")
    if float(array.min()) < -1e-6 or float(array.max()) > 1.0 + 1e-6:
        raise ValueError("Normalized observations must be in [0, 1].")
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def _validate_payload(
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    raw_rewards: np.ndarray,
    dones: np.ndarray,
    positions: np.ndarray,
    game_times: np.ndarray,
    metadata: dict,
) -> None:
    if observations.ndim != 4 or observations.shape[1:] != (
        FRAME_STACK,
        FRAME_HEIGHT,
        FRAME_WIDTH,
    ):
        raise ValueError(
            "observations must have shape "
            f"(T, {FRAME_STACK}, {FRAME_HEIGHT}, {FRAME_WIDTH})."
        )
    if observations.dtype != np.uint8:
        raise ValueError("observations must be stored as uint8.")
    length = observations.shape[0]
    if length == 0:
        raise ValueError("A demonstration episode cannot be empty.")
    expected_vectors = {
        "actions": (actions, (length,)),
        "rewards": (rewards, (length,)),
        "raw_rewards": (raw_rewards, (length,)),
        "dones": (dones, (length,)),
        "game_times": (game_times, (length,)),
        "positions": (positions, (length, 3)),
    }
    for name, (array, expected_shape) in expected_vectors.items():
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {array.shape}."
            )
    if not np.issubdtype(actions.dtype, np.integer):
        raise ValueError("actions must have an integer dtype.")
    if np.any(actions < -1) or np.any(actions >= AGENT_CONTRACT.action_size):
        raise ValueError("actions contain an index outside the frozen catalogue.")
    if not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(raw_rewards)):
        raise ValueError("Reward arrays must be finite.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("Position metadata must be finite.")
    if metadata.get("schema_version") != DEMONSTRATION_SCHEMA_VERSION:
        raise ValueError("Demonstration metadata has an incompatible schema version.")
    if metadata.get("agent_contract_sha256") != ACTION_CATALOGUE_HASH:
        raise ValueError("Demonstration uses a different agent/action contract.")
    valid_actions = bool(np.all(actions >= 0))
    if bool(metadata.get("valid_action_contract")) != valid_actions:
        raise ValueError(
            "valid_action_contract does not agree with the recorded action labels."
        )
    if bool(metadata.get("completed")) and not bool(dones[-1]):
        raise ValueError("A completed demonstration must end with done=True.")


def save_episode(
    path: Path,
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    raw_rewards: np.ndarray,
    dones: np.ndarray,
    positions: np.ndarray,
    game_times: np.ndarray,
    metadata: dict,
) -> str:
    """Atomically save one compressed episode and return its SHA-256."""
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite episode: {path}")
    _validate_payload(
        observations=observations,
        actions=actions,
        rewards=rewards,
        raw_rewards=raw_rewards,
        dones=dones,
        positions=positions,
        game_times=game_times,
        metadata=metadata,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                observations=observations,
                actions=actions,
                rewards=rewards,
                raw_rewards=raw_rewards,
                dones=dones,
                positions=positions,
                game_times=game_times,
                metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
            )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return sha256_file(path)


def load_episode(path: Path) -> dict:
    """Load and validate an episode file without allowing pickled objects."""
    path = path.expanduser().resolve()
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_ARRAYS) - set(archive.files))
        if missing:
            raise ValueError(f"Episode is missing arrays: {missing}")
        payload = {name: np.array(archive[name], copy=True) for name in REQUIRED_ARRAYS}
    metadata = json.loads(str(payload.pop("metadata_json").item()))
    _validate_payload(metadata=metadata, **payload)
    payload["metadata"] = metadata
    return payload


def episode_summary(path: Path, *, successful_only: bool = False) -> dict:
    payload = load_episode(path)
    metadata = payload["metadata"]
    if successful_only and not (
        metadata.get("completed") and metadata.get("valid_action_contract")
    ):
        raise ValueError("Episode is not a valid successful demonstration.")
    return {
        "episode_id": metadata["episode_id"],
        "file": Path(path).name,
        "sha256": sha256_file(Path(path)),
        "completed": bool(metadata["completed"]),
        "valid_action_contract": bool(metadata["valid_action_contract"]),
        "terminal_reason": metadata["terminal_reason"],
        "steps": int(payload["actions"].shape[0]),
        "created_at": metadata["created_at"],
        "notes": metadata.get("notes", ""),
    }


def _new_manifest() -> dict:
    return {
        "schema_version": DEMONSTRATION_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "agent_contract": AGENT_CONTRACT.to_dict(),
        "agent_contract_sha256": ACTION_CATALOGUE_HASH,
        "episodes": [],
    }


def load_manifest(dataset_dir: Path) -> dict:
    path = dataset_dir.expanduser().resolve() / MANIFEST_FILENAME
    if not path.exists():
        return _new_manifest()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DEMONSTRATION_SCHEMA_VERSION:
        raise ValueError("Manifest schema version is incompatible.")
    if manifest.get("agent_contract_sha256") != ACTION_CATALOGUE_HASH:
        raise ValueError("Manifest uses a different agent/action contract.")
    return manifest


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def append_manifest_entry(dataset_dir: Path, entry: dict) -> Path:
    """Append one unique episode summary to the versioned manifest."""
    dataset_dir = dataset_dir.expanduser().resolve()
    manifest = load_manifest(dataset_dir)
    episode_ids = {item["episode_id"] for item in manifest["episodes"]}
    files = {item["file"] for item in manifest["episodes"]}
    if entry["episode_id"] in episode_ids or entry["file"] in files:
        raise ValueError("Manifest already contains this episode or filename.")
    manifest["episodes"].append(dict(entry))
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    path = dataset_dir / MANIFEST_FILENAME
    _write_json_atomic(path, manifest)
    return path


def validate_manifest(dataset_dir: Path) -> dict:
    """Verify every manifest checksum and episode/summary relationship."""
    dataset_dir = dataset_dir.expanduser().resolve()
    manifest = load_manifest(dataset_dir)
    for entry in manifest["episodes"]:
        episode_path = dataset_dir / entry["file"]
        if not episode_path.is_file():
            raise FileNotFoundError(f"Manifest episode is missing: {episode_path}")
        if sha256_file(episode_path) != entry["sha256"]:
            raise ValueError(f"Checksum mismatch for {episode_path.name}.")
        current = episode_summary(episode_path)
        for field in (
            "episode_id",
            "completed",
            "valid_action_contract",
            "terminal_reason",
            "steps",
        ):
            if current[field] != entry[field]:
                raise ValueError(
                    f"Manifest field '{field}' disagrees for {episode_path.name}."
                )
    return manifest


def split_episode_entries(
    entries: Iterable[dict],
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
    successful_only: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Split whole episodes deterministically, never individual frames."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    selected = [
        dict(entry)
        for entry in entries
        if not successful_only
        or (entry.get("completed") and entry.get("valid_action_contract"))
    ]
    if len(selected) < 2:
        raise ValueError("At least two eligible episodes are required for a split.")
    random.Random(seed).shuffle(selected)
    validation_count = max(1, round(len(selected) * validation_fraction))
    validation_count = min(validation_count, len(selected) - 1)
    validation = selected[:validation_count]
    training = selected[validation_count:]
    return training, validation
