"""Compact, checksummed storage and schema contracts for MAP01 vision sequences."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from agent_config import FRAME_HEIGHT, FRAME_STACK, FRAME_WIDTH
from engine_state_config import ENGINE_STATE_CONTRACT_HASH, FEATURE_NAMES


VISION_SCHEMA_VERSION = 1
VISION_MANIFEST_FILENAME = "manifest.json"
REQUIRED_VISION_ARRAYS = (
    "frames",
    "features",
    "feature_valid",
    "targets",
    "applied_targets",
    "rewards",
    "dones",
    "positions",
    "route_progress",
    "metadata_json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_vision_sequence(
    *,
    frames: np.ndarray,
    features: np.ndarray,
    feature_valid: np.ndarray,
    targets: np.ndarray,
    applied_targets: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    positions: np.ndarray,
    route_progress: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Validate shape, type, and values of vision sequence arrays."""
    if frames.ndim != 4 or frames.shape[1:] != (FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError(
            f"frames must have shape (T, {FRAME_STACK}, {FRAME_HEIGHT}, {FRAME_WIDTH}), "
            f"got {frames.shape}."
        )
    length = frames.shape[0]
    if length == 0:
        raise ValueError("A vision sequence cannot be empty.")
    if frames.dtype != np.uint8:
        raise ValueError(f"frames must have uint8 dtype, got {frames.dtype}.")

    if features.ndim != 2 or features.shape != (length, len(FEATURE_NAMES)):
        raise ValueError(
            f"features must have shape ({length}, {len(FEATURE_NAMES)}), got {features.shape}."
        )
    if features.dtype != np.float32 or not np.all(np.isfinite(features)):
        raise ValueError("features must be finite float32 values.")

    if feature_valid.shape != (length, len(FEATURE_NAMES)) or feature_valid.dtype != np.bool_:
        raise ValueError(
            f"feature_valid must have shape ({length}, {len(FEATURE_NAMES)}) boolean."
        )

    if targets.shape != (length, 4) or not np.issubdtype(targets.dtype, np.integer):
        raise ValueError(f"targets must have shape ({length}, 4) integers.")
    if applied_targets.shape != (length, 4) or not np.issubdtype(applied_targets.dtype, np.integer):
        raise ValueError(f"applied_targets must have shape ({length}, 4) integers.")

    if np.any((targets[:, 0] < 0) | (targets[:, 0] >= 5)):
        raise ValueError("locomotion targets are outside 0..4.")
    if np.any((targets[:, 1] < 0) | (targets[:, 1] >= 3)):
        raise ValueError("turning targets are outside 0..2.")
    if np.any((targets[:, 2:] < 0) | (targets[:, 2:] > 1)):
        raise ValueError("attack and use targets must be binary (0 or 1).")

    if rewards.shape != (length,) or not np.all(np.isfinite(rewards)):
        raise ValueError("rewards must have shape (T,) finite floats.")
    if dones.shape != (length,) or dones.dtype != np.bool_:
        raise ValueError("dones must have shape (T,) booleans.")
    if positions.shape != (length, 3) or not np.all(np.isfinite(positions)):
        raise ValueError("positions must have shape (T, 3) finite floats.")
    if route_progress.shape != (length,) or not np.all(np.isfinite(route_progress)):
        raise ValueError("route_progress must have shape (T,) finite floats.")

    if metadata.get("schema_version") != VISION_SCHEMA_VERSION:
        raise ValueError("Vision schema version mismatch.")
    if metadata.get("engine_state_contract_hash") != ENGINE_STATE_CONTRACT_HASH:
        raise ValueError("Engine-state contract hash mismatch.")


def save_vision_sequence(path: Path, *, metadata: dict[str, Any], **arrays: np.ndarray) -> str:
    """Atomically save a validated vision sequence to a compressed .npz file."""
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite vision sequence: {path}")
    validate_vision_sequence(metadata=metadata, **arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                **arrays,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def load_vision_sequence(path: Path) -> dict[str, Any]:
    """Load and validate a vision sequence archive."""
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_VISION_ARRAYS) - set(archive.files))
        if missing:
            raise ValueError(f"Vision sequence is missing arrays: {missing}")
        payload = {name: np.array(archive[name], copy=True) for name in REQUIRED_VISION_ARRAYS}
    metadata = json.loads(str(payload.pop("metadata_json").item()))
    validate_vision_sequence(metadata=metadata, **payload)
    payload["metadata"] = metadata
    return payload


def load_vision_manifest(dataset_dir: Path) -> dict[str, Any]:
    """Load or initialize the vision dataset manifest."""
    path = Path(dataset_dir).expanduser().resolve() / VISION_MANIFEST_FILENAME
    if not path.is_file():
        return {
            "schema_version": VISION_SCHEMA_VERSION,
            "engine_state_contract_hash": ENGINE_STATE_CONTRACT_HASH,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "episodes": [],
        }
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != VISION_SCHEMA_VERSION:
        raise ValueError("Vision manifest schema version mismatch.")
    if manifest.get("engine_state_contract_hash") != ENGINE_STATE_CONTRACT_HASH:
        raise ValueError("Vision manifest contract hash mismatch.")
    return manifest


def append_vision_manifest_entry(dataset_dir: Path, entry: dict[str, Any]) -> Path:
    """Atomically append an episode record to the vision manifest."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_vision_manifest(dataset_dir)
    manifest["episodes"] = [
        item for item in manifest["episodes"] if item["episode_id"] != entry["episode_id"]
    ]
    manifest["episodes"].append(entry)
    path = dataset_dir / VISION_MANIFEST_FILENAME
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".manifest-", suffix=".tmp", dir=dataset_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def validate_vision_manifest(dataset_dir: Path) -> dict[str, Any]:
    """Verify integrity of all files listed in the vision manifest."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    manifest = load_vision_manifest(dataset_dir)
    for entry in manifest["episodes"]:
        path = dataset_dir / entry["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing vision episode: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Checksum mismatch for vision episode: {path}")
        payload = load_vision_sequence(path)
        if len(payload["frames"]) != int(entry["steps"]):
            raise ValueError(f"Step count mismatch for vision episode: {path}")
    return manifest
