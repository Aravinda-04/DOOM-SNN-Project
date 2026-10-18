"""Compact, checksummed storage for generated engine-state sequences."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from engine_state_config import ENGINE_STATE_CONTRACT_HASH, FEATURE_NAMES


ENGINE_STATE_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
REQUIRED_ARRAYS = (
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


def _validate(
    *,
    features: np.ndarray,
    feature_valid: np.ndarray,
    targets: np.ndarray,
    applied_targets: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    positions: np.ndarray,
    route_progress: np.ndarray,
    metadata: dict,
) -> None:
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"features must have shape (T, {len(FEATURE_NAMES)})."
        )
    length = features.shape[0]
    expected = {
        "feature_valid": (feature_valid, (length, len(FEATURE_NAMES))),
        "targets": (targets, (length, 4)),
        "applied_targets": (applied_targets, (length, 4)),
        "rewards": (rewards, (length,)),
        "dones": (dones, (length,)),
        "positions": (positions, (length, 3)),
        "route_progress": (route_progress, (length,)),
    }
    if length == 0:
        raise ValueError("An engine-state sequence cannot be empty.")
    for name, (array, shape) in expected.items():
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    if features.dtype != np.float32 or not np.all(np.isfinite(features)):
        raise ValueError("features must be finite float32 values.")
    if feature_valid.dtype != np.bool_:
        raise ValueError("feature_valid must have boolean dtype.")
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("targets must have an integer dtype.")
    if not np.issubdtype(applied_targets.dtype, np.integer):
        raise ValueError("applied_targets must have an integer dtype.")
    if np.any((targets[:, 0] < 0) | (targets[:, 0] >= 5)):
        raise ValueError("locomotion targets are outside 0..4.")
    if np.any((targets[:, 1] < 0) | (targets[:, 1] >= 3)):
        raise ValueError("turning targets are outside 0..2.")
    if np.any((targets[:, 2:] < 0) | (targets[:, 2:] > 1)):
        raise ValueError("attack and use targets must be binary.")
    if not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(positions)):
        raise ValueError("rewards and positions must be finite.")
    if not np.all(np.isfinite(route_progress)):
        raise ValueError("route_progress must be finite.")
    if metadata.get("schema_version") != ENGINE_STATE_SCHEMA_VERSION:
        raise ValueError("Engine-state schema version mismatch.")
    if metadata.get("engine_state_contract_hash") != ENGINE_STATE_CONTRACT_HASH:
        raise ValueError("Engine-state contract hash mismatch.")


def save_sequence(path: Path, *, metadata: dict, **arrays: np.ndarray) -> str:
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite engine-state data: {path}")
    _validate(metadata=metadata, **arrays)
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


def load_sequence(path: Path) -> dict:
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_ARRAYS) - set(archive.files))
        if missing:
            raise ValueError(f"Engine-state sequence is missing arrays: {missing}")
        payload = {name: np.array(archive[name], copy=True) for name in REQUIRED_ARRAYS}
    metadata = json.loads(str(payload.pop("metadata_json").item()))
    _validate(metadata=metadata, **payload)
    payload["metadata"] = metadata
    return payload


def load_manifest(dataset_dir: Path) -> dict:
    path = Path(dataset_dir).expanduser().resolve() / MANIFEST_FILENAME
    if not path.is_file():
        return {
            "schema_version": ENGINE_STATE_SCHEMA_VERSION,
            "engine_state_contract_hash": ENGINE_STATE_CONTRACT_HASH,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "episodes": [],
        }
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ENGINE_STATE_SCHEMA_VERSION:
        raise ValueError("Engine-state manifest schema version mismatch.")
    if manifest.get("engine_state_contract_hash") != ENGINE_STATE_CONTRACT_HASH:
        raise ValueError("Engine-state manifest contract hash mismatch.")
    return manifest

def append_manifest_entry(dataset_dir: Path, entry: dict) -> Path:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(dataset_dir)
    if any(item["episode_id"] == entry["episode_id"] for item in manifest["episodes"]):
        raise ValueError(f"Duplicate engine-state episode: {entry['episode_id']}")
    manifest["episodes"].append(entry)
    path = dataset_dir / MANIFEST_FILENAME
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


def validate_manifest(dataset_dir: Path) -> dict:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    manifest = load_manifest(dataset_dir)
    for entry in manifest["episodes"]:
        path = dataset_dir / entry["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing engine-state episode: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Checksum mismatch for engine-state episode: {path}")
        payload = load_sequence(path)
        if len(payload["features"]) != int(entry["steps"]):
            raise ValueError(f"Step count mismatch for engine-state episode: {path}")
    return manifest

