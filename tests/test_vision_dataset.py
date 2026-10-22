"""Tests for vision dataset schema, serialization, and PyTorch dataset loaders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from agent_config import FRAME_HEIGHT, FRAME_STACK, FRAME_WIDTH
from engine_state_config import ENGINE_STATE_CONTRACT_HASH, FEATURE_NAMES
from vision_data_schema import (
    VISION_SCHEMA_VERSION,
    append_vision_manifest_entry,
    load_vision_manifest,
    load_vision_sequence,
    save_vision_sequence,
    validate_vision_manifest,
    validate_vision_sequence,
)
from vision_dataset import (
    VisionFrameDataset,
    VisionSequenceDataset,
    vision_episode_splits,
)


def _make_dummy_vision_data(length: int = 16) -> tuple[dict, dict[str, np.ndarray]]:
    rng = np.random.default_rng(1234)
    metadata = {
        "schema_version": VISION_SCHEMA_VERSION,
        "engine_state_contract_hash": ENGINE_STATE_CONTRACT_HASH,
        "seed": 42,
    }
    arrays = {
        "frames": rng.integers(0, 256, size=(length, FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8),
        "features": rng.uniform(0, 1, size=(length, len(FEATURE_NAMES))).astype(np.float32),
        "feature_valid": np.ones((length, len(FEATURE_NAMES)), dtype=np.bool_),
        "targets": rng.integers(0, 2, size=(length, 4), dtype=np.int64),
        "applied_targets": rng.integers(0, 2, size=(length, 4), dtype=np.int64),
        "rewards": np.zeros(length, dtype=np.float32),
        "dones": np.zeros(length, dtype=np.bool_),
        "positions": np.zeros((length, 3), dtype=np.float32),
        "route_progress": np.linspace(0, 1, length, dtype=np.float32),
    }
    return metadata, arrays


def test_validate_vision_sequence_checks_dimensions():
    metadata, arrays = _make_dummy_vision_data(10)
    # Valid call should not raise
    validate_vision_sequence(metadata=metadata, **arrays)

    # Valid RGB call (12 channels) should not raise
    rgb_arrays = arrays.copy()
    rgb_arrays["frames"] = np.zeros((10, 12, 84, 84), dtype=np.uint8)
    validate_vision_sequence(metadata=metadata, **rgb_arrays)

    # Bad frame shape should raise
    bad_frames = arrays.copy()
    bad_frames["frames"] = np.zeros((10, 1, 84, 84), dtype=np.uint8)
    with pytest.raises(ValueError, match="frames must have shape"):
        validate_vision_sequence(metadata=metadata, **bad_frames)

    # Non-uint8 frames should raise
    float_frames = arrays.copy()
    float_frames["frames"] = np.zeros((10, 4, 84, 84), dtype=np.float32)
    with pytest.raises(ValueError, match="frames must have uint8 dtype"):
        validate_vision_sequence(metadata=metadata, **float_frames)


def test_vision_sequence_roundtrip(tmp_path: Path):
    metadata, arrays = _make_dummy_vision_data(20)
    filepath = tmp_path / "ep01.npz"

    sha256 = save_vision_sequence(filepath, metadata=metadata, **arrays)
    assert filepath.is_file()
    assert len(sha256) == 64

    loaded = load_vision_sequence(filepath)
    assert np.array_equal(loaded["frames"], arrays["frames"])
    assert np.allclose(loaded["features"], arrays["features"])
    assert np.array_equal(loaded["feature_valid"], arrays["feature_valid"])
    assert loaded["metadata"]["seed"] == 42


def test_vision_manifest_and_datasets(tmp_path: Path):
    metadata_1, arrays_1 = _make_dummy_vision_data(24)
    metadata_2, arrays_2 = _make_dummy_vision_data(32)

    file_1 = tmp_path / "ep_01.npz"
    file_2 = tmp_path / "ep_02.npz"

    sha_1 = save_vision_sequence(file_1, metadata=metadata_1, **arrays_1)
    sha_2 = save_vision_sequence(file_2, metadata=metadata_2, **arrays_2)

    append_vision_manifest_entry(
        tmp_path,
        {"episode_id": "ep_01", "file": "ep_01.npz", "steps": 24, "sha256": sha_1},
    )
    append_vision_manifest_entry(
        tmp_path,
        {"episode_id": "ep_02", "file": "ep_02.npz", "steps": 32, "sha256": sha_2},
    )

    manifest = validate_vision_manifest(tmp_path)
    assert len(manifest["episodes"]) == 2

    # Test VisionFrameDataset
    frame_ds = VisionFrameDataset(tmp_path)
    assert len(frame_ds) == 24 + 32
    item = frame_ds[0]
    assert item["frames"].shape == (FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH)
    assert item["frames"].dtype == torch.float32
    assert 0.0 <= item["frames"].max().item() <= 1.0
    assert item["features"].shape == (len(FEATURE_NAMES),)

    # Test VisionSequenceDataset
    seq_ds = VisionSequenceDataset(tmp_path, sequence_length=16, stride=8)
    assert len(seq_ds) > 0
    seq_item = seq_ds[0]
    assert seq_item["frames"].shape == (16, FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH)

    # Test splits
    train, val = vision_episode_splits(tmp_path, val_fraction=0.5, seed=1)
    assert len(train) == 1
    assert len(val) == 1
    assert train[0] != val[0]
