from __future__ import annotations

import json

import numpy as np
import pytest

from agent_config import ACTIONS, ACTION_CATALOGUE_HASH
from demonstrations import (
    DEMONSTRATION_SCHEMA_VERSION,
    append_manifest_entry,
    episode_summary,
    load_episode,
    load_manifest,
    observation_to_uint8,
    save_episode,
    split_episode_entries,
    validate_manifest,
)
from record_demonstrations import _record_one_episode


def _metadata(episode_id: str = "episode-1", *, completed: bool = True) -> dict:
    return {
        "schema_version": DEMONSTRATION_SCHEMA_VERSION,
        "episode_id": episode_id,
        "created_at": "2026-10-11T00:00:00+00:00",
        "agent_contract_sha256": ACTION_CATALOGUE_HASH,
        "completed": completed,
        "terminal_reason": "completed" if completed else "timed_out",
        "valid_action_contract": True,
        "notes": "test",
    }


def _arrays(length: int = 3, *, completed: bool = True) -> dict:
    dones = np.zeros(length, dtype=np.bool_)
    if completed:
        dones[-1] = True
    return {
        "observations": np.zeros((length, 4, 84, 84), dtype=np.uint8),
        "actions": np.arange(length, dtype=np.int16),
        "rewards": np.linspace(0, 1, length, dtype=np.float32),
        "raw_rewards": np.zeros(length, dtype=np.float32),
        "dones": dones,
        "positions": np.zeros((length, 3), dtype=np.float32),
        "game_times": np.arange(length, dtype=np.int32),
    }


def test_normalized_observation_round_trip_encoding():
    observation = np.linspace(0.0, 1.0, 4 * 84 * 84, dtype=np.float32).reshape(
        4, 84, 84
    )
    encoded = observation_to_uint8(observation)
    assert encoded.dtype == np.uint8
    assert encoded.shape == observation.shape
    assert encoded.min() == 0
    assert encoded.max() == 255


def test_episode_save_load_and_manifest_validation(tmp_path):
    episode_path = tmp_path / "episode-1.npz"
    save_episode(episode_path, metadata=_metadata(), **_arrays())

    payload = load_episode(episode_path)
    assert payload["metadata"]["episode_id"] == "episode-1"
    assert payload["observations"].dtype == np.uint8
    assert payload["actions"].tolist() == [0, 1, 2]

    summary = episode_summary(episode_path, successful_only=True)
    append_manifest_entry(tmp_path, summary)
    manifest = validate_manifest(tmp_path)
    assert len(manifest["episodes"]) == 1
    assert manifest["episodes"][0]["sha256"] == summary["sha256"]


def test_manifest_checksum_detects_modified_episode(tmp_path):
    episode_path = tmp_path / "episode-1.npz"
    save_episode(episode_path, metadata=_metadata(), **_arrays())
    append_manifest_entry(tmp_path, episode_summary(episode_path))

    with episode_path.open("ab") as handle:
        handle.write(b"changed")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        validate_manifest(tmp_path)


def test_invalid_action_label_must_match_metadata(tmp_path):
    arrays = _arrays()
    arrays["actions"][1] = -1
    with pytest.raises(ValueError, match="valid_action_contract"):
        save_episode(tmp_path / "invalid.npz", metadata=_metadata(), **arrays)


def test_manifest_rejects_a_different_contract(tmp_path):
    manifest = load_manifest(tmp_path)
    manifest["agent_contract_sha256"] = "different"
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="different agent/action contract"):
        load_manifest(tmp_path)


def test_episode_split_never_splits_frames_or_overlaps():
    entries = [
        {
            "episode_id": f"ep-{index}",
            "completed": True,
            "valid_action_contract": True,
        }
        for index in range(10)
    ]
    training, validation = split_episode_entries(entries, seed=123)
    training_ids = {entry["episode_id"] for entry in training}
    validation_ids = {entry["episode_id"] for entry in validation}
    assert len(training) == 8
    assert len(validation) == 2
    assert training_ids.isdisjoint(validation_ids)
    assert training_ids | validation_ids == {f"ep-{index}" for index in range(10)}


class _FakeSpectatorEnvironment:
    def __init__(self, *, stable: bool = True):
        self.initial = np.full((4, 84, 84), 0.25, dtype=np.float32)
        self.next = np.full((4, 84, 84), 0.5, dtype=np.float32)
        first = tuple(float(value) for value in ACTIONS[0])
        second = first if stable else tuple(float(value) for value in ACTIONS[1])
        self.last_spectator_tic_actions = (first, first, second, second)

    def reset(self):
        return self.initial

    def spectator_step(self):
        return self.next, 1.5, True, self.last_spectator_tic_actions[-1]

    def diagnostics(self):
        return {
            "exit_triggered": True,
            "terminal_reason": "completed",
            "last_raw_reward": 2.0,
            "position_x": 1.0,
            "position_y": 2.0,
            "angle": 90.0,
            "episode_time": 4,
        }


def test_recorder_aligns_pre_action_observation_with_stable_action():
    metadata, arrays = _record_one_episode(
        _FakeSpectatorEnvironment(stable=True),
        episode_id="aligned",
        seed=0,
        notes="",
    )
    assert metadata["completed"] is True
    assert metadata["valid_action_contract"] is True
    assert arrays["actions"].tolist() == [0]
    np.testing.assert_array_equal(
        arrays["observations"][0], observation_to_uint8(np.full((4, 84, 84), 0.25))
    )
    assert arrays["positions"].tolist() == [[1.0, 2.0, 90.0]]


def test_recorder_marks_mid_decision_action_changes_invalid():
    metadata, arrays = _record_one_episode(
        _FakeSpectatorEnvironment(stable=False),
        episode_id="unstable",
        seed=0,
        notes="",
    )
    assert metadata["valid_action_contract"] is False
    assert metadata["unsupported_decisions"] == 1
    assert arrays["actions"].tolist() == [-1]
