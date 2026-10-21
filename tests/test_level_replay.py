from __future__ import annotations

import numpy as np
import pytest

from train_level import EpisodeReplayMemory


def _observation(value: float) -> np.ndarray:
    return np.full((4, 84, 84), value, dtype=np.float32)


def _push_episode(memory: EpisodeReplayMemory, length: int, offset: float = 0.0) -> None:
    for index in range(length):
        value = min(offset + index / 10.0, 1.0)
        next_value = min(offset + (index + 1) / 10.0, 1.0)
        memory.push(
            _observation(value),
            action=index % 8,
            reward=float(index),
            next_obs=_observation(next_value),
            done=index == length - 1,
        )


def test_capacity_is_measured_in_transitions():
    memory = EpisodeReplayMemory(capacity=5, seq_len=2)
    _push_episode(memory, 3)
    _push_episode(memory, 3, offset=0.2)

    assert len(memory) == 3
    assert memory.num_episodes == 1


def test_replay_stores_observations_as_uint8():
    memory = EpisodeReplayMemory(capacity=10, seq_len=2)
    _push_episode(memory, 3)

    stored = memory._episodes[0]  # Internal representation is part of this contract test.
    assert stored.transitions[0][0].dtype == np.uint8
    assert stored.final_observation.dtype == np.uint8


def test_sample_reconstructs_normalized_next_observations():
    memory = EpisodeReplayMemory(capacity=10, seq_len=3)
    _push_episode(memory, 3)

    sequence = memory.sample(1)[0]

    assert len(sequence) == 3
    for obs, _, _, next_obs, _ in sequence:
        assert obs.dtype == np.float32
        assert next_obs.dtype == np.float32
        assert 0.0 <= float(obs.min()) <= float(obs.max()) <= 1.0
        assert 0.0 <= float(next_obs.min()) <= float(next_obs.max()) <= 1.0
    assert sequence[-1][4] is True


def test_sampling_empty_memory_fails_clearly():
    memory = EpisodeReplayMemory(capacity=10, seq_len=2)
    with pytest.raises(ValueError, match="no complete eligible episodes"):
        memory.sample(1)


def test_single_long_episode_is_trimmed_to_capacity():
    memory = EpisodeReplayMemory(capacity=4, seq_len=2)
    _push_episode(memory, 7)

    assert len(memory) == 4
    assert memory.num_episodes == 1
    assert len(memory.sample(1)[0]) == 2


def test_sample_includes_burn_in_prefix():
    memory = EpisodeReplayMemory(capacity=10, seq_len=3, burn_in=2)
    _push_episode(memory, 6)

    sequence = memory.sample(1)[0]

    assert len(sequence) == 5
