from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from agent_config import ACTION_CATALOGUE_HASH, AGENT_CONTRACT
from demonstrations import (
    DEMONSTRATION_SCHEMA_VERSION,
    append_manifest_entry,
    episode_summary,
    save_episode,
)
from imitation_data import (
    DemonstrationSequenceDataset,
    action_class_weights,
    successful_episode_splits,
)
from train_imitation import imitation_sequence_loss


def _add_episode(dataset_dir, episode_index: int, steps: int = 30) -> None:
    episode_id = f"episode-{episode_index}"
    observations = np.full(
        (steps, 4, 84, 84), episode_index * 20, dtype=np.uint8
    )
    actions = np.arange(steps, dtype=np.int16) % AGENT_CONTRACT.action_size
    dones = np.zeros(steps, dtype=np.bool_)
    dones[-1] = True
    metadata = {
        "schema_version": DEMONSTRATION_SCHEMA_VERSION,
        "episode_id": episode_id,
        "created_at": f"2026-10-11T00:00:0{episode_index}+00:00",
        "agent_contract_sha256": ACTION_CATALOGUE_HASH,
        "completed": True,
        "terminal_reason": "completed",
        "valid_action_contract": True,
        "label_valid_fraction": 1.0,
        "minimum_label_valid_fraction": 0.95,
        "notes": "synthetic",
    }
    episode_path = dataset_dir / f"{episode_id}.npz"
    save_episode(
        episode_path,
        observations=observations,
        actions=actions,
        label_valid=np.ones(steps, dtype=np.bool_),
        rewards=np.zeros(steps, dtype=np.float32),
        raw_rewards=np.zeros(steps, dtype=np.float32),
        dones=dones,
        positions=np.zeros((steps, 3), dtype=np.float32),
        game_times=np.arange(steps, dtype=np.int32),
        metadata=metadata,
    )
    append_manifest_entry(dataset_dir, episode_summary(episode_path))


def test_sequence_dataset_uses_whole_episode_split_and_expected_windows(tmp_path):
    for index in range(3):
        _add_episode(tmp_path, index)
    training, validation, _ = successful_episode_splits(
        tmp_path, validation_fraction=1 / 3, seed=0
    )
    assert {entry["episode_id"] for entry in training}.isdisjoint(
        {entry["episode_id"] for entry in validation}
    )
    dataset = DemonstrationSequenceDataset(
        tmp_path,
        training,
        sequence_length=8,
        burn_in=2,
        stride=8,
    )
    sample = dataset[0]
    assert sample["observations"].shape == (10, 4, 84, 84)
    assert sample["observations"].dtype == torch.float32
    assert sample["actions"].shape == (10,)
    assert sample["label_valid"].shape == (10,)
    assert 0.0 <= float(sample["observations"].min()) <= 1.0
    assert 0.0 <= float(sample["observations"].max()) <= 1.0


def test_action_weights_are_finite_and_normalized_for_present_actions(tmp_path):
    for index in range(2):
        _add_episode(tmp_path, index)
    training, _, _ = successful_episode_splits(
        tmp_path, validation_fraction=0.5, seed=0
    )
    weights = action_class_weights(tmp_path, training)
    assert weights.shape == (AGENT_CONTRACT.action_size,)
    assert torch.isfinite(weights).all()
    assert weights[weights > 0].mean().item() == pytest.approx(1.0)


class _TinyRecurrentPolicy(nn.Module):
    spike_opportunities_per_sample = 4

    def __init__(self, action_size: int = 2):
        super().__init__()
        self.input = nn.Linear(1, 8)
        self.recurrent = nn.Linear(8, 8, bias=False)
        self.output = nn.Linear(8, action_size)

    def forward(self, observation, hidden=None):
        value = observation.mean(dim=tuple(range(1, observation.ndim))).unsqueeze(1)
        previous = (
            torch.zeros(observation.shape[0], 8, device=observation.device)
            if hidden is None
            else hidden[0]
        )
        state = torch.tanh(self.input(value) + self.recurrent(previous))
        logits = self.output(state)
        spikes = state.sigmoid().sum()
        return logits, spikes, (state,)


def test_imitation_loss_detaches_burn_in_but_trains_learning_window():
    torch.manual_seed(0)
    model = _TinyRecurrentPolicy()
    observations = torch.rand(2, 5, 1, 2, 2, requires_grad=True)
    actions = torch.tensor([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])
    loss, metrics = imitation_sequence_loss(
        model,
        observations,
        actions,
        burn_in=2,
        class_weights=None,
        sparsity_weight=1e-3,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["examples"].item() == 6
    assert torch.count_nonzero(observations.grad[:, :2]).item() == 0
    assert torch.count_nonzero(observations.grad[:, 2:]).item() > 0
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_tiny_recurrent_policy_can_overfit_a_fixed_sequence():
    torch.manual_seed(1)
    model = _TinyRecurrentPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    observations = torch.ones(4, 4, 1, 2, 2)
    actions = torch.zeros(4, 4, dtype=torch.long)
    losses = []
    for _ in range(30):
        loss, _ = imitation_sequence_loss(
            model,
            observations,
            actions,
            burn_in=1,
            class_weights=None,
            sparsity_weight=0.0,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    assert losses[-1] < losses[0] * 0.1

