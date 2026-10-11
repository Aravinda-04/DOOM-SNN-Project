from __future__ import annotations

import pytest
import torch

from networks import build_model
from networks.level_cnn import LevelCNNQNetwork
from networks.level_drqn import LevelDRQNQNetwork


@pytest.mark.parametrize("model_name", ["level_cnn", "level_drqn"])
def test_level_baseline_contract(model_name):
    model = build_model(model_name, action_size=8, in_channels=4)
    observation = torch.rand(2, 4, 84, 84)

    q_values, spikes, hidden = model(observation, None)

    assert q_values.shape == (2, 8)
    assert spikes.shape == torch.Size([])
    assert spikes.item() == 0.0
    assert isinstance(hidden, tuple)


def test_level_cnn_is_explicitly_non_recurrent():
    model = LevelCNNQNetwork()
    observation = torch.rand(1, 4, 84, 84)
    q1, _, hidden1 = model(observation, None)
    q2, _, hidden2 = model(observation, hidden1)

    assert model.is_recurrent is False
    assert hidden1 == hidden2 == ()
    assert torch.allclose(q1, q2)


def test_level_drqn_carries_cross_frame_state():
    model = LevelDRQNQNetwork()
    observation = torch.rand(1, 4, 84, 84)
    q_fresh, _, hidden = model(observation, None)
    q_recurrent, _, _ = model(observation, hidden)

    assert model.is_recurrent is True
    assert hidden[0].shape == (1, 128)
    assert hidden[1].shape == (1, 128)
    assert not torch.allclose(q_fresh, q_recurrent)


def test_level_drqn_gradient_crosses_frame_boundary():
    model = LevelDRQNQNetwork()
    first = torch.rand(1, 4, 84, 84, requires_grad=True)
    _, _, hidden = model(first, None)
    second = torch.rand(1, 4, 84, 84)
    q_values, _, _ = model(second, hidden)

    gradient = torch.autograd.grad(q_values.sum(), first)[0]

    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient).item() > 0
