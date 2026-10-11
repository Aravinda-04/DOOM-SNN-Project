from __future__ import annotations

import torch

import numpy as np

from train_level import bellman_targets, select_action


def test_terminal_transition_does_not_bootstrap():
    rewards = torch.tensor([1.0, 1.0])
    max_next_q = torch.tensor([10.0, 10.0])
    dones = torch.tensor([1.0, 0.0])

    targets = bellman_targets(rewards, max_next_q, dones, gamma=0.9)

    assert targets[0].item() == 1.0
    assert targets[1].item() == 10.0


class _CountingPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, observation, hidden=None):
        self.calls += 1
        q_values = torch.tensor([[0.0, 1.0]], device=observation.device)
        spikes = torch.tensor(2.0, device=observation.device)
        state = torch.tensor(float(self.calls), device=observation.device)
        return q_values, spikes, (state, state)


def test_random_action_still_advances_recurrent_state(monkeypatch):
    policy = _CountingPolicy()
    monkeypatch.setattr("train_level.random.random", lambda: 0.0)
    monkeypatch.setattr("train_level.random.randrange", lambda _: 0)

    action, hidden, spikes = select_action(
        np.zeros((4, 84, 84), dtype=np.float32),
        policy,
        hidden=None,
        epsilon=1.0,
        action_size=2,
        device=torch.device("cpu"),
    )

    assert action == 0
    assert policy.calls == 1
    assert hidden[0].item() == 1.0
    assert spikes == 2.0
