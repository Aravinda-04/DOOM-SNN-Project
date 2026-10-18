from __future__ import annotations

import torch

import numpy as np

from train_level import bellman_targets, evaluate_policy, select_action


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


class _ShortEvaluationEnvironment:
    def __init__(self, **_):
        self.steps = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def reset(self):
        self.steps = 0
        return np.zeros((4, 84, 84), dtype=np.float32)

    def step(self, _action):
        self.steps += 1
        return (
            np.zeros((4, 84, 84), dtype=np.float32),
            1.0,
            self.steps == 2,
        )

    def diagnostics(self):
        return {
            "exit_triggered": self.steps == 2,
            "terminal_reason": "completed" if self.steps == 2 else "in_progress",
            "kill_count": 0.0,
            "sectors_visited": self.steps,
        }


def test_evaluation_progress_reports_periodic_and_terminal_status(capsys, tmp_path):
    stats = evaluate_policy(
        _ShortEvaluationEnvironment,
        tmp_path / "unused.cfg",
        _CountingPolicy(),
        torch.device("cpu"),
        episodes=1,
        seeds=(0,),
        progress=True,
        progress_interval=1,
    )
    output = capsys.readouterr().out
    assert "eval run 1/1 (seed=0) started" in output
    assert "steps=1" in output
    assert "terminal=completed completion=True" in output
    assert "actions: backward=100.0%; longest_repeat=2" in output
    assert stats["completion_rate"] == 1.0
    assert stats["dominant_action_fraction"] == 1.0
    assert stats["unique_actions_used"] == 1
    assert stats["mean_longest_action_streak"] == 2.0
