"""Four-frame convolutional DQN baseline for full-level experiments."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LevelCNNQNetwork(nn.Module):
    """Non-recurrent CNN with the same visual dimensions as the level SNNs."""

    is_recurrent = False

    def __init__(
        self,
        action_size: int = 8,
        in_channels: int = 4,
        **_: object,
    ):
        super().__init__()
        self.action_size = action_size
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc1 = nn.Linear(3136, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, action_size)

    def forward(
        self,
        observation: torch.Tensor,
        hidden: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        del hidden
        features = self.features(observation)
        q_values = self.fc2(self.relu(self.fc1(features)))
        spikes = torch.tensor(0.0, device=observation.device)
        return q_values, spikes, ()
