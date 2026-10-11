"""Four-frame LSTM DQN baseline for full-level recurrent experiments."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

Hidden = tuple[torch.Tensor, torch.Tensor]


class LevelDRQNQNetwork(nn.Module):
    """CNN encoder followed by an LSTMCell carried across game frames."""

    is_recurrent = True

    def __init__(
        self,
        action_size: int = 8,
        in_channels: int = 4,
        hidden_size: int = 128,
        **_: object,
    ):
        super().__init__()
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.recurrent = nn.LSTMCell(3136, hidden_size)
        self.fc = nn.Linear(hidden_size, action_size)

    def forward(
        self,
        observation: torch.Tensor,
        hidden: Optional[Hidden] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Hidden]:
        batch_size = observation.shape[0]
        features = self.features(observation)
        if hidden is None:
            h = torch.zeros(
                batch_size,
                self.hidden_size,
                device=observation.device,
                dtype=features.dtype,
            )
            c = torch.zeros_like(h)
        else:
            h, c = hidden
        h, c = self.recurrent(features, (h, c))
        q_values = self.fc(h)
        spikes = torch.tensor(0.0, device=observation.device)
        return q_values, spikes, (h, c)
