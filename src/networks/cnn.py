"""Conventional dueling CNN policy for normalized grayscale game frames."""

import torch
from torch import nn


class CNNQNetwork(nn.Module):
    """Estimate state value and relative action advantages from (B, 1, 84, 84)."""

    def __init__(self, action_size: int = 3):
        super().__init__()
        self.action_size = action_size
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        # Keep the action head name for checkpoint/export action-size discovery.
        self.fc2 = nn.Linear(512, action_size)
        self.value_fc1 = nn.Linear(64 * 7 * 7, 512)
        self.value_fc2 = nn.Linear(512, 1)

    def forward(self, x):
        features = torch.relu(self.conv1(x))
        features = torch.relu(self.conv2(features))
        features = self.flatten(torch.relu(self.conv3(features)))
        advantage = self.fc2(torch.relu(self.fc1(features)))
        value = self.value_fc2(torch.relu(self.value_fc1(features)))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        # Shared policy contract; conventional activations generate no spikes.
        return q_values, q_values.new_zeros(())
