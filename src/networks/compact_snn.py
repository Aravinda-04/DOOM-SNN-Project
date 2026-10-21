"""Small experimental visual SNN; hardware compatibility is not established."""

import torch
from torch import nn
import snntorch as snn


class CompactSpikingQNetwork(nn.Module):
    def __init__(self, action_size=5, num_steps=10, beta=0.9):
        super().__init__()
        self.action_size = action_size
        self.num_steps = num_steps
        self.pool = nn.AdaptiveAvgPool2d((16, 16))
        self.fc1 = nn.Linear(256, 128)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(128, action_size)
        self.lif2 = snn.Leaky(beta=beta, reset_mechanism="none")

    def forward(self, x):
        current = self.fc1(self.pool(x).flatten(1))
        mem1, mem2 = self.lif1.init_leaky(), self.lif2.init_leaky()
        q = x.new_zeros((x.shape[0], self.action_size))
        spikes = x.new_zeros(())
        for _ in range(self.num_steps):
            spk, mem1 = self.lif1(current, mem1)
            _, mem2 = self.lif2(self.fc2(spk), mem2)
            q = q + mem2
            spikes = spikes + spk.sum()
        return q, spikes
