"""Experimental compact visual SNN with spatial features retained before readout."""

import torch
from torch import nn
import snntorch as snn


class CompactConvSpikingQNetwork(nn.Module):
    def __init__(self, action_size=7, num_steps=10, beta=0.9):
        super().__init__()
        self.action_size = action_size
        self.num_steps = num_steps
        self.conv1 = nn.Conv2d(1, 4, kernel_size=8, stride=4)
        self.lif1 = snn.Leaky(beta=beta)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=4, stride=2)
        self.lif2 = snn.Leaky(beta=beta)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(8 * 4 * 4, 96)
        self.lif3 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(96, action_size)
        self.lif4 = snn.Leaky(beta=beta, reset_mechanism='none')

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        q = x.new_zeros((x.shape[0], self.action_size))
        spikes = x.new_zeros(())
        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(self.conv1(x), mem1)
            spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
            spk3, mem3 = self.lif3(self.fc1(self.pool(spk2).flatten(1)), mem3)
            _, mem4 = self.lif4(self.fc2(spk3), mem4)
            q = q + mem4
            spikes = spikes + spk1.sum() + spk2.sum() + spk3.sum()
        return q, spikes
