"""Cross-Frame Recurrent Spiking Neural Network Q-network (CRSNN).

The key architectural difference from rsnn.py is that the recurrent hidden
state (spk4, mem4) persists **across game frames**, not just within the
10 internal time-steps of a single forward call.

Input:  (B, 4, 84, 84)  — 4-frame grayscale stack from DoomLevelEnvironment
Output: q_values (B, action_size), total_spikes scalar, new_hidden tuple

Usage
-----
During inference the caller is responsible for carrying ``hidden`` between
steps and resetting it to ``None`` at the start of each episode::

    hidden = None
    obs = env.reset()
    while not done:
        t = torch.from_numpy(obs).unsqueeze(0).to(device)  # (1,4,84,84)
        q, spikes, hidden = model(t, hidden)
        action = q.argmax(1).item()
        obs, reward, done = env.step(action)

During training (TBPTT) the hidden state is initialised to ``None`` at the
start of every sampled sequence window so gradients do not flow beyond the
sequence boundary.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import snntorch as snn

Hidden = tuple[torch.Tensor, torch.Tensor]


class CRSNNQNetwork(nn.Module):
    """Cross-Frame Recurrent Spiking Q-Network.

    Parameters
    ----------
    action_size:
        Number of discrete actions (9 for the current full-level contract).
    num_steps:
        Internal LIF simulation steps per game frame (temporal depth).
    beta:
        Membrane potential decay factor for all LIF neurons (0 < β < 1).
    in_channels:
        Number of input channels. Matches the frame-stack depth (4).
    """

    # Mark the network so that training loops know to manage hidden state.
    is_recurrent: bool = True

    def __init__(
        self,
        action_size: int = 9,
        num_steps: int = 10,
        beta: float = 0.9,
        in_channels: int = 4,
    ):
        super().__init__()
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        self.num_steps = num_steps
        self.action_size = action_size
        self.spike_opportunities_per_sample = num_steps * (
            16 * 20 * 20 + 32 * 9 * 9 + 64 * 7 * 7 + 512
        )

        # ── Convolutional feature extractor ─────────────────────────────────
        # Input  (B, 4, 84, 84)
        # After conv1 (k=8, s=4): (B, 16, 20, 20)
        # After conv2 (k=4, s=2): (B, 32,  9,  9)
        # After conv3 (k=3, s=1): (B, 64,  7,  7)  → flatten → 3136
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        self.lif1 = snn.Leaky(beta=beta)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.lif2 = snn.Leaky(beta=beta)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.lif3 = snn.Leaky(beta=beta)

        self.flatten = nn.Flatten()

        # ── Cross-frame recurrent hidden layer ───────────────────────────────
        # fc1:      feed-forward path from CNN features → 512
        # recurrent: previous-frame spike feedback        → 512
        # lif4:     LIF neuron whose state crosses frames
        self.fc1 = nn.Linear(3136, 512)
        self.recurrent = nn.Linear(512, 512)
        self.lif4 = snn.Leaky(beta=beta)

        # ── Output Q-value layer ─────────────────────────────────────────────
        # Accumulates membrane potential (no reset) to produce Q-values.
        self.fc2 = nn.Linear(512, action_size)
        self.lif5 = snn.Leaky(beta=beta, reset_mechanism="none")

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Hidden] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Hidden]:
        """Forward pass over ``num_steps`` internal time steps.

        Parameters
        ----------
        x:
            Observation tensor of shape ``(B, 4, 84, 84)``.
        hidden:
            Tuple ``(spk4_prev, mem4)`` carrying the cross-frame recurrent
            state.  Pass ``None`` at the start of an episode or a TBPTT
            sequence boundary to initialise with zeros.

        Returns
        -------
        q_values:
            Accumulated Q-values, shape ``(B, action_size)``.
        total_spikes:
            Scalar spike count for the sparsity regularisation term.
        new_hidden:
            Updated ``(spk4, mem4)`` to pass into the next frame's call.
        """
        batch = x.size(0)

        # Per-call membrane potentials for the convolutional layers.
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem5 = self.lif5.init_leaky()

        # Cross-frame hidden state: initialised to zeros at episode start.
        if hidden is None:
            spk4 = torch.zeros(batch, 512, device=x.device)
            mem4 = self.lif4.init_leaky()
        else:
            # The caller owns TBPTT boundaries. Keeping the supplied state
            # attached here allows gradients to flow through every frame in a
            # sampled training window. Inference already runs under no_grad().
            spk4, mem4 = hidden

        total_spikes = torch.tensor(0.0, device=x.device)
        encoded_current = self.conv1(x)

        for _ in range(self.num_steps):
            # ── Convolutional spiking encoder ────────────────────────────────
            spk1, mem1 = self.lif1(encoded_current, mem1)

            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.conv3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            flat = self.flatten(spk3)

            # ── Cross-frame recurrent step ───────────────────────────────────
            # Combine feed-forward input with the previous frame's spike output.
            cur4 = self.fc1(flat) + self.recurrent(spk4)
            spk4, mem4 = self.lif4(cur4, mem4)

            # ── Output accumulation ──────────────────────────────────────────
            cur5 = self.fc2(spk4)
            _, mem5 = self.lif5(cur5, mem5)

            # Track total spikes for sparsity loss.
            total_spikes = (
                total_spikes
                + spk1.sum()
                + spk2.sum()
                + spk3.sum()
                + spk4.sum()
            )

        # lif5 already integrates output current over the neural simulation;
        # summing every intermediate membrane would integrate it twice.
        return mem5, total_spikes, (spk4, mem4)
