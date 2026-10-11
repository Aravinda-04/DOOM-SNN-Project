"""RSNNQNetwork v2 — upgraded RSNN with cross-frame hidden state API.

This is a lightweight recurrent alternative to CRSNN that:
  1. Accepts a 4-channel frame stack (in_channels=4) to match DoomLevelEnvironment.
  2. Uses an element-wise recurrent LIF layer instead of CRSNN's dense 512 x
     512 recurrent projection, providing a meaningfully smaller recurrent
     parameterization.
  3. Exposes a ``hidden`` tuple that persists (spk4, mem4) across game frames,
     making the recurrence genuinely temporal rather than limited to the
     internal simulation loop.
  4. Returns ``(q_values, total_spikes, new_hidden)`` — the same signature as
     CRSNNQNetwork — so training code can switch between them with no changes.

The existing rsnn.py (single-channel, within-frame recurrence) is preserved
untouched for backward compatibility with basic.cfg experiments.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import snntorch as snn

Hidden = tuple[torch.Tensor, torch.Tensor]


class RSNNv2QNetwork(nn.Module):
    """Upgraded RSNN Q-network with genuine cross-frame recurrence.

    Parameters
    ----------
    action_size:
        Number of discrete actions.
    num_steps:
        Internal LIF simulation steps per game frame.
    beta:
        Membrane decay constant for all LIF neurons.
    in_channels:
        Input channels (4 for frame-stacked full-level observations).
    """

    is_recurrent: bool = True

    def __init__(
        self,
        action_size: int = 8,
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

        # ── Convolutional spiking encoder ────────────────────────────────────
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        self.lif1 = snn.Leaky(beta=beta)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.lif2 = snn.Leaky(beta=beta)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.lif3 = snn.Leaky(beta=beta)

        self.flatten = nn.Flatten()

        # ── Recurrent hidden layer ───────────────────────────────────────────
        # Unlike CRSNN's dense all-to-all 512 x 512 projection, RLeaky uses a
        # learned element-wise recurrent connection. This is substantially
        # smaller and creates a controlled architecture comparison.
        self.fc1 = nn.Linear(3136, 512)
        self.lif4 = snn.RLeaky(
            beta=beta,
            all_to_all=False,
            V=1.0,
            learn_recurrent=True,
        )

        # ── Output layer ─────────────────────────────────────────────────────
        self.fc2 = nn.Linear(512, action_size)
        self.lif5 = snn.Leaky(beta=beta, reset_mechanism="none")

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Hidden] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Hidden]:
        """One game-frame forward pass.

        Parameters
        ----------
        x:
            ``(B, in_channels, 84, 84)`` observation.
        hidden:
            ``(spk4_prev, mem4)`` from the previous game frame, or ``None``
            to initialise with zeros (start of episode / TBPTT boundary).

        Returns
        -------
        q_values, total_spikes, new_hidden
        """
        batch = x.size(0)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem5 = self.lif5.init_leaky()

        if hidden is None:
            spk4 = torch.zeros(batch, 512, device=x.device)
            _, mem4 = self.lif4.init_rleaky()
        else:
            # Sequence-boundary detachment is a trainer responsibility; doing
            # it here would reduce the configured TBPTT window to one frame.
            spk4, mem4 = hidden

        total_spikes = torch.tensor(0.0, device=x.device)
        encoded_current = self.conv1(x)

        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(encoded_current, mem1)

            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.conv3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            flat = self.flatten(spk3)

            cur4 = self.fc1(flat)
            spk4, mem4 = self.lif4(cur4, spk4, mem4)

            cur5 = self.fc2(spk4)
            _, mem5 = self.lif5(cur5, mem5)

            total_spikes = (
                total_spikes
                + spk1.sum()
                + spk2.sum()
                + spk3.sum()
                + spk4.sum()
            )

        return mem5, total_spikes, (spk4, mem4)
