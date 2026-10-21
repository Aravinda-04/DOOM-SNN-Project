"""Compact recurrent policy for privileged MAP01 engine state."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import snntorch as snn


class EngineStateGRUPolicy(nn.Module):
    """Shared recurrent trunk with factorized control heads."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
    ):
        super().__init__()
        if input_size <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError("Policy dimensions must be positive.")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.recurrent = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.locomotion_head = nn.Linear(hidden_size, 5)
        self.turning_head = nn.Linear(hidden_size, 3)
        self.attack_head = nn.Linear(hidden_size, 1)
        self.use_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        squeeze_time = inputs.ndim == 2
        if squeeze_time:
            inputs = inputs.unsqueeze(1)
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected input shape (B, T, {self.input_size}) or "
                f"(B, {self.input_size}), got {tuple(inputs.shape)}."
            )
        encoded = self.encoder(inputs)
        recurrent, new_hidden = self.recurrent(encoded, hidden)
        outputs = {
            "locomotion": self.locomotion_head(recurrent),
            "turning": self.turning_head(recurrent),
            "attack": self.attack_head(recurrent).squeeze(-1),
            "use": self.use_head(recurrent).squeeze(-1),
        }
        if squeeze_time:
            outputs = {name: value[:, 0] for name, value in outputs.items()}
        return outputs, new_hidden


class EngineStateMLPPolicy(nn.Module):
    """Purely reactive policy for an already-Markovian engine-state vector."""

    def __init__(self, input_size: int, hidden_size: int = 192, num_layers: int = 2):
        super().__init__()
        if input_size <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError("Policy dimensions must be positive.")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        layers: list[nn.Module] = [nn.LayerNorm(input_size)]
        current = input_size
        for _ in range(num_layers):
            layers.extend((nn.Linear(current, hidden_size), nn.ReLU()))
            current = hidden_size
        self.encoder = nn.Sequential(*layers)
        self.locomotion_head = nn.Linear(hidden_size, 5)
        self.turning_head = nn.Linear(hidden_size, 3)
        self.attack_head = nn.Linear(hidden_size, 1)
        self.use_head = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor, hidden=None):
        if inputs.ndim not in (2, 3) or inputs.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected (..., {self.input_size}) inputs, got {tuple(inputs.shape)}."
            )
        encoded = self.encoder(inputs)
        return {
            "locomotion": self.locomotion_head(encoded),
            "turning": self.turning_head(encoded),
            "attack": self.attack_head(encoded).squeeze(-1),
            "use": self.use_head(encoded).squeeze(-1),
        }, None


class EngineStateSpikingPolicy(nn.Module):
    """Recurrent spiking neural network policy for privileged MAP01 engine state."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_steps: int = 4,
        beta: float = 0.9,
    ):
        super().__init__()
        if input_size <= 0 or hidden_size <= 0 or num_layers <= 0 or num_steps <= 0:
            raise ValueError("Policy dimensions and num_steps must be positive.")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_steps = int(num_steps)
        self.beta = float(beta)

        self.encoder = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
        )
        self.recurrent = nn.Linear(hidden_size, hidden_size)
        self.lif = snn.Leaky(beta=beta, spike_grad=snn.surrogate.fast_sigmoid())

        self.locomotion_head = nn.Linear(hidden_size, 5)
        self.turning_head = nn.Linear(hidden_size, 3)
        self.attack_head = nn.Linear(hidden_size, 1)
        self.use_head = nn.Linear(hidden_size, 1)

        self.lif_locomotion = snn.Leaky(beta=beta, reset_mechanism="none")
        self.lif_turning = snn.Leaky(beta=beta, reset_mechanism="none")
        self.lif_attack = snn.Leaky(beta=beta, reset_mechanism="none")
        self.lif_use = snn.Leaky(beta=beta, reset_mechanism="none")

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        squeeze_time = inputs.ndim == 2
        if squeeze_time:
            inputs = inputs.unsqueeze(1)
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected input shape (B, T, {self.input_size}) or "
                f"(B, {self.input_size}), got {tuple(inputs.shape)}."
            )
        batch, seq_len, _ = inputs.shape

        if hidden is None:
            spk = torch.zeros(batch, self.hidden_size, device=inputs.device, dtype=inputs.dtype)
            mem = torch.zeros(batch, self.hidden_size, device=inputs.device, dtype=inputs.dtype)
        else:
            spk, mem = hidden

        locomotion_steps = []
        turning_steps = []
        attack_steps = []
        use_steps = []

        for t in range(seq_len):
            cur_in = self.encoder(inputs[:, t])
            mem_loco = torch.zeros(batch, 5, device=inputs.device, dtype=inputs.dtype)
            mem_turn = torch.zeros(batch, 3, device=inputs.device, dtype=inputs.dtype)
            mem_attack = torch.zeros(batch, 1, device=inputs.device, dtype=inputs.dtype)
            mem_use = torch.zeros(batch, 1, device=inputs.device, dtype=inputs.dtype)

            for _ in range(self.num_steps):
                cur_rec = cur_in + self.recurrent(spk)
                spk, mem = self.lif(cur_rec, mem)

                _, mem_loco = self.lif_locomotion(self.locomotion_head(spk), mem_loco)
                _, mem_turn = self.lif_turning(self.turning_head(spk), mem_turn)
                _, mem_attack = self.lif_attack(self.attack_head(spk), mem_attack)
                _, mem_use = self.lif_use(self.use_head(spk), mem_use)

            locomotion_steps.append(mem_loco)
            turning_steps.append(mem_turn)
            attack_steps.append(mem_attack.squeeze(-1))
            use_steps.append(mem_use.squeeze(-1))

        outputs = {
            "locomotion": torch.stack(locomotion_steps, dim=1),
            "turning": torch.stack(turning_steps, dim=1),
            "attack": torch.stack(attack_steps, dim=1),
            "use": torch.stack(use_steps, dim=1),
        }
        if squeeze_time:
            outputs = {name: value[:, 0] for name, value in outputs.items()}
        return outputs, (spk, mem)

