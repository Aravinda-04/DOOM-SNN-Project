"""Compact recurrent policy for privileged MAP01 engine state."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


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
