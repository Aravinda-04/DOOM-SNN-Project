"""Multi-task Visual Perception Encoder for bridging raw pixels to MAP01 state."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_config import FRAME_HEIGHT, FRAME_STACK, FRAME_WIDTH


class VisionPerceptionOutputs(NamedTuple):
    combat: torch.Tensor       # (B, 7): [vis_logit, screen_x, area, dist, sin, cos, cone_logit]
    navigation: torch.Tensor   # (B, 5): [dist, bearing_sin, bearing_cos, heading_err_sin, heading_err_cos]
    interaction: torch.Tensor  # (B, 1): [use_logit]
    embedding: torch.Tensor    # (B, 256): latent visual feature vector


class VisionPerceptionEncoder(nn.Module):
    """Convolutional perception encoder predicting structured scene variables from 4-frame stacks."""

    def __init__(
        self,
        in_channels: int = FRAME_STACK,
        embedding_dim: int = 256,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.embedding_dim = int(embedding_dim)

        # 4-layer Convolutional Backbone
        # Input: (B, 4, 84, 84)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)  # -> (B, 32, 20, 20)
        self.gn1 = nn.GroupNorm(4, 32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)           # -> (B, 64, 9, 9)
        self.gn2 = nn.GroupNorm(8, 64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)           # -> (B, 64, 7, 7)
        self.gn3 = nn.GroupNorm(8, 64)

        # Flattened spatial features: 64 * 7 * 7 = 3136
        self.fc_latent = nn.Linear(64 * 7 * 7, embedding_dim)
        self.ln_latent = nn.LayerNorm(embedding_dim)
        self.drop_conv = nn.Dropout2d(p=0.15)
        self.drop_fc = nn.Dropout(p=0.2)

        # Factorized Perception Heads
        # 1. Combat Head: 7 signals
        # [enemy_visible_logit, screen_x, screen_area, distance, bearing_sin, bearing_cos, cone_logit]
        self.combat_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(128, 7),
        )

        # 2. Navigation Head: 5 signals
        # [route_dist, bearing_sin, bearing_cos, heading_err_sin, heading_err_cos]
        self.navigation_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(128, 5),
        )

        # 3. Interaction Head: 1 signal
        # [use_recommended_logit]
        self.interaction_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def extract_embedding(self, frames: torch.Tensor) -> torch.Tensor:
        """Process (B, 4, 84, 84) frame tensor into a 256-dim latent vector."""
        if frames.ndim != 4 or frames.shape[1:] != (self.in_channels, FRAME_HEIGHT, FRAME_WIDTH):
            raise ValueError(
                f"Expected frames of shape (B, {self.in_channels}, {FRAME_HEIGHT}, {FRAME_WIDTH}), "
                f"got {tuple(frames.shape)}."
            )
        x = F.gelu(self.gn1(self.conv1(frames)))
        x = F.gelu(self.gn2(self.conv2(x)))
        x = self.drop_conv(x)
        x = F.gelu(self.gn3(self.conv3(x)))
        x = self.drop_conv(x)
        x = torch.flatten(x, start_dim=1)
        z = F.gelu(self.ln_latent(self.fc_latent(x)))
        z = self.drop_fc(z)
        return z

    def forward(self, frames: torch.Tensor) -> VisionPerceptionOutputs:
        z = self.extract_embedding(frames)
        combat = self.combat_head(z)
        navigation = self.navigation_head(z)
        interaction = self.interaction_head(z)
        return VisionPerceptionOutputs(
            combat=combat,
            navigation=navigation,
            interaction=interaction,
            embedding=z,
        )

    def decode_predictions(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inference helper converting raw logits into bounded, normalized physical estimates."""
        outputs = self.forward(frames)
        combat_raw = outputs.combat
        nav_raw = outputs.navigation
        inter_raw = outputs.interaction

        # Normalize combat azimuth unit vector
        bearing_vec = combat_raw[:, 4:6]
        bearing_norm = F.normalize(bearing_vec, p=2, dim=-1)

        # Normalize navigation unit vectors
        route_bearing = F.normalize(nav_raw[:, 1:3], p=2, dim=-1)
        heading_err = F.normalize(nav_raw[:, 3:5], p=2, dim=-1)

        return {
            # Combat
            "enemy_visible_prob": torch.sigmoid(combat_raw[:, 0]),
            "enemy_screen_x": torch.sigmoid(combat_raw[:, 1]),
            "enemy_screen_area": torch.sigmoid(combat_raw[:, 2]),
            "enemy_distance": torch.clamp(combat_raw[:, 3], 0.0, 1.0),
            "enemy_bearing_sin": bearing_norm[:, 0],
            "enemy_bearing_cos": bearing_norm[:, 1],
            "enemy_in_firing_cone_prob": torch.sigmoid(combat_raw[:, 6]),
            # Navigation
            "route_distance": torch.clamp(nav_raw[:, 0], 0.0, 1.0),
            "route_bearing_sin": route_bearing[:, 0],
            "route_bearing_cos": route_bearing[:, 1],
            "route_heading_error_sin": heading_err[:, 0],
            "route_heading_error_cos": heading_err[:, 1],
            # Interaction
            "use_recommended_prob": torch.sigmoid(inter_raw[:, 0]),
            # Raw embedding
            "embedding": outputs.embedding,
        }
