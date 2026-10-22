"""Unit tests for the Multi-Task Visual Perception Encoder."""

from __future__ import annotations

import pytest
import torch

from agent_config import FRAME_HEIGHT, FRAME_STACK, FRAME_WIDTH
from networks.vision_encoder import VisionPerceptionEncoder


def test_vision_encoder_shapes():
    model = VisionPerceptionEncoder()
    batch_size = 4
    dummy_input = torch.rand(batch_size, FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH)

    outputs = model(dummy_input)
    assert outputs.combat.shape == (batch_size, 7)
    assert outputs.navigation.shape == (batch_size, 5)
    assert outputs.interaction.shape == (batch_size, 1)
    assert outputs.embedding.shape == (batch_size, 256)


def test_vision_encoder_rgb_shapes():
    model = VisionPerceptionEncoder(in_channels=12)
    batch_size = 4
    dummy_input = torch.rand(batch_size, 12, FRAME_HEIGHT, FRAME_WIDTH)

    outputs = model(dummy_input)
    assert outputs.combat.shape == (batch_size, 7)
    assert outputs.navigation.shape == (batch_size, 5)
    assert outputs.interaction.shape == (batch_size, 1)
    assert outputs.embedding.shape == (batch_size, 256)


def test_vision_encoder_decode_predictions():
    model = VisionPerceptionEncoder()
    dummy_input = torch.rand(2, FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH)

    decoded = model.decode_predictions(dummy_input)

    assert "enemy_visible_prob" in decoded
    assert "enemy_screen_x" in decoded
    assert "enemy_bearing_sin" in decoded
    assert "route_heading_error_cos" in decoded
    assert "use_recommended_prob" in decoded

    # Bounds check
    assert (decoded["enemy_visible_prob"] >= 0.0).all() and (decoded["enemy_visible_prob"] <= 1.0).all()
    assert (decoded["enemy_distance"] >= 0.0).all() and (decoded["enemy_distance"] <= 1.0).all()

    # Unit vector check: sin^2 + cos^2 == 1.0
    sin = decoded["enemy_bearing_sin"]
    cos = decoded["enemy_bearing_cos"]
    norm_sq = sin ** 2 + cos ** 2
    assert torch.allclose(norm_sq, torch.ones_like(norm_sq), atol=1e-5)


def test_vision_encoder_backward_gradients():
    model = VisionPerceptionEncoder()
    dummy_input = torch.rand(2, FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH, requires_grad=True)

    outputs = model(dummy_input)
    loss = (
        outputs.combat.sum()
        + outputs.navigation.sum()
        + outputs.interaction.sum()
    )
    loss.backward()

    # Ensure gradients propagate to conv backbone and all heads
    assert model.conv1.weight.grad is not None
    assert model.conv3.weight.grad is not None
    assert model.combat_head[-1].weight.grad is not None
    assert model.navigation_head[-1].weight.grad is not None
    assert model.interaction_head[-1].weight.grad is not None


def test_vision_encoder_rejects_bad_input():
    model = VisionPerceptionEncoder()
    bad_channels = torch.rand(2, 3, FRAME_HEIGHT, FRAME_WIDTH)
    with pytest.raises(ValueError, match="Expected frames of shape"):
        model(bad_channels)
