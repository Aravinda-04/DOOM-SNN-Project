from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from engine_state_config import (
    BUTTON_NAMES,
    ENGINE_STATE_CONTRACT_HASH,
    FEATURE_NAMES,
    FactorizedAction,
    Locomotion,
    Turning,
    action_targets,
    compose_buttons,
)
from engine_state_demonstrations import (
    ENGINE_STATE_SCHEMA_VERSION,
    load_sequence,
    save_sequence,
)
from engine_state_expert import EngineStateExpert
from engine_state_observation import EngineStateBuilder, visible_enemies
from evaluate_engine_state_policy import predicted_action
from map01_route import Map01Route, angle_delta_degrees
from networks.engine_state_policy import (
    EngineStateGRUPolicy,
    EngineStateMLPPolicy,
    EngineStateSpikingPolicy,
)
from train_engine_state_policy import policy_loss


def _diagnostics(**overrides):
    values = {
        "position_x": 0.0,
        "position_y": 0.0,
        "angle": 0.0,
        "health": 100.0,
        "armor": 0.0,
        "ammo2": 50.0,
        "ammo3": 0.0,
        "kill_count": 0.0,
        "item_count": 0.0,
        "episode_time": 0,
    }
    values.update(overrides)
    return values


def _enemy(*, x=100.0, y=0.0, screen_x=75, width=10, height=20):
    return SimpleNamespace(
        object_name="Zombieman",
        object_position_x=x,
        object_position_y=y,
        x=screen_x,
        y=40,
        width=width,
        height=height,
    )


def test_factorized_action_composes_independent_buttons():
    action = FactorizedAction(
        locomotion=Locomotion.FORWARD,
        turning=Turning.LEFT,
        attack=True,
    )
    buttons = dict(zip(BUTTON_NAMES, compose_buttons(action)))
    assert buttons["MOVE_FORWARD"] is True
    assert buttons["TURN_LEFT"] is True
    assert buttons["ATTACK"] is True
    assert buttons["TURN_RIGHT"] is False
    assert action_targets(action).tolist() == [1, 1, 1, 0]


def test_route_context_advances_and_recommends_use():
    route = Map01Route(
        np.asarray(((0, 0), (20, 0), (40, 0), (60, 0)), dtype=np.float32),
        source_sha256="a" * 64,
        use_points=np.asarray(((40, 0),), dtype=np.float32),
    )
    context = route.context(np.asarray((21, 0), dtype=np.float32), 0)
    assert context.progress_index >= 1
    assert context.progress_fraction > 0
    assert context.use_recommended is True
    assert angle_delta_degrees(5, 355) == pytest.approx(10)


def test_sparse_route_exposes_graph_edges_and_recovery():
    route = Map01Route(
        np.asarray(((0, 0), (100, 0), (200, 0), (300, 0)), dtype=np.float32),
        source_sha256="f" * 64,
        use_points=np.asarray(((1000, 1000),), dtype=np.float32),
    )
    assert route.edges == ((0, 1), (1, 2), (2, 3))
    context = route.context(np.asarray((100, 200), dtype=np.float32), 1)
    assert context.off_track
    assert context.recovery


def test_engine_state_marks_enemy_fields_invalid_when_no_enemy():
    route = Map01Route(
        np.asarray(((0, 0), (100, 0)), dtype=np.float32),
        source_sha256="b" * 64,
        use_points=np.asarray(((1000, 1000),), dtype=np.float32),
    )
    context = route.context(np.asarray((0, 0), dtype=np.float32), 0)
    builder = EngineStateBuilder(stuck_window=3)
    observation = builder.observe(_diagnostics(), [], context)
    assert observation.features.shape == (len(FEATURE_NAMES),)
    assert observation.validity.shape == observation.features.shape
    assert observation.nearest_enemy is None
    assert observation.validity.sum() == len(FEATURE_NAMES) - 6


def test_visible_enemy_world_bearing_and_expert_attack():
    enemies = visible_enemies([_enemy()], player_x=0.0, player_y=0.0)
    assert len(enemies) == 1
    assert enemies[0].distance == pytest.approx(100.0)
    assert enemies[0].bearing_degrees == pytest.approx(0.0)

    route = Map01Route(
        np.asarray(((0, 0), (100, 0)), dtype=np.float32),
        source_sha256="c" * 64,
        use_points=np.asarray(((1000, 1000),), dtype=np.float32),
    )
    context = route.context(np.asarray((0, 0), dtype=np.float32), 0)
    observation = EngineStateBuilder().observe(
        _diagnostics(), [_enemy()], context
    )
    action = EngineStateExpert().act(
        observation,
        context,
        player_angle=0.0,
        ammunition=50.0,
    )
    assert action.attack is True


def test_engine_state_sequence_round_trip(tmp_path):
    length = 4
    arrays = {
        "features": np.zeros((length, len(FEATURE_NAMES)), dtype=np.float32),
        "feature_valid": np.ones((length, len(FEATURE_NAMES)), dtype=np.bool_),
        "targets": np.zeros((length, 4), dtype=np.int16),
        "applied_targets": np.zeros((length, 4), dtype=np.int16),
        "rewards": np.zeros(length, dtype=np.float32),
        "dones": np.asarray((False, False, False, True), dtype=np.bool_),
        "positions": np.zeros((length, 3), dtype=np.float32),
        "route_progress": np.linspace(0, 1, length, dtype=np.float32),
    }
    metadata = {
        "schema_version": ENGINE_STATE_SCHEMA_VERSION,
        "engine_state_contract_hash": ENGINE_STATE_CONTRACT_HASH,
    }
    path = tmp_path / "episode.npz"
    digest = save_sequence(path, metadata=metadata, **arrays)
    payload = load_sequence(path)
    assert len(digest) == 64
    assert np.array_equal(payload["features"], arrays["features"])
    assert payload["metadata"] == metadata


def test_engine_state_policy_all_heads_backpropagate():
    model = EngineStateGRUPolicy(input_size=len(FEATURE_NAMES) * 2, hidden_size=16)
    inputs = torch.randn(3, 5, len(FEATURE_NAMES) * 2)
    targets = torch.zeros(3, 5, 4, dtype=torch.long)
    targets[:, 2:, 2] = 1
    targets[:, 4:, 3] = 1
    outputs, hidden = model(inputs)
    loss, losses = policy_loss(
        outputs,
        targets,
        locomotion_weights=torch.ones(5),
        turning_weights=torch.ones(3),
        attack_positive_weight=torch.tensor(2.0),
        use_positive_weight=torch.tensor(3.0),
    )
    loss.backward()
    assert hidden.shape == (1, 3, 16)
    assert set(losses) == {"locomotion", "turning", "attack", "use"}
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_reactive_mlp_preserves_sequence_shape_without_hidden_state():
    model = EngineStateMLPPolicy(
        input_size=len(FEATURE_NAMES) * 2, hidden_size=16, num_layers=2
    )
    outputs, hidden = model(torch.randn(3, 5, len(FEATURE_NAMES) * 2))
    assert hidden is None
    assert outputs["locomotion"].shape == (3, 5, 5)
    assert outputs["attack"].shape == (3, 5)


def test_policy_decoder_does_not_inject_use_at_interaction_point():
    outputs = {
        "locomotion": torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0]]),
        "turning": torch.tensor([[0.0, 5.0, 0.0]]),
        "attack": torch.tensor([-5.0]),
        "use": torch.tensor([5.0]),
    }
    route = Map01Route(
        np.asarray(((0, 0), (10, 0)), dtype=np.float32),
        source_sha256="d" * 64,
        use_points=np.asarray(((0, 0),), dtype=np.float32),
    ).context(np.asarray((0, 0), dtype=np.float32), 0)
    action = predicted_action(outputs, route=route)
    assert action == FactorizedAction(locomotion=Locomotion.FORWARD, use=True)


def test_policy_decoder_does_not_inject_missing_use():
    outputs = {
        "locomotion": torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0]]),
        "turning": torch.tensor([[5.0, 0.0, 0.0]]),
        "attack": torch.tensor([-5.0]),
        "use": torch.tensor([-5.0]),
    }
    route = Map01Route(
        np.asarray(((0, 0), (10, 0)), dtype=np.float32),
        source_sha256="e" * 64,
        use_points=np.asarray(((0, 0),), dtype=np.float32),
    ).context(np.asarray((0, 0), dtype=np.float32), 0)
    assert route.use_recommended
    assert predicted_action(outputs, route=route) == FactorizedAction(
        locomotion=Locomotion.FORWARD
    )


def test_engine_state_spiking_policy_forward_and_backward():
    model = EngineStateSpikingPolicy(
        input_size=len(FEATURE_NAMES) * 2, hidden_size=16, num_steps=2
    )
    # Test 2D input (single step inference)
    inputs_2d = torch.randn(2, len(FEATURE_NAMES) * 2)
    outputs_2d, hidden = model(inputs_2d)
    assert outputs_2d["locomotion"].shape == (2, 5)
    assert outputs_2d["turning"].shape == (2, 3)
    assert outputs_2d["attack"].shape == (2,)
    assert outputs_2d["use"].shape == (2,)
    assert isinstance(hidden, tuple) and len(hidden) == 2

    # Test 3D input (sequence window training) with carried hidden state
    inputs_3d = torch.randn(2, 4, len(FEATURE_NAMES) * 2)
    outputs_3d, next_hidden = model(inputs_3d, hidden)
    assert outputs_3d["locomotion"].shape == (2, 4, 5)
    assert outputs_3d["turning"].shape == (2, 4, 3)
    assert outputs_3d["attack"].shape == (2, 4)
    assert outputs_3d["use"].shape == (2, 4)

    # Test backpropagation through factorized loss
    targets = torch.zeros(2, 4, 4, dtype=torch.long)
    targets[:, 1:, 2] = 1
    loss, losses = policy_loss(
        outputs_3d,
        targets,
        locomotion_weights=torch.ones(5),
        turning_weights=torch.ones(3),
        attack_positive_weight=torch.tensor(1.0),
        use_positive_weight=torch.tensor(1.0),
    )
    loss.backward()
    assert all(
        param.grad is not None
        for param in model.parameters()
        if param.requires_grad
    )

