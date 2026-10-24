"""Unit tests for GeneralizedRoute and SNN contract compatibility (src/generalized_route.py)."""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from engine_state_observation import EngineStateBuilder
from engine_state_config import FEATURE_INDEX, FEATURE_NAMES
from map01_route import Map01Route, RouteContext
from generalized_route import GeneralizedRoute

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FREEDOOM2_WAD = PROJECT_ROOT / "freedoom2.wad"


def test_generalized_route_map01_construction_and_context():
    """Verify GeneralizedRoute on MAP01 generates compliant RouteContext instances."""
    route = GeneralizedRoute.from_wad(FREEDOOM2_WAD, "MAP01")

    assert len(route.points) >= 15
    assert len(route.use_points) >= 4

    # Context at spawn: because player is already within waypoint_radius of point 0,
    # route tracker immediately targets point 1 (progress_index = 1).
    spawn_pos = route.points[0]
    ctx = route.context(spawn_pos, 0)

    assert isinstance(ctx, RouteContext)
    assert ctx.progress_index in (0, 1)
    assert 0.0 <= ctx.progress_fraction <= 0.1
    assert not ctx.off_track
    assert not ctx.use_recommended

    # Door interaction recommendation: near door at (216, -616)
    door_ctx = route.context(np.array([216.0, -616.0], dtype=np.float32), 30)
    assert door_ctx.use_recommended is True


def test_generalized_route_map02_zero_shot():
    """Verify GeneralizedRoute works zero-shot on MAP02 with no demonstrations."""
    route = GeneralizedRoute.from_wad(FREEDOOM2_WAD, "MAP02")

    assert len(route.points) >= 10
    assert len(route.use_points) >= 5

    # Start at MAP02 spawn (640, -576)
    spawn_ctx = route.context(np.array([640.0, -576.0], dtype=np.float32), 0)
    assert isinstance(spawn_ctx, RouteContext)
    assert 0.0 <= spawn_ctx.progress_fraction <= 0.1

    # Waypoint advancement when positioned at waypoint 1
    next_pos = route.points[1]
    adv_ctx = route.context(next_pos, 0)
    assert adv_ctx.progress_index >= 1


def test_dynamic_health_detour_reroute():
    """Verify runtime re-routing via nearest Medikit modifies waypoints."""
    route = GeneralizedRoute.from_wad(FREEDOOM2_WAD, "MAP01")

    # Request health detour near spawn
    success = route.request_health_detour((-192.0, -192.0))
    assert success is True
    assert len(route.points) > 0


def test_snn_engine_state_builder_integration():
    """Verify that EngineStateBuilder encodes GeneralizedRoute context into exact 84D features."""
    route = GeneralizedRoute.from_wad(FREEDOOM2_WAD, "MAP01")
    builder = EngineStateBuilder()

    diagnostics = {
        "position_x": -192.0,
        "position_y": -192.0,
        "angle": 0.0,
        "health": 100.0,
        "armor": 0.0,
        "ammo2": 50.0,
        "ammo3": 0.0,
        "kill_count": 0.0,
        "item_count": 0.0,
        "episode_time": 100,
    }

    ctx = route.context(np.array([-192.0, -192.0], dtype=np.float32), 0)
    obs = builder.observe(diagnostics, labels=[], route=ctx)

    assert len(obs.features) == len(FEATURE_NAMES)
    assert len(obs.validity) == len(FEATURE_NAMES)
    assert obs.features.shape == (42,)
    assert obs.validity.shape == (42,)

    # Full 84D policy input contract
    snn_inputs = np.concatenate((obs.features, obs.validity.astype(np.float32)))
    assert snn_inputs.shape == (84,)

    # Verify route feature fields
    dx_idx = FEATURE_INDEX["route_dx"]
    dy_idx = FEATURE_INDEX["route_dy"]
    dist_idx = FEATURE_INDEX["route_distance"]
    prog_idx = FEATURE_INDEX["route_progress"]
    use_idx = FEATURE_INDEX["use_recommended"]

    assert -1.0 <= obs.features[dx_idx] <= 1.0
    assert -1.0 <= obs.features[dy_idx] <= 1.0
    assert 0.0 <= obs.features[dist_idx] <= 1.0
    assert 0.0 <= obs.features[prog_idx] <= 0.1
    assert obs.features[use_idx] == 0.0
