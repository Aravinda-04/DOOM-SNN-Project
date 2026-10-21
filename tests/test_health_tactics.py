"""Unit tests for the HealthTacticsArbitrator."""

from __future__ import annotations

import pytest

from engine_state_observation import VisibleEnemy, VisibleHealthItem
from health_tactics import HealthSubgoal, HealthTacticsArbitrator


def _mock_item(name="Medikit", distance=100.0, bearing=10.0, x=500.0, y=-200.0, heal=25, priority=1.0):
    return VisibleHealthItem(
        name=name,
        heal_amount=heal,
        priority=priority,
        distance=distance,
        bearing_degrees=bearing,
        position=(x, y),
        screen_x=0.5,
        screen_area=0.05,
    )


def _mock_enemy(distance=100.0, bearing=0.0):
    return VisibleEnemy(
        name="Shotgunguy",
        distance=distance,
        bearing_degrees=bearing,
        screen_x=0.5,
        screen_area=0.05,
    )


def test_no_detour_when_health_healthy():
    arbitrator = HealthTacticsArbitrator(health_threshold=50.0)
    item = _mock_item()
    subgoal = arbitrator.step(
        player_health=80.0,
        visible_items=[item],
        nearest_enemy=None,
        player_angle=0.0,
    )
    assert subgoal is None


def test_detour_selected_when_health_low():
    arbitrator = HealthTacticsArbitrator(health_threshold=50.0)
    item = _mock_item(name="Medikit", distance=120.0, bearing=15.0, x=600.0, y=-150.0)
    subgoal = arbitrator.step(
        player_health=35.0,
        visible_items=[item],
        nearest_enemy=None,
        player_angle=0.0,
    )
    assert subgoal is not None
    assert subgoal.name == "Medikit"
    assert subgoal.position == (600.0, -150.0)
    assert subgoal.heal_amount == 25


def test_combat_threat_preempts_health_detour():
    arbitrator = HealthTacticsArbitrator(health_threshold=50.0, combat_threat_distance=128.0)
    item = _mock_item(distance=80.0, bearing=10.0)
    # Enemy is 90 units away right in crosshairs (0 degrees)
    enemy = _mock_enemy(distance=90.0, bearing=0.0)
    subgoal = arbitrator.step(
        player_health=25.0,
        visible_items=[item],
        nearest_enemy=enemy,
        player_angle=0.0,
    )
    # Must preempt to allow shotgun flinch!
    assert subgoal is None


def test_forward_cone_filters_rear_items():
    arbitrator = HealthTacticsArbitrator(health_threshold=50.0, max_bearing_error=75.0)
    # Item is directly behind player (bearing 180 when player faces 0)
    item = _mock_item(bearing=180.0)
    subgoal = arbitrator.step(
        player_health=30.0,
        visible_items=[item],
        nearest_enemy=None,
        player_angle=0.0,
    )
    assert subgoal is None


def test_acid_pit_items_rejected():
    arbitrator = HealthTacticsArbitrator(health_threshold=50.0)
    # Inside MAP01 toxic nukage pit (X=700, Y=-900)
    item = _mock_item(x=700.0, y=-900.0, distance=80.0, bearing=0.0)
    subgoal = arbitrator.step(
        player_health=30.0,
        visible_items=[item],
        nearest_enemy=None,
        player_angle=0.0,
    )
    assert subgoal is None


def test_detour_timeout_blacklists_unreachable_item():
    arbitrator = HealthTacticsArbitrator(health_threshold=50.0, max_pursuit_ticks=5, blackout_ticks=20)
    item = _mock_item(distance=100.0, bearing=0.0)
    # Step 5 times (within max_pursuit_ticks)
    for _ in range(5):
        subgoal = arbitrator.step(
            player_health=30.0,
            visible_items=[item],
            nearest_enemy=None,
            player_angle=0.0,
        )
        assert subgoal is not None

    # Step 6: Exceeds max_pursuit_ticks! Should time out and blacklist
    subgoal_timed_out = arbitrator.step(
        player_health=30.0,
        visible_items=[item],
        nearest_enemy=None,
        player_angle=0.0,
    )
    assert subgoal_timed_out is None
    assert arbitrator.is_blacklisted(item.position) is True
