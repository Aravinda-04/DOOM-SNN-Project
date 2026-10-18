"""Causal privileged-state extraction for the MAP01 policy."""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from engine_state_config import (
    FEATURE_INDEX,
    FEATURE_NAMES,
    FactorizedAction,
    Locomotion,
    Turning,
    validate_feature_vector,
)
from map01_route import RouteContext, angle_delta_degrees


HOSTILE_NAME_TOKENS = (
    "zombieman",
    "shotgunguy",
    "chaingunner",
    "imp",
    "demon",
    "spectre",
    "lostsoul",
    "cacodemon",
    "hellknight",
    "baron",
    "revenant",
    "mancubus",
    "arachnotron",
    "painelemental",
    "archvile",
    "wolfenstein",
)


@dataclass(frozen=True)
class VisibleEnemy:
    name: str
    distance: float
    bearing_degrees: float
    screen_x: float
    screen_area: float


@dataclass(frozen=True)
class EngineStateObservation:
    features: np.ndarray
    validity: np.ndarray
    nearest_enemy: Optional[VisibleEnemy]
    visible_enemy_count: int
    stuck_steps: int


def _is_hostile_label(label: object) -> bool:
    name = str(getattr(label, "object_name", "")).lower()
    return not name.startswith("dead") and any(
        token in name for token in HOSTILE_NAME_TOKENS
    )


def visible_enemies(
    labels: Iterable[object],
    *,
    player_x: float,
    player_y: float,
) -> list[VisibleEnemy]:
    enemies: list[VisibleEnemy] = []
    for label in labels:
        if not _is_hostile_label(label):
            continue
        enemy_x = float(getattr(label, "object_position_x", player_x))
        enemy_y = float(getattr(label, "object_position_y", player_y))
        dx = enemy_x - player_x
        dy = enemy_y - player_y
        width = max(float(getattr(label, "width", 0.0)), 0.0)
        height = max(float(getattr(label, "height", 0.0)), 0.0)
        left = float(getattr(label, "x", 80.0))
        enemies.append(
            VisibleEnemy(
                name=str(getattr(label, "object_name", "unknown")),
                distance=float(math.hypot(dx, dy)),
                bearing_degrees=float(math.degrees(math.atan2(dy, dx)) % 360.0),
                screen_x=float(np.clip((left + width / 2.0) / 160.0, 0.0, 1.0)),
                screen_area=float(np.clip(width * height / (160.0 * 120.0), 0.0, 1.0)),
            )
        )
    return sorted(enemies, key=lambda enemy: enemy.distance)


class EngineStateBuilder:
    """Build normalized state while retaining only causal short-term history."""

    def __init__(self, *, stuck_window: int = 25, stuck_distance: float = 10.0):
        if stuck_window < 2:
            raise ValueError("stuck_window must be at least 2.")
        if stuck_distance <= 0:
            raise ValueError("stuck_distance must be positive.")
        self.stuck_window = int(stuck_window)
        self.stuck_distance = float(stuck_distance)
        self._positions: collections.deque[np.ndarray] = collections.deque(
            maxlen=self.stuck_window
        )
        self._previous_angle: Optional[float] = None
        self._stuck_steps = 0

    def reset(self) -> None:
        self._positions.clear()
        self._previous_angle = None
        self._stuck_steps = 0

    def observe(
        self,
        diagnostics: dict,
        labels: Iterable[object],
        route: RouteContext,
        previous_action: FactorizedAction = FactorizedAction(),
    ) -> EngineStateObservation:
        x = float(diagnostics["position_x"])
        y = float(diagnostics["position_y"])
        angle = float(diagnostics["angle"]) % 360.0
        position = np.asarray((x, y), dtype=np.float32)
        previous_position = self._positions[-1] if self._positions else position
        delta = position - previous_position
        self._positions.append(position)
        if (
            len(self._positions) == self.stuck_window
            and float(np.linalg.norm(self._positions[-1] - self._positions[0]))
            < self.stuck_distance
        ):
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0

        previous_angle = angle if self._previous_angle is None else self._previous_angle
        angular_delta = math.radians(angle_delta_degrees(angle, previous_angle))
        self._previous_angle = angle

        enemies = visible_enemies(labels, player_x=x, player_y=y)
        nearest = enemies[0] if enemies else None
        heading_radians = math.radians(angle)
        route_bearing_radians = math.radians(route.bearing_degrees)
        route_error_radians = math.radians(
            angle_delta_degrees(route.bearing_degrees, angle)
        )
        values = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        valid = np.ones(len(FEATURE_NAMES), dtype=np.bool_)

        def set_value(name: str, value: float) -> None:
            values[FEATURE_INDEX[name]] = float(value)

        set_value("position_x", np.clip((x + 384.0) / 2688.0, 0.0, 1.0))
        set_value("position_y", np.clip((y + 1408.0) / 1920.0, 0.0, 1.0))
        set_value("heading_sin", math.sin(heading_radians))
        set_value("heading_cos", math.cos(heading_radians))
        set_value("delta_x", np.clip(delta[0] / 64.0, -1.0, 1.0))
        set_value("delta_y", np.clip(delta[1] / 64.0, -1.0, 1.0))
        set_value("delta_angle_sin", math.sin(angular_delta))
        set_value("delta_angle_cos", math.cos(angular_delta))
        set_value("health", np.clip(float(diagnostics["health"]) / 100.0, 0.0, 2.0))
        set_value("armor", np.clip(float(diagnostics["armor"]) / 200.0, 0.0, 1.0))
        set_value("ammo_bullets", np.clip(float(diagnostics["ammo2"]) / 200.0, 0.0, 1.0))
        set_value("ammo_shells", np.clip(float(diagnostics["ammo3"]) / 50.0, 0.0, 1.0))
        set_value("kill_count", np.clip(float(diagnostics["kill_count"]) / 20.0, 0.0, 1.0))
        set_value("item_count", np.clip(float(diagnostics["item_count"]) / 20.0, 0.0, 1.0))
        set_value("time_fraction", np.clip(float(diagnostics["episode_time"]) / 21000.0, 0.0, 1.0))
        set_value("stuck_fraction", np.clip(self._stuck_steps / self.stuck_window, 0.0, 1.0))
        set_value("enemy_visible", float(nearest is not None))
        set_value("enemy_count", np.clip(len(enemies) / 8.0, 0.0, 1.0))

        enemy_fields = (
            "enemy_distance",
            "enemy_bearing_sin",
            "enemy_bearing_cos",
            "enemy_screen_x",
            "enemy_screen_area",
            "enemy_in_firing_cone",
        )
        if nearest is None:
            for name in enemy_fields:
                valid[FEATURE_INDEX[name]] = False
        else:
            enemy_bearing = math.radians(nearest.bearing_degrees)
            enemy_error = abs(angle_delta_degrees(nearest.bearing_degrees, angle))
            set_value("enemy_distance", np.clip(nearest.distance / 1024.0, 0.0, 1.0))
            set_value("enemy_bearing_sin", math.sin(enemy_bearing))
            set_value("enemy_bearing_cos", math.cos(enemy_bearing))
            set_value("enemy_screen_x", nearest.screen_x)
            set_value("enemy_screen_area", nearest.screen_area)
            set_value("enemy_in_firing_cone", float(enemy_error <= 5.0))

        set_value("route_dx", np.clip(float(route.delta[0]) / 512.0, -1.0, 1.0))
        set_value("route_dy", np.clip(float(route.delta[1]) / 512.0, -1.0, 1.0))
        set_value("route_distance", np.clip(route.distance / 512.0, 0.0, 1.0))
        set_value("route_bearing_sin", math.sin(route_bearing_radians))
        set_value("route_bearing_cos", math.cos(route_bearing_radians))
        set_value("route_heading_error_sin", math.sin(route_error_radians))
        set_value("route_heading_error_cos", math.cos(route_error_radians))
        set_value("route_progress", route.progress_fraction)
        set_value("route_off_track", float(route.off_track))
        set_value("use_recommended", float(route.use_recommended))

        set_value("previous_forward", float(previous_action.locomotion == Locomotion.FORWARD))
        set_value("previous_backward", float(previous_action.locomotion == Locomotion.BACKWARD))
        set_value("previous_strafe_left", float(previous_action.locomotion == Locomotion.STRAFE_LEFT))
        set_value("previous_strafe_right", float(previous_action.locomotion == Locomotion.STRAFE_RIGHT))
        set_value("previous_turn_left", float(previous_action.turning == Turning.LEFT))
        set_value("previous_turn_right", float(previous_action.turning == Turning.RIGHT))
        set_value("previous_attack", float(previous_action.attack))
        set_value("previous_use", float(previous_action.use))

        return EngineStateObservation(
            features=validate_feature_vector(values),
            validity=valid,
            nearest_enemy=nearest,
            visible_enemy_count=len(enemies),
            stuck_steps=self._stuck_steps,
        )

