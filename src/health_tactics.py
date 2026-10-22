"""Tactical decision arbitrator for opportunistic health seeking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from engine_state_observation import VisibleEnemy, VisibleHealthItem
from map01_route import angle_delta_degrees


@dataclass(frozen=True)
class HealthSubgoal:
    position: tuple[float, float]
    name: str
    heal_amount: int
    distance: float
    bearing_degrees: float


# MAP01 Nukage Pit bounding box (damaging radioactive slime)
MAP01_ACID_PIT_X_MIN = 520.0
MAP01_ACID_PIT_X_MAX = 1080.0
MAP01_ACID_PIT_Y_MAX = -740.0


def _is_in_map01_acid_pit(x: float, y: float) -> bool:
    return MAP01_ACID_PIT_X_MIN <= x <= MAP01_ACID_PIT_X_MAX and y <= MAP01_ACID_PIT_Y_MAX


class HealthTacticsArbitrator:
    """Evaluates visible health items and yields safe, opportunistic detour subgoals."""

    def __init__(
        self,
        *,
        health_threshold: float = 50.0,
        max_detour_distance: float = 256.0,
        max_bearing_error: float = 75.0,
        max_pursuit_ticks: int = 35,
        blackout_ticks: int = 105,
        combat_threat_distance: float = 128.0,
        combat_threat_angle: float = 35.0,
    ):
        self.health_threshold = float(health_threshold)
        self.max_detour_distance = float(max_detour_distance)
        self.max_bearing_error = float(max_bearing_error)
        self.max_pursuit_ticks = int(max_pursuit_ticks)
        self.blackout_ticks = int(blackout_ticks)
        self.combat_threat_distance = float(combat_threat_distance)
        self.combat_threat_angle = float(combat_threat_angle)

        self._active_target_pos: Optional[tuple[float, float]] = None
        self._pursuit_ticks: int = 0
        self._blacklisted_positions: dict[tuple[int, int], int] = {}
        self._current_tick: int = 0

    def reset(self) -> None:
        self._active_target_pos = None
        self._pursuit_ticks = 0
        self._blacklisted_positions.clear()
        self._current_tick = 0

    def _pos_key(self, pos: tuple[float, float]) -> tuple[int, int]:
        return int(round(pos[0] / 16.0)), int(round(pos[1] / 16.0))

    def is_blacklisted(self, pos: tuple[float, float]) -> bool:
        key = self._pos_key(pos)
        expiry = self._blacklisted_positions.get(key)
        if expiry is None:
            return False
        if self._current_tick >= expiry:
            del self._blacklisted_positions[key]
            return False
        return True

    def blacklist(self, pos: tuple[float, float]) -> None:
        key = self._pos_key(pos)
        self._blacklisted_positions[key] = self._current_tick + self.blackout_ticks

    def step(
        self,
        *,
        player_health: float,
        visible_items: list[VisibleHealthItem],
        nearest_enemy: Optional[VisibleEnemy],
        player_angle: float,
    ) -> Optional[HealthSubgoal]:
        self._current_tick += 1

        # 1. Healthy check: if HP >= threshold, no detour needed
        if player_health >= self.health_threshold:
            self._active_target_pos = None
            self._pursuit_ticks = 0
            return None

        # 2. Combat threat check:
        # At critical health (<= 25 HP), survival requires healing; only point-blank threats (<120) block detours
        threat_dist = 120.0 if player_health <= 25.0 else self.combat_threat_distance
        if nearest_enemy is not None:
            enemy_error = abs(angle_delta_degrees(nearest_enemy.bearing_degrees, player_angle))
            if (
                nearest_enemy.distance < threat_dist
                and enemy_error <= self.combat_threat_angle
            ):
                # Combat flinching takes precedence
                return None

        # 3. Filter candidates
        valid_candidates: list[VisibleHealthItem] = []
        for item in visible_items:
            # Distance filter
            if item.distance > self.max_detour_distance:
                continue

            # Forward cone filter (never turn 180 away from route)
            angle_error = abs(angle_delta_degrees(item.bearing_degrees, player_angle))
            if angle_error > self.max_bearing_error:
                continue

            # Slime pit check
            if _is_in_map01_acid_pit(item.position[0], item.position[1]):
                continue

            # Blacklist check (unreachable / timeout items)
            if self.is_blacklisted(item.position):
                continue

            valid_candidates.append(item)

        if not valid_candidates:
            self._active_target_pos = None
            self._pursuit_ticks = 0
            return None

        # Best candidate (already sorted by priority / distance)
        best = valid_candidates[0]

        # 4. Pursuit timeout tracking
        if self._active_target_pos is not None:
            dist_to_active = math.hypot(
                best.position[0] - self._active_target_pos[0],
                best.position[1] - self._active_target_pos[1],
            )
            if dist_to_active < 24.0:
                self._pursuit_ticks += 1
                if self._pursuit_ticks > self.max_pursuit_ticks:
                    # Timed out! Blacklist and abort
                    self.blacklist(best.position)
                    self._active_target_pos = None
                    self._pursuit_ticks = 0
                    return None
            else:
                self._active_target_pos = best.position
                self._pursuit_ticks = 1
        else:
            self._active_target_pos = best.position
            self._pursuit_ticks = 1

        return HealthSubgoal(
            position=best.position,
            name=best.name,
            heal_amount=best.heal_amount,
            distance=best.distance,
            bearing_degrees=best.bearing_degrees,
        )
