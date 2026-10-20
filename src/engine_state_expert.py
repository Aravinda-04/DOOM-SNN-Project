"""Scripted MAP01 expert used to generate privileged-state supervision."""

from __future__ import annotations

from dataclasses import dataclass

from engine_state_config import FactorizedAction, Locomotion, Turning
from engine_state_observation import EngineStateObservation
from map01_route import RouteContext, angle_delta_degrees


@dataclass(frozen=True)
class ExpertConfig:
    hard_turn_degrees: float = 12.0
    moving_turn_degrees: float = 4.0
    # One four-tic turn can cross more than five degrees. Matching the route
    # controller's tolerance prevents left/right oscillation around a target.
    firing_cone_degrees: float = 12.0
    minimum_enemy_screen_area: float = 0.0015
    fire_cooldown_steps: int = 0
    use_cooldown_steps: int = 4


class EngineStateExpert:
    """Priority controller for combat, interaction, and route following."""

    def __init__(self, config: ExpertConfig = ExpertConfig()):
        self.config = config
        self._fire_cooldown = 0
        self._use_cooldown = 0

    def reset(self) -> None:
        self._fire_cooldown = 0
        self._use_cooldown = 0

    def act(
        self,
        observation: EngineStateObservation,
        route: RouteContext,
        *,
        player_angle: float,
        ammunition: float,
    ) -> FactorizedAction:
        self._fire_cooldown = max(0, self._fire_cooldown - 1)
        self._use_cooldown = max(0, self._use_cooldown - 1)

        if route.use_recommended and self._use_cooldown == 0:
            self._use_cooldown = self.config.use_cooldown_steps
            return FactorizedAction(locomotion=Locomotion.FORWARD, use=True)

        # Recovery has priority over optional combat. Otherwise an enemy that
        # repeatedly enters/leaves the minimum screen area can make the expert
        # alternate between route turning and aim turning without moving.
        if route.recovery or observation.stuck_steps >= 8:
            return self._navigate(route, player_angle)

        enemy = observation.nearest_enemy
        should_engage = (
            enemy is not None
            and ammunition > 0
            and enemy.screen_area >= self.config.minimum_enemy_screen_area
        )
        if should_engage and enemy is not None:
            error = angle_delta_degrees(enemy.bearing_degrees, player_angle)
            if error > self.config.firing_cone_degrees:
                fire = self._fire_cooldown == 0 and error <= 25.0
                if fire:
                    self._fire_cooldown = self.config.fire_cooldown_steps
                return FactorizedAction(turning=Turning.LEFT, attack=fire)
            if error < -self.config.firing_cone_degrees:
                fire = self._fire_cooldown == 0 and error >= -25.0
                if fire:
                    self._fire_cooldown = self.config.fire_cooldown_steps
                return FactorizedAction(turning=Turning.RIGHT, attack=fire)
            if self._fire_cooldown == 0:
                self._fire_cooldown = self.config.fire_cooldown_steps
                return FactorizedAction(
                    locomotion=(
                        Locomotion.FORWARD
                        if enemy.distance > 96.0
                        else Locomotion.NONE
                    ),
                    attack=True,
                )
            return FactorizedAction(
                locomotion=(
                    Locomotion.FORWARD
                    if enemy.distance > 128.0
                    else Locomotion.STRAFE_LEFT
                ),
                attack=True,
            )

        return self._navigate(route, player_angle)

    def _navigate(
        self, route: RouteContext, player_angle: float
    ) -> FactorizedAction:
        error = angle_delta_degrees(route.bearing_degrees, player_angle)
        if error > self.config.hard_turn_degrees:
            return FactorizedAction(turning=Turning.LEFT)
        if error < -self.config.hard_turn_degrees:
            return FactorizedAction(turning=Turning.RIGHT)
        if error > self.config.moving_turn_degrees:
            return FactorizedAction(
                locomotion=Locomotion.FORWARD,
                turning=Turning.LEFT,
            )
        if error < -self.config.moving_turn_degrees:
            return FactorizedAction(
                locomotion=Locomotion.FORWARD,
                turning=Turning.RIGHT,
            )
        return FactorizedAction(locomotion=Locomotion.FORWARD)
