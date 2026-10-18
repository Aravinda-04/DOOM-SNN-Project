"""Versioned contracts for the privileged MAP01 engine-state agent."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import numpy as np

from agent_config import BUTTON_NAMES


ENGINE_STATE_CONTRACT_VERSION = 1


class Locomotion(IntEnum):
    NONE = 0
    FORWARD = 1
    BACKWARD = 2
    STRAFE_LEFT = 3
    STRAFE_RIGHT = 4


class Turning(IntEnum):
    NONE = 0
    LEFT = 1
    RIGHT = 2


LOCOMOTION_NAMES = tuple(value.name.lower() for value in Locomotion)
TURNING_NAMES = tuple(value.name.lower() for value in Turning)


@dataclass(frozen=True)
class FactorizedAction:
    """One safe command from the four independent policy heads."""

    locomotion: Locomotion = Locomotion.NONE
    turning: Turning = Turning.NONE
    attack: bool = False
    use: bool = False


def compose_buttons(action: FactorizedAction) -> tuple[bool, ...]:
    """Convert a factorized command to the configured ViZDoom button order."""
    pressed: set[str] = set()
    locomotion_button = {
        Locomotion.NONE: None,
        Locomotion.FORWARD: "MOVE_FORWARD",
        Locomotion.BACKWARD: "MOVE_BACKWARD",
        Locomotion.STRAFE_LEFT: "MOVE_LEFT",
        Locomotion.STRAFE_RIGHT: "MOVE_RIGHT",
    }[Locomotion(action.locomotion)]
    turning_button = {
        Turning.NONE: None,
        Turning.LEFT: "TURN_LEFT",
        Turning.RIGHT: "TURN_RIGHT",
    }[Turning(action.turning)]
    if locomotion_button:
        pressed.add(locomotion_button)
    if turning_button:
        pressed.add(turning_button)
    if action.attack:
        pressed.add("ATTACK")
    if action.use:
        pressed.add("USE")
    return tuple(name in pressed for name in BUTTON_NAMES)


def action_targets(action: FactorizedAction) -> np.ndarray:
    """Encode a factorized command as four integer supervision targets."""
    return np.asarray(
        [
            int(action.locomotion),
            int(action.turning),
            int(bool(action.attack)),
            int(bool(action.use)),
        ],
        dtype=np.int64,
    )


FEATURE_NAMES = (
    "position_x",
    "position_y",
    "heading_sin",
    "heading_cos",
    "delta_x",
    "delta_y",
    "delta_angle_sin",
    "delta_angle_cos",
    "health",
    "armor",
    "ammo_bullets",
    "ammo_shells",
    "kill_count",
    "item_count",
    "time_fraction",
    "stuck_fraction",
    "enemy_visible",
    "enemy_count",
    "enemy_distance",
    "enemy_bearing_sin",
    "enemy_bearing_cos",
    "enemy_screen_x",
    "enemy_screen_area",
    "enemy_in_firing_cone",
    "route_dx",
    "route_dy",
    "route_distance",
    "route_bearing_sin",
    "route_bearing_cos",
    "route_heading_error_sin",
    "route_heading_error_cos",
    "route_progress",
    "route_off_track",
    "use_recommended",
    "previous_forward",
    "previous_backward",
    "previous_strafe_left",
    "previous_strafe_right",
    "previous_turn_left",
    "previous_turn_right",
    "previous_attack",
    "previous_use",
)

FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


def _contract_payload() -> dict:
    return {
        "version": ENGINE_STATE_CONTRACT_VERSION,
        "features": FEATURE_NAMES,
        "locomotion": LOCOMOTION_NAMES,
        "turning": TURNING_NAMES,
        "buttons": BUTTON_NAMES,
    }


ENGINE_STATE_CONTRACT_HASH = hashlib.sha256(
    json.dumps(_contract_payload(), sort_keys=True).encode("utf-8")
).hexdigest()


def validate_feature_vector(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.float32)
    if array.shape != (len(FEATURE_NAMES),):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} engine-state features, "
            f"got shape {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Engine-state feature vector contains non-finite values.")
    return array
