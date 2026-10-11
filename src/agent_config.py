"""Immutable visual-agent and action-space contract for MAP01.

Every environment, dataset, trainer, checkpoint, and player must use this
module rather than maintaining its own action ordering. Changing this contract
invalidates demonstrations and model output heads, so the serialized contract
is hashed and should be stored with future artifacts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Optional


MODEL_NAME = "crsnn"
MAP_NAME = "map01"
CONTRACT_VERSION = 1
PREPROCESSING_VERSION = 1

FRAME_STACK = 4
FRAME_HEIGHT = 84
FRAME_WIDTH = 84
DEFAULT_FRAME_SKIP = 4

# This order must match ``available_buttons`` in scenarios/level01.cfg.
BUTTON_NAMES = (
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "ATTACK",
    "USE",
)

ACTION_NAMES = (
    "forward",
    "backward",
    "strafe_left",
    "strafe_right",
    "turn_left",
    "turn_right",
    "attack",
    "use",
    "noop",
)

# Atomic actions are intentionally frozen for Phase 0 validation. If a manual
# MAP01 completion proves combinations are essential, replace this catalogue
# once before collecting permanent demonstrations or training checkpoints.
ACTIONS = tuple(
    tuple(button_index == action_index for button_index in range(len(BUTTON_NAMES)))
    for action_index in range(len(BUTTON_NAMES))
) + (tuple(False for _ in BUTTON_NAMES),)


@dataclass(frozen=True)
class AgentContract:
    """Serializable identity of observations and model outputs."""

    contract_version: int
    model: str
    map_name: str
    preprocessing_version: int
    frame_stack: int
    frame_height: int
    frame_width: int
    default_frame_skip: int
    button_names: tuple[str, ...]
    action_names: tuple[str, ...]
    actions: tuple[tuple[bool, ...], ...]

    @property
    def action_size(self) -> int:
        return len(self.actions)

    def to_dict(self) -> dict:
        """Return a JSON-compatible representation."""
        return asdict(self)

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


AGENT_CONTRACT = AgentContract(
    contract_version=CONTRACT_VERSION,
    model=MODEL_NAME,
    map_name=MAP_NAME,
    preprocessing_version=PREPROCESSING_VERSION,
    frame_stack=FRAME_STACK,
    frame_height=FRAME_HEIGHT,
    frame_width=FRAME_WIDTH,
    default_frame_skip=DEFAULT_FRAME_SKIP,
    button_names=BUTTON_NAMES,
    action_names=ACTION_NAMES,
    actions=ACTIONS,
)
ACTION_CATALOGUE_HASH = AGENT_CONTRACT.sha256


def validate_agent_contract(contract: AgentContract = AGENT_CONTRACT) -> None:
    """Reject ambiguous or internally inconsistent action definitions."""
    if contract.frame_stack <= 0:
        raise ValueError("frame_stack must be positive.")
    if contract.frame_height <= 0 or contract.frame_width <= 0:
        raise ValueError("frame dimensions must be positive.")
    if contract.default_frame_skip <= 0:
        raise ValueError("default_frame_skip must be positive.")
    if not contract.actions:
        raise ValueError("At least one action is required.")
    if len(contract.action_names) != len(contract.actions):
        raise ValueError("Every action must have exactly one name.")
    if len(set(contract.action_names)) != len(contract.action_names):
        raise ValueError("Action names must be unique.")
    if len(set(contract.actions)) != len(contract.actions):
        raise ValueError("Action button vectors must be unique.")
    for name, action in zip(contract.action_names, contract.actions):
        if len(action) != len(contract.button_names):
            raise ValueError("Every action must define every configured button.")
        expected_pressed = 0 if name == "noop" else 1
        if sum(action) != expected_pressed:
            raise ValueError(
                f"Action '{name}' must press exactly {expected_pressed} buttons."
            )


def match_button_vector(
    values: Iterable[float | bool],
) -> tuple[str, Optional[int]]:
    """Classify a spectator button vector against the frozen catalogue.

    Returns ``("supported", index)`` for an exact catalogue action,
    ``("idle", None)`` when no configured button is active, and
    ``("unsupported", None)`` for simultaneous or otherwise unknown inputs.
    """
    vector = tuple(bool(value) for value in values)
    if len(vector) != len(BUTTON_NAMES):
        raise ValueError(
            f"Expected {len(BUTTON_NAMES)} button values, got {len(vector)}."
        )
    try:
        return "supported", ACTIONS.index(vector)
    except ValueError:
        if not any(vector):
            return "idle", None
        return "unsupported", None


validate_agent_contract()
