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
from collections import Counter
from typing import Iterable, Optional, Sequence


MODEL_NAME = "crsnn"
MAP_NAME = "map01"
CONTRACT_VERSION = 2
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

def _action(*pressed: str) -> tuple[bool, ...]:
    unknown = sorted(set(pressed) - set(BUTTON_NAMES))
    if unknown:
        raise ValueError(f"Unknown action buttons: {unknown}")
    return tuple(name in pressed for name in BUTTON_NAMES)


# Keep the original atomic action indices stable, append no-op, then append the
# combinations people naturally use while navigating and fighting. Less common
# combinations remain recordable as ambiguous/masked decisions.
ACTION_DEFINITIONS = (
    ("forward", _action("MOVE_FORWARD")),
    ("backward", _action("MOVE_BACKWARD")),
    ("strafe_left", _action("MOVE_LEFT")),
    ("strafe_right", _action("MOVE_RIGHT")),
    ("turn_left", _action("TURN_LEFT")),
    ("turn_right", _action("TURN_RIGHT")),
    ("attack", _action("ATTACK")),
    ("use", _action("USE")),
    ("noop", _action()),
    ("forward_turn_left", _action("MOVE_FORWARD", "TURN_LEFT")),
    ("forward_turn_right", _action("MOVE_FORWARD", "TURN_RIGHT")),
    ("forward_strafe_left", _action("MOVE_FORWARD", "MOVE_LEFT")),
    ("forward_strafe_right", _action("MOVE_FORWARD", "MOVE_RIGHT")),
    ("forward_attack", _action("MOVE_FORWARD", "ATTACK")),
    ("backward_attack", _action("MOVE_BACKWARD", "ATTACK")),
    ("strafe_left_attack", _action("MOVE_LEFT", "ATTACK")),
    ("strafe_right_attack", _action("MOVE_RIGHT", "ATTACK")),
    ("turn_left_attack", _action("TURN_LEFT", "ATTACK")),
    ("turn_right_attack", _action("TURN_RIGHT", "ATTACK")),
    (
        "forward_turn_left_attack",
        _action("MOVE_FORWARD", "TURN_LEFT", "ATTACK"),
    ),
    (
        "forward_turn_right_attack",
        _action("MOVE_FORWARD", "TURN_RIGHT", "ATTACK"),
    ),
    ("forward_use", _action("MOVE_FORWARD", "USE")),
)
ACTION_NAMES = tuple(name for name, _ in ACTION_DEFINITIONS)
ACTIONS = tuple(action for _, action in ACTION_DEFINITIONS)


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
        pressed = sum(action)
        if name == "noop" and pressed != 0:
            raise ValueError(
                "The noop action cannot press any buttons."
            )
        if name != "noop" and pressed == 0:
            raise ValueError(f"Action '{name}' must press at least one button.")


def match_button_vector(
    values: Iterable[float | bool],
) -> tuple[str, Optional[int]]:
    """Classify a spectator button vector against the frozen catalogue.

    Returns ``("supported", index)`` for an exact catalogue action (including
    no-op) and ``("unsupported", None)`` for any other combination.
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


def resolve_tic_actions(
    tic_actions: Sequence[Iterable[float | bool]],
    *,
    minimum_agreement: float = 0.75,
) -> tuple[Optional[int], bool, float]:
    """Resolve raw per-tic input to one catalogue label plus a validity mask.

    A decision is valid when one supported catalogue action occupies at least
    ``minimum_agreement`` of its engine tics. This tolerates a one-tic keyboard
    transition in the normal four-tic decision window while masking 2/2 ties,
    rapidly changing input, and unsupported combinations.
    """
    if not 0.5 < minimum_agreement <= 1.0:
        raise ValueError("minimum_agreement must be in (0.5, 1.0].")
    if not tic_actions:
        raise ValueError("At least one tic action is required.")
    matched: list[int] = []
    for values in tic_actions:
        status, index = match_button_vector(values)
        if status == "supported" and index is not None:
            matched.append(index)
    if not matched:
        return None, False, 0.0
    counts = Counter(matched)
    action_index, count = counts.most_common(1)[0]
    agreement = count / len(tic_actions)
    return action_index, agreement >= minimum_agreement, agreement


validate_agent_contract()
