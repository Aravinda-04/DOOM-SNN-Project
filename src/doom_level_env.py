"""ViZDoom environment wrapper for a full DOOM level (FreeDoom2 MAP01).

Differences from env.py (basic scenario):
- 4-frame grayscale stack as observation (shape 4 x 84 x 84).
- 9-action space: eight atomic controls plus no-op.
- Dense reward shaping: kills, health/ammo/key pickups, door novelty, exit.
- Episode ends on death, exit trigger, or configurable timeout.
- Game-variable diagnostics extended for navigation metrics.
"""

from __future__ import annotations

import collections
import math
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import vizdoom as vzd

from agent_config import (
    ACTION_NAMES,
    ACTIONS,
    AGENT_CONTRACT,
    BUTTON_NAMES,
    DEFAULT_FRAME_SKIP,
    FRAME_HEIGHT,
    FRAME_STACK,
    FRAME_WIDTH,
    MAP_NAME,
)
from project_paths import LEVEL_CONFIG_PATH


class TerminalReason(str, Enum):
    """Why an episode stopped."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DIED = "died"
    TIMED_OUT = "timed_out"

# Index positions for game variables returned by ViZDoom (must match cfg order).
_VAR_HEALTH = 0
_VAR_ARMOR = 1
_VAR_AMMO2 = 2   # bullets
_VAR_AMMO3 = 3   # shells
_VAR_KILLS = 4
_VAR_ITEMS = 5
_VAR_SECRETS = 6
_VAR_POS_X = 7
_VAR_POS_Y = 8
_VAR_ANGLE = 9

# Reward shaping weights.
REWARD_KILL = 5.0
REWARD_HEALTH = 0.5
REWARD_AMMO = 0.3
REWARD_ITEM = 1.0
REWARD_DOOR_NOVELTY = 2.0
REWARD_STEP = -0.01
REWARD_DEATH = -25.0
REWARD_EXIT = 100.0
REWARD_SCALE = 1.0 / 25.0   # scale all shaped rewards down for stable TD targets

EXPECTED_BUTTONS = tuple(getattr(vzd.Button, name) for name in BUTTON_NAMES)
EXPECTED_GAME_VARIABLES = (
    vzd.GameVariable.HEALTH,
    vzd.GameVariable.ARMOR,
    vzd.GameVariable.AMMO2,
    vzd.GameVariable.AMMO3,
    vzd.GameVariable.KILLCOUNT,
    vzd.GameVariable.ITEMCOUNT,
    vzd.GameVariable.SECRETCOUNT,
    vzd.GameVariable.POSITION_X,
    vzd.GameVariable.POSITION_Y,
    vzd.GameVariable.ANGLE,
)


class DoomLevelEnvironment:
    """ViZDoom wrapper for a full DOOM level with frame-stacking and shaped rewards.

    Parameters
    ----------
    config_file:
        Path to the ViZDoom .cfg for the full level. Defaults to
        ``scenarios/level01.cfg`` relative to the project root.
    render:
        Whether to open the ViZDoom window.
    seed:
        Optional engine random seed.
    frame_skip:
        Number of game tics to repeat each action (4 matches the basic env).
    sector_novelty:
        If True, award ``REWARD_DOOR_NOVELTY`` when the player first enters a
        new coarse map region. The name is retained for API compatibility;
        ViZDoom object labels do not expose sector indices.
    """

    def __init__(
        self,
        config_file: Optional[Path] = None,
        render: bool = False,
        seed: Optional[int] = None,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        sector_novelty: bool = True,
        episode_timeout: Optional[int] = None,
        spectator: bool = False,
    ):
        if frame_skip <= 0:
            raise ValueError("frame_skip must be positive.")
        self._frame_skip = frame_skip
        self._sector_novelty = sector_novelty
        self._spectator = spectator

        self.game = vzd.DoomGame()
        config_path = Path(config_file or LEVEL_CONFIG_PATH).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(
                f"ViZDoom level config not found: {config_path}\n"
                "Place freedoom2.wad in the project root and ensure "
                "scenarios/level01.cfg is present."
            )

        try:
            self.game.load_config(str(config_path))
            if episode_timeout is not None:
                if episode_timeout <= 0:
                    raise ValueError("episode_timeout must be positive.")
                self.game.set_episode_timeout(episode_timeout)
            if seed is not None:
                self.game.set_seed(seed)

            self.game.set_doom_map(MAP_NAME)
            if spectator:
                self.game.set_mode(vzd.Mode.SPECTATOR)
            
            self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
            self.game.set_screen_format(vzd.ScreenFormat.GRAY8)
            self.game.set_window_visible(render)
            self.game.set_labels_buffer_enabled(True)
            self.game.set_automap_buffer_enabled(False)

            self.game.init()
            self._validate_engine_contract(config_path)
        except Exception as exc:
            self.game.close()
            raise RuntimeError(
                f"Could not initialise ViZDoom with config '{config_path}': {exc}"
            ) from exc

        self.actions = ACTIONS
        self.action_names = ACTION_NAMES
        self.num_actions = AGENT_CONTRACT.action_size

        # State maintained across steps within an episode.
        self._frame_buf: collections.deque[np.ndarray] = collections.deque(
            maxlen=FRAME_STACK
        )
        self._prev_vars: Optional[np.ndarray] = None
        self._visited_sectors: set[tuple[int, int]] = set()
        self._exit_triggered: bool = False
        self._terminal_reason = TerminalReason.IN_PROGRESS
        self._last_raw_reward = 0.0
        self._last_shaped_reward = 0.0
        self._last_reward_components = self._empty_reward_components()
        self._last_spectator_tic_actions: tuple[tuple[float, ...], ...] = ()

    def _validate_engine_contract(self, config_path: Path) -> None:
        """Reject scenario settings that would silently remap model outputs."""
        actual_buttons = tuple(self.game.get_available_buttons())
        if actual_buttons != EXPECTED_BUTTONS:
            raise ValueError(
                f"Unexpected button order in {config_path}: "
                f"expected {EXPECTED_BUTTONS}, got {actual_buttons}."
            )
        actual_variables = tuple(self.game.get_available_game_variables())
        if actual_variables != EXPECTED_GAME_VARIABLES:
            raise ValueError(
                f"Unexpected game-variable order in {config_path}: "
                f"expected {EXPECTED_GAME_VARIABLES}, got {actual_variables}."
            )
        if self.game.get_screen_format() != vzd.ScreenFormat.GRAY8:
            raise ValueError(
                f"Unexpected screen format in {config_path}; GRAY8 is required."
            )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Start a new episode and return the initial stacked observation."""
        self.game.new_episode()
        self._visited_sectors = set()
        self._exit_triggered = False
        self._terminal_reason = TerminalReason.IN_PROGRESS
        self._last_raw_reward = 0.0
        self._last_shaped_reward = 0.0
        self._last_reward_components = self._empty_reward_components()
        self._last_spectator_tic_actions = ()

        raw = self.game.get_state().screen_buffer  # type: ignore[union-attr]
        frame = self._preprocess(raw)

        # Fill the stack with copies of the first frame.
        self._frame_buf.clear()
        for _ in range(FRAME_STACK):
            self._frame_buf.append(frame)

        self._prev_vars = self._read_vars()
        region = self._position_region(self._prev_vars)
        if region is not None:
            self._visited_sectors.add(region)
        return self._stacked_obs()

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool]:
        """Execute one action and return (next_obs, shaped_reward, done).

        Parameters
        ----------
        action_idx:
            Index into the frozen action catalogue.
        """
        if getattr(self, "_spectator", False):
            raise RuntimeError(
                "step() is unavailable in spectator mode; use spectator_step()."
            )
        if not isinstance(action_idx, (int, np.integer)):
            raise TypeError(f"action_idx must be an integer, got {type(action_idx).__name__}.")
        if not 0 <= int(action_idx) < self.num_actions:
            raise IndexError(
                f"action_idx {action_idx} is outside the valid range "
                f"0..{self.num_actions - 1}."
            )

        action = list(self.actions[int(action_idx)])
        raw_reward = float(self.game.make_action(action, self._frame_skip))
        done = self.game.is_episode_finished()

        observation, shaped = self._finish_transition(raw_reward, done)
        return observation, shaped, done

    def spectator_step(
        self,
    ) -> tuple[np.ndarray, float, bool, tuple[float, ...]]:
        """Advance human-controlled spectator play through the normal pipeline.

        The returned button vector uses the exact configured button order and
        can be checked against :mod:`agent_config`. This method intentionally
        shares observation, reward, and terminal processing with ``step``.
        """
        if not getattr(self, "_spectator", False):
            raise RuntimeError(
                "spectator_step() requires DoomLevelEnvironment(spectator=True)."
            )
        tic_actions: list[tuple[float, ...]] = []
        raw_reward = 0.0
        for _ in range(self._frame_skip):
            if self.game.is_episode_finished():
                break
            self.game.advance_action(1)
            tic_actions.append(
                tuple(float(value) for value in self.game.get_last_action())
            )
            raw_reward += float(self.game.get_last_reward())
        if not tic_actions:
            raise RuntimeError("Spectator episode ended before an action was observed.")
        self._last_spectator_tic_actions = tuple(tic_actions)
        action = tic_actions[-1]
        done = self.game.is_episode_finished()
        observation, shaped = self._finish_transition(raw_reward, done)
        return observation, shaped, done, action

    @property
    def last_spectator_tic_actions(self) -> tuple[tuple[float, ...], ...]:
        """Raw human button vectors observed inside the latest model step."""
        return self._last_spectator_tic_actions

    def _finish_transition(
        self, raw_reward: float, done: bool
    ) -> tuple[np.ndarray, float]:
        """Apply common post-action observation, reward, and terminal logic."""

        if not done:
            state = self.game.get_state()
            frame = self._preprocess(state.screen_buffer)
            self._frame_buf.append(frame)
            new_vars = self._read_vars()
            shaped = self._shape_reward(raw_reward, new_vars, state, done)
            self._prev_vars = new_vars
        else:
            # State buffers are unavailable after a terminal action. Preserve
            # the last readable game variables and use explicit engine flags
            # to distinguish completion, death, and timeout.
            self._frame_buf.append(
                np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)
            )
            self._terminal_reason = self._classify_terminal_reason()
            self._exit_triggered = self._terminal_reason is TerminalReason.COMPLETED
            shaped = self._terminal_reward(raw_reward)

        self._last_raw_reward = raw_reward
        self._last_shaped_reward = shaped
        return self._stacked_obs(), shaped

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize 160×120 grayscale → 84×84 float32 in [0, 1]."""
        resized = cv2.resize(
            frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA
        )
        return resized.astype(np.float32) / 255.0

    def _stacked_obs(self) -> np.ndarray:
        """Return (FRAME_STACK, 84, 84) observation array."""
        return np.stack(self._frame_buf, axis=0)

    # ------------------------------------------------------------------
    # Variable helpers
    # ------------------------------------------------------------------

    def _read_vars(self) -> Optional[np.ndarray]:
        """Read game variables as a numpy array; returns None if episode ended."""
        if self.game.is_episode_finished():
            return self._prev_vars
        state = self.game.get_state()
        if state is None:
            return self._prev_vars
        return np.array(state.game_variables, dtype=np.float32)

    def _get_var(self, vars_arr: Optional[np.ndarray], idx: int) -> float:
        if vars_arr is None or idx >= len(vars_arr):
            return 0.0
        return float(vars_arr[idx])

    def _position_region(
        self, vars_arr: Optional[np.ndarray], cell_size: float = 128.0
    ) -> Optional[tuple[int, int]]:
        """Map the current position to a stable coarse exploration cell."""
        if vars_arr is None or len(vars_arr) <= _VAR_POS_Y:
            return None
        x = float(vars_arr[_VAR_POS_X])
        y = float(vars_arr[_VAR_POS_Y])
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        return (math.floor(x / cell_size), math.floor(y / cell_size))

    @staticmethod
    def _empty_reward_components() -> dict[str, float]:
        return {
            "raw_game_reward": 0.0,
            "kill_reward": 0.0,
            "health_reward": 0.0,
            "ammo_reward": 0.0,
            "item_reward": 0.0,
            "exploration_reward": 0.0,
            "step_penalty": 0.0,
            "death_penalty": 0.0,
            "completion_reward": 0.0,
            "unscaled_total": 0.0,
            "scaled_total": 0.0,
        }

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------

    def _shape_reward(
        self,
        raw: float,
        new_vars: Optional[np.ndarray],
        state,
        done: bool,
    ) -> float:
        components = self._empty_reward_components()
        components["raw_game_reward"] = float(raw)

        prev = self._prev_vars
        # Kill reward.
        dk = self._get_var(new_vars, _VAR_KILLS) - self._get_var(prev, _VAR_KILLS)
        if dk > 0:
            components["kill_reward"] = dk * REWARD_KILL

        # Health pickup (only when health increases).
        dh = self._get_var(new_vars, _VAR_HEALTH) - self._get_var(prev, _VAR_HEALTH)
        if dh > 0:
            components["health_reward"] = min(dh, 100.0) * REWARD_HEALTH / 100.0

        # Ammo pickup (bullets + shells).
        da2 = self._get_var(new_vars, _VAR_AMMO2) - self._get_var(prev, _VAR_AMMO2)
        da3 = self._get_var(new_vars, _VAR_AMMO3) - self._get_var(prev, _VAR_AMMO3)
        if da2 > 0:
            components["ammo_reward"] += REWARD_AMMO
        if da3 > 0:
            components["ammo_reward"] += REWARD_AMMO

        # ITEMCOUNT does not identify which item was collected, so this is a
        # deliberately generic item signal rather than a claimed key reward.
        di = self._get_var(new_vars, _VAR_ITEMS) - self._get_var(prev, _VAR_ITEMS)
        if di > 0:
            components["item_reward"] = REWARD_ITEM * di

        # ViZDoom labels contain visible-object metadata, not map-sector IDs.
        # Use coarse position cells for a deterministic exploration signal.
        if self._sector_novelty:
            region = self._position_region(new_vars)
            if region is not None and region not in self._visited_sectors:
                self._visited_sectors.add(region)
                components["exploration_reward"] = REWARD_DOOR_NOVELTY

        # Per-step living penalty to encourage fast completion.
        components["step_penalty"] = REWARD_STEP

        unscaled = sum(
            value
            for key, value in components.items()
            if key not in {"unscaled_total", "scaled_total"}
        )
        components["unscaled_total"] = unscaled
        components["scaled_total"] = unscaled * REWARD_SCALE
        self._last_reward_components = components
        return components["scaled_total"]

    def _terminal_reward(self, raw: float) -> float:
        """Apply death/exit penalties at episode end."""
        components = self._empty_reward_components()
        components["raw_game_reward"] = float(raw)
        if self.game.is_player_dead():
            components["death_penalty"] = REWARD_DEATH
        if self._exit_triggered:
            components["completion_reward"] = REWARD_EXIT
        unscaled = sum(
            value
            for key, value in components.items()
            if key not in {"unscaled_total", "scaled_total"}
        )
        components["unscaled_total"] = unscaled
        components["scaled_total"] = unscaled * REWARD_SCALE
        self._last_reward_components = components
        return components["scaled_total"]

    def _classify_terminal_reason(self) -> TerminalReason:
        """Classify an engine-finished episode using explicit ViZDoom APIs."""
        if not self.game.is_episode_finished():
            return TerminalReason.IN_PROGRESS
        if self.game.is_player_dead():
            return TerminalReason.DIED
        if self.game.is_episode_timeout_reached():
            return TerminalReason.TIMED_OUT
        return TerminalReason.COMPLETED

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict:
        """Return a snapshot of game state for logging/evaluation."""
        finished = bool(self.game.is_episode_finished())
        if finished:
            v = self._prev_vars if self._prev_vars is not None else np.zeros(10)
        else:
            state = self.game.get_state()
            v = (
                np.array(state.game_variables, dtype=np.float32)
                if state else np.zeros(10)
            )

        def g(i: int) -> float:
            return float(v[i]) if i < len(v) else 0.0

        return {
            "finished": finished,
            "terminal_reason": self._terminal_reason.value,
            "player_dead": bool(self.game.is_player_dead()),
            "exit_triggered": self._exit_triggered,
            "sectors_visited": len(self._visited_sectors),
            "episode_time": int(self.game.get_episode_time()),
            "total_reward": float(self.game.get_total_reward()),
            "last_raw_reward": self._last_raw_reward,
            "last_shaped_reward": self._last_shaped_reward,
            "last_reward_components": dict(self._last_reward_components),
            "health": g(_VAR_HEALTH),
            "armor": g(_VAR_ARMOR),
            "ammo2": g(_VAR_AMMO2),
            "ammo3": g(_VAR_AMMO3),
            "kill_count": g(_VAR_KILLS),
            "item_count": g(_VAR_ITEMS),
            "secret_count": g(_VAR_SECRETS),
            "position_x": g(_VAR_POS_X),
            "position_y": g(_VAR_POS_Y),
            "angle": g(_VAR_ANGLE),
        }

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shut down the ViZDoom engine cleanly."""
        self.game.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
