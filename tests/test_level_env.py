"""Contract tests for DoomLevelEnvironment.

These tests do NOT launch ViZDoom (which requires freedoom2.wad).
They instead patch the vzd.DoomGame class to verify that the wrapper
correctly assembles observations, applies reward shaping, and manages
internal state.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Action space contract tests (no engine required)
# ---------------------------------------------------------------------------

def test_action_count():
    from doom_level_env import ACTIONS, ACTION_NAMES
    assert len(ACTIONS) == 22
    assert len(ACTION_NAMES) == 22


def test_action_buttons_match_named_contract():
    from doom_level_env import ACTIONS, ACTION_NAMES
    for i, (name, action) in enumerate(zip(ACTION_NAMES, ACTIONS)):
        if name == "noop":
            assert sum(action) == 0
        else:
            assert sum(action) >= 1, f"Action {i} does not press a button."


def test_frame_stack_depth():
    from doom_level_env import FRAME_STACK
    assert FRAME_STACK == 4


# ---------------------------------------------------------------------------
# Reward shaping contract tests (patched engine)
# ---------------------------------------------------------------------------

def _make_mock_env():
    """Return a DoomLevelEnvironment with a fully mocked DoomGame."""
    with patch("doom_level_env.vzd") as mock_vzd, \
         patch("doom_level_env.LEVEL_CONFIG_PATH") as mock_cfg:
        mock_cfg.is_file.return_value = True

        mock_game = MagicMock()
        mock_vzd.DoomGame.return_value = mock_game
        mock_vzd.ScreenResolution.RES_160X120 = 0
        mock_vzd.ScreenFormat.GRAY8 = 0

        # Fake a 160x120 grayscale frame.
        mock_state = MagicMock()
        mock_state.screen_buffer = np.zeros((120, 160), dtype=np.uint8)
        mock_state.game_variables = [100.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0]
        mock_state.labels = []
        mock_game.get_state.return_value = mock_state
        mock_game.is_episode_finished.return_value = False
        mock_game.is_episode_timeout_reached.return_value = False
        mock_game.is_player_dead.return_value = False
        mock_game.get_episode_time.return_value = 0
        mock_game.get_total_reward.return_value = 0.0

        from doom_level_env import DoomLevelEnvironment, TerminalReason
        env = DoomLevelEnvironment.__new__(DoomLevelEnvironment)
        import collections
        env.game = mock_game
        env._frame_skip = 4
        env._sector_novelty = True
        env._spectator = False
        env.actions = __import__("doom_level_env").ACTIONS
        env.action_names = __import__("doom_level_env").ACTION_NAMES
        env.num_actions = len(env.actions)
        env._frame_buf = collections.deque(maxlen=4)
        env._prev_vars = np.array([100., 0., 50., 0., 0., 0., 0., 0., 0., 0.],
                                  dtype=np.float32)
        env._visited_sectors = {env._position_region(env._prev_vars)}
        env._exit_triggered = False
        env._terminal_reason = TerminalReason.IN_PROGRESS
        env._last_raw_reward = 0.0
        env._last_shaped_reward = 0.0
        env._last_reward_components = env._empty_reward_components()
        return env, mock_game, mock_state


def test_kill_reward_shapes():
    """A kill should produce a positive shaped bonus."""
    env, mock_game, mock_state = _make_mock_env()
    # Simulate kill: KILLCOUNT goes from 0 → 1
    new_vars = np.array([100., 0., 50., 0., 1., 0., 0., 0., 0., 0.], dtype=np.float32)
    bonus = env._shape_reward(0.0, new_vars, mock_state, done=False)
    # Kill bonus (REWARD_KILL=5.0) * REWARD_SCALE(1/25) = 0.2
    assert bonus > 0.0, f"Expected positive kill bonus, got {bonus}"


def test_death_reward_negative():
    env, mock_game, mock_state = _make_mock_env()
    mock_game.is_player_dead.return_value = True
    penalty = env._terminal_reward(0.0)
    assert penalty < 0.0, f"Expected negative death penalty, got {penalty}"


def test_exit_reward_positive():
    env, mock_game, mock_state = _make_mock_env()
    env._exit_triggered = True
    mock_game.is_player_dead.return_value = False
    bonus = env._terminal_reward(0.0)
    assert bonus > 0.0, f"Expected positive exit reward, got {bonus}"


def test_step_penalty_applied():
    """Each step without any other event should produce a small negative reward."""
    env, mock_game, mock_state = _make_mock_env()
    # No change in vars → only step penalty applies.
    same_vars = env._prev_vars.copy()
    bonus = env._shape_reward(0.0, same_vars, mock_state, done=False)
    assert bonus < 0.0, f"Expected negative step penalty, got {bonus}"


def test_reward_components_are_exposed():
    env, _, mock_state = _make_mock_env()
    new_vars = env._prev_vars.copy()
    new_vars[4] = 1.0

    reward = env._shape_reward(0.25, new_vars, mock_state, done=False)

    components = env._last_reward_components
    assert components["raw_game_reward"] == pytest.approx(0.25)
    assert components["kill_reward"] > 0.0
    assert components["scaled_total"] == pytest.approx(reward)


@pytest.mark.parametrize(
    ("dead", "timed_out", "expected"),
    [
        (True, False, "died"),
        (False, True, "timed_out"),
        (False, False, "completed"),
    ],
)
def test_terminal_reason_classification(dead, timed_out, expected):
    from doom_level_env import TerminalReason

    env, mock_game, _ = _make_mock_env()
    mock_game.is_episode_finished.return_value = True
    mock_game.is_player_dead.return_value = dead
    mock_game.is_episode_timeout_reached.return_value = timed_out

    reason = env._classify_terminal_reason()

    assert isinstance(reason, TerminalReason)
    assert reason.value == expected


def test_terminal_diagnostics_preserve_last_variables():
    from doom_level_env import TerminalReason

    env, mock_game, _ = _make_mock_env()
    env._prev_vars = np.array(
        [75.0, 20.0, 33.0, 4.0, 2.0, 3.0, 1.0, 12.0, 34.0, 90.0],
        dtype=np.float32,
    )
    env._terminal_reason = TerminalReason.COMPLETED
    env._exit_triggered = True
    mock_game.is_episode_finished.return_value = True
    mock_game.get_episode_time.return_value = 123
    mock_game.get_total_reward.return_value = 10.0

    info = env.diagnostics()

    assert info["terminal_reason"] == "completed"
    assert info["exit_triggered"] is True
    assert info["kill_count"] == 2.0
    assert info["item_count"] == 3.0
    assert info["position_x"] == 12.0
    assert info["position_y"] == 34.0


@pytest.mark.parametrize("action", [-1, 22, 100])
def test_invalid_action_index_is_rejected(action):
    env, _, _ = _make_mock_env()
    with pytest.raises(IndexError):
        env.step(action)


def test_observation_shape():
    """_stacked_obs must return (FRAME_STACK, 84, 84)."""
    import collections
    from doom_level_env import FRAME_STACK, DoomLevelEnvironment
    import numpy as np
    env = DoomLevelEnvironment.__new__(DoomLevelEnvironment)
    env._frame_buf = collections.deque(maxlen=FRAME_STACK)
    for _ in range(FRAME_STACK):
        env._frame_buf.append(np.zeros((84, 84), dtype=np.float32))
    obs = env._stacked_obs()
    assert obs.shape == (FRAME_STACK, 84, 84)
    assert obs.dtype == np.float32


def test_spectator_step_uses_shared_transition_pipeline():
    env, mock_game, mock_state = _make_mock_env()
    env._spectator = True
    env.reset()
    mock_game.get_last_action.return_value = [1, 0, 0, 0, 0, 0, 0, 0]
    mock_game.get_last_reward.return_value = 0.25
    mock_game.is_episode_finished.return_value = False

    observation, reward, done, buttons = env.spectator_step()

    assert mock_game.advance_action.call_count == 4
    mock_game.advance_action.assert_called_with(1)
    assert observation.shape == (4, 84, 84)
    assert reward == pytest.approx((1.0 - 0.01) / 25.0)
    assert done is False
    assert buttons == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert len(env.last_spectator_tic_actions) == 4


def test_programmatic_step_is_rejected_in_spectator_mode():
    env, _, _ = _make_mock_env()
    env._spectator = True
    with pytest.raises(RuntimeError, match="spectator_step"):
        env.step(0)


def test_step_buttons_executes_factorized_button_vector():
    env, mock_game, _ = _make_mock_env()
    env.reset()
    mock_game.make_action.return_value = 0.0
    buttons = (True, False, False, False, False, True, True, False)

    observation, _, done = env.step_buttons(buttons)

    mock_game.make_action.assert_called_once_with(list(buttons), 4)
    assert observation.shape == (4, 84, 84)
    assert done is False


def test_step_buttons_rejects_wrong_shape_and_non_binary_values():
    env, _, _ = _make_mock_env()
    with pytest.raises(ValueError, match="Expected 8"):
        env.step_buttons((True,))
    with pytest.raises(ValueError, match="binary"):
        env.step_buttons((0, 0, 0, 0, 0, 0, 2, 0))
