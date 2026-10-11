from __future__ import annotations

import pytest

from agent_config import BUTTON_NAMES
from doom_level_env import DoomLevelEnvironment
from project_paths import LEVEL_CONFIG_PATH, PROJECT_ROOT


pytestmark = pytest.mark.skipif(
    not (PROJECT_ROOT / "freedoom2.wad").is_file(),
    reason="FreeDoom WAD is not available for real-engine tests.",
)


def test_real_engine_timeout_preserves_terminal_reason():
    """Exercise a real terminal transition without waiting for MAP01's timeout."""
    with DoomLevelEnvironment(
        config_file=LEVEL_CONFIG_PATH,
        render=False,
        seed=123,
        episode_timeout=1,
    ) as env:
        observation = env.reset()
        next_observation, _, done = env.step(0)
        info = env.diagnostics()

    assert observation.shape == (4, 84, 84)
    assert next_observation.shape == (4, 84, 84)
    assert done is True
    assert info["terminal_reason"] == "timed_out"
    assert info["exit_triggered"] is False
    assert info["episode_time"] >= 1


def test_real_engine_death_preserves_terminal_reason():
    with DoomLevelEnvironment(
        config_file=LEVEL_CONFIG_PATH,
        render=False,
        seed=456,
    ) as env:
        env.reset()
        env.game.send_game_command("kill")
        _, reward, done = env.step(0)
        info = env.diagnostics()

    assert done is True
    assert reward < 0.0
    assert info["terminal_reason"] == "died"
    assert info["player_dead"] is True
    assert info["exit_triggered"] is False


def test_real_engine_headless_spectator_timeout_uses_shared_pipeline():
    with DoomLevelEnvironment(
        config_file=LEVEL_CONFIG_PATH,
        render=False,
        seed=789,
        episode_timeout=1,
        spectator=True,
    ) as env:
        env.reset()
        observation, reward, done, buttons = env.spectator_step()
        info = env.diagnostics()

    assert observation.shape == (4, 84, 84)
    assert isinstance(reward, float)
    assert done is True
    assert len(buttons) == len(BUTTON_NAMES)
    assert info["terminal_reason"] == "timed_out"
