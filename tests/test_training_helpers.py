import pytest

from train import aiming_reward, spawn_category


def test_spawn_category_uses_initial_target_offset():
    assert spawn_category(
        {"target_visible": True, "target_horizontal_offset": -0.3}, 0.1
    ) == "hard_left"
    assert spawn_category(
        {"target_visible": True, "target_horizontal_offset": 0.3}, 0.1
    ) == "hard_right"
    assert spawn_category(
        {"target_visible": True, "target_horizontal_offset": 0.05}, 0.1
    ) == "easy"


def test_aiming_reward_rewards_progress_and_penalizes_bad_shots():
    before = {"target_visible": True, "target_horizontal_offset": -0.5}
    after = {"target_visible": True, "target_horizontal_offset": -0.2}

    reward, bad_shot = aiming_reward(
        before,
        after,
        "turn_left",
        easy_target_offset=0.1,
        progress_weight=20.0,
        off_target_attack_penalty=4.0,
    )
    assert reward == pytest.approx(6.0)
    assert not bad_shot

    reward, bad_shot = aiming_reward(
        before,
        after,
        "attack",
        easy_target_offset=0.1,
        progress_weight=20.0,
        off_target_attack_penalty=4.0,
    )
    assert reward == pytest.approx(2.0)
    assert bad_shot
