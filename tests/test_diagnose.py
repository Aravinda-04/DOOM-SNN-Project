from diagnose import (
    aggregate_results,
    classify_spawn,
    classify_termination,
    spawn_matches,
)


def test_classify_termination_distinguishes_outcomes():
    start = {"kill_count": 0}

    assert classify_termination(start, {"kill_count": 1, "player_dead": False}) == (
        True,
        "kill",
    )
    assert classify_termination(start, {"kill_count": 0, "player_dead": True}) == (
        False,
        "death",
    )
    assert classify_termination(start, {"kill_count": 0, "player_dead": False}) == (
        False,
        "timeout",
    )


def test_spawn_classification_and_filters():
    assert classify_spawn(
        {"target_visible": True, "target_horizontal_offset": 0.05}, 0.1
    ) == "easy"
    assert classify_spawn(
        {"target_visible": True, "target_horizontal_offset": -0.4}, 0.1
    ) == "hard_left"
    assert classify_spawn(
        {"target_visible": True, "target_horizontal_offset": 0.4}, 0.1
    ) == "hard_right"
    assert classify_spawn(
        {"target_visible": False, "target_horizontal_offset": None}, 0.1
    ) == "invisible"
    assert spawn_matches("hard", "hard_left")
    assert spawn_matches("hard", "invisible")
    assert not spawn_matches("hard", "easy")
    assert spawn_matches("left", "hard_left")


def test_aggregate_results_weights_spike_rates_by_opportunities():
    base = {
        "decision_steps": 2,
        "termination": "kill",
        "wall_loop_suspected": False,
        "instant_kill": False,
        "noninstant_success": True,
        "hard_spawn": True,
        "hard_spawn_success": True,
        "spawn_category": "hard_left",
        "action_counts": {"attack": 2},
    }
    results = [
        {
            **base,
            "reward": 95.0,
            "success": True,
            "spike_activity": {
                "lif1": {"spikes": 1, "opportunities": 10, "rate": 0.1}
            },
        },
        {
            **base,
            "reward": -300.0,
            "success": False,
            "noninstant_success": False,
            "hard_spawn_success": False,
            "termination": "timeout",
            "wall_loop_suspected": True,
            "spike_activity": {
                "lif1": {"spikes": 9, "opportunities": 90, "rate": 0.1}
            },
        },
    ]

    summary = aggregate_results(results, ("move_left", "move_right", "attack"))

    assert summary["success_rate"] == 0.5
    assert summary["wall_loop_episodes"] == 1
    assert summary["spike_rates"]["lif1"] == 0.1
    assert summary["action_fractions"]["attack"] == 1.0
