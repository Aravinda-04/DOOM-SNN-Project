from __future__ import annotations

import dataclasses

import pytest

from agent_config import (
    ACTION_CATALOGUE_HASH,
    ACTION_NAMES,
    ACTIONS,
    AGENT_CONTRACT,
    BUTTON_NAMES,
    CONTRACT_VERSION,
    match_button_vector,
    resolve_tic_actions,
    validate_agent_contract,
)


def test_canonical_action_contract_is_frozen_and_consistent():
    assert CONTRACT_VERSION == 2
    assert len(ACTIONS) == len(ACTION_NAMES) == 22
    assert len(BUTTON_NAMES) == 8
    assert AGENT_CONTRACT.action_size == 22
    assert len(ACTION_CATALOGUE_HASH) == 64
    assert dataclasses.is_dataclass(AGENT_CONTRACT)
    validate_agent_contract()


def test_action_output_order_is_explicitly_frozen():
    assert ACTION_NAMES == (
        "forward",
        "backward",
        "strafe_left",
        "strafe_right",
        "turn_left",
        "turn_right",
        "attack",
        "use",
        "noop",
        "forward_turn_left",
        "forward_turn_right",
        "forward_strafe_left",
        "forward_strafe_right",
        "forward_attack",
        "backward_attack",
        "strafe_left_attack",
        "strafe_right_attack",
        "turn_left_attack",
        "turn_right_attack",
        "forward_turn_left_attack",
        "forward_turn_right_attack",
        "forward_use",
    )


@pytest.mark.parametrize("index", range(22))
def test_every_catalogue_action_round_trips(index):
    status, matched_index = match_button_vector(ACTIONS[index])
    assert status == "supported"
    assert matched_index == index


def test_idle_and_simultaneous_spectator_inputs_are_distinguished():
    assert match_button_vector([False] * 8) == ("supported", 8)
    assert match_button_vector(
        [True, False, False, False, True, False, False, False]
    ) == ("supported", 9)
    status, index = match_button_vector(
        [False, True, False, False, False, False, False, True]
    )
    assert status == "unsupported"
    assert index is None


def test_wrong_button_vector_length_is_rejected():
    with pytest.raises(ValueError, match="Expected 8"):
        match_button_vector([True, False])


def test_four_tic_majority_tolerates_one_transition_tic():
    action_index, valid, agreement = resolve_tic_actions(
        [ACTIONS[0], ACTIONS[0], ACTIONS[0], ACTIONS[9]]
    )
    assert action_index == 0
    assert valid is True
    assert agreement == pytest.approx(0.75)


def test_two_two_tic_tie_is_masked():
    action_index, valid, agreement = resolve_tic_actions(
        [ACTIONS[0], ACTIONS[0], ACTIONS[9], ACTIONS[9]]
    )
    assert action_index in {0, 9}
    assert valid is False
    assert agreement == pytest.approx(0.5)


def test_unsupported_tics_reduce_majority_agreement():
    unsupported = [False, True, False, False, False, False, False, True]
    action_index, valid, agreement = resolve_tic_actions(
        [ACTIONS[0], ACTIONS[0], unsupported, unsupported]
    )
    assert action_index == 0
    assert valid is False
    assert agreement == pytest.approx(0.5)
