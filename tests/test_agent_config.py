from __future__ import annotations

import dataclasses

import pytest

from agent_config import (
    ACTION_CATALOGUE_HASH,
    ACTION_NAMES,
    ACTIONS,
    AGENT_CONTRACT,
    BUTTON_NAMES,
    match_button_vector,
    validate_agent_contract,
)


def test_canonical_action_contract_is_frozen_and_consistent():
    assert len(ACTIONS) == len(ACTION_NAMES) == 9
    assert len(BUTTON_NAMES) == 8
    assert AGENT_CONTRACT.action_size == 9
    assert len(ACTION_CATALOGUE_HASH) == 64
    assert dataclasses.is_dataclass(AGENT_CONTRACT)
    validate_agent_contract()


@pytest.mark.parametrize("index", range(9))
def test_every_catalogue_action_round_trips(index):
    status, matched_index = match_button_vector(ACTIONS[index])
    assert status == "supported"
    assert matched_index == index


def test_idle_and_simultaneous_spectator_inputs_are_distinguished():
    assert match_button_vector([False] * 8) == ("supported", 8)
    status, index = match_button_vector([True, False, False, False, True, False, False, False])
    assert status == "unsupported"
    assert index is None


def test_wrong_button_vector_length_is_rejected():
    with pytest.raises(ValueError, match="Expected 8"):
        match_button_vector([True, False])
