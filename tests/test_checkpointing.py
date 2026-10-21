from __future__ import annotations

import torch
import pytest

from checkpointing import (
    checkpoint_action_size,
    load_checkpoint,
    model_state_signature,
    save_checkpoint,
)
from networks import build_model
from networks.level_drqn import LevelDRQNQNetwork


def test_structured_checkpoint_round_trip(tmp_path):
    device = torch.device("cpu")
    source = build_model("ffnn")
    optimizer = torch.optim.Adam(source.parameters(), lr=1e-4)
    path = tmp_path / "best.pth"

    save_checkpoint(
        path,
        source,
        optimizer,
        episode=9,
        global_step=321,
        best_eval_mean=88.0,
        best_eval_completion=0.75,
        config={"model": "ffnn"},
    )
    restored = build_model("ffnn")
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4)
    metadata = load_checkpoint(path, restored, device, restored_optimizer)

    assert metadata["episode"] == 9
    assert metadata["global_step"] == 321
    assert metadata["best_eval_mean"] == 88.0
    assert metadata["best_eval_completion"] == 0.75
    assert metadata["format_version"] == 3
    assert metadata["model_signature"] == model_state_signature(restored)
    for expected, actual in zip(source.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)


def test_legacy_state_dict_is_still_supported(tmp_path):
    device = torch.device("cpu")
    source = build_model("ffnn")
    path = tmp_path / "legacy.pth"
    torch.save(source.state_dict(), path)

    restored = build_model("ffnn")
    metadata = load_checkpoint(path, restored, device)

    assert metadata["format_version"] == 0
    for expected, actual in zip(source.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)


def test_incompatible_checkpoint_config_is_rejected_before_loading(tmp_path):
    device = torch.device("cpu")
    source = build_model("ffnn")
    optimizer = torch.optim.Adam(source.parameters(), lr=1e-4)
    path = tmp_path / "incompatible.pth"
    save_checkpoint(
        path,
        source,
        optimizer,
        episode=0,
        global_step=1,
        best_eval_mean=0.0,
        config={"model": "ffnn", "action_size": 3},
    )

    restored = build_model("ffnn")
    with pytest.raises(ValueError, match="saved action_size=3, expected 8"):
        load_checkpoint(
            path,
            restored,
            device,
            expected_config={"action_size": 8},
        )


def test_incompatible_model_signature_is_rejected(tmp_path):
    device = torch.device("cpu")
    source = LevelDRQNQNetwork(hidden_size=512)
    optimizer = torch.optim.Adam(source.parameters(), lr=1e-4)
    path = tmp_path / "old-drqn-shape.pth"
    save_checkpoint(
        path,
        source,
        optimizer,
        episode=0,
        global_step=1,
        best_eval_mean=0.0,
        config={"model": "level_drqn"},
    )

    parameter_matched = LevelDRQNQNetwork(hidden_size=128)
    with pytest.raises(ValueError, match="architecture signature differs"):
        load_checkpoint(path, parameter_matched, device)


def test_action_size_inference_supports_level_drqn_output_head(tmp_path):
    model = LevelDRQNQNetwork(action_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    path = tmp_path / "level-drqn.pth"
    save_checkpoint(
        path,
        model,
        optimizer,
        episode=0,
        global_step=0,
        best_eval_mean=0.0,
        config={"model": "level_drqn"},
    )

    assert checkpoint_action_size(path) == 8


def test_checkpoint_round_trips_non_conflicting_extra_state(tmp_path):
    device = torch.device("cpu")
    model = build_model("ffnn")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    path = tmp_path / "imitation.pth"
    save_checkpoint(
        path,
        model,
        optimizer,
        episode=2,
        global_step=7,
        best_eval_mean=0.5,
        config={"training_type": "imitation"},
        extra_state={"scaler_state_dict": {}, "best_macro_recall": 0.25},
    )
    payload = load_checkpoint(path, model, device, optimizer)
    assert payload["scaler_state_dict"] == {}
    assert payload["best_macro_recall"] == pytest.approx(0.25)


def test_checkpoint_extra_state_cannot_replace_core_fields(tmp_path):
    model = build_model("ffnn")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="core checkpoint fields"):
        save_checkpoint(
            tmp_path / "bad.pth",
            model,
            optimizer,
            episode=0,
            global_step=0,
            best_eval_mean=0.0,
            config={},
            extra_state={"episode": 99},
        )
