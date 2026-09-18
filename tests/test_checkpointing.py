from __future__ import annotations

import pytest
import torch

from checkpointing import checkpoint_action_size, load_checkpoint, save_checkpoint
from networks import build_model


@pytest.mark.parametrize("model_name", ["ffnn", "cnn"])
def test_structured_checkpoint_round_trip(tmp_path, model_name):
    device = torch.device("cpu")
    source = build_model(model_name)
    optimizer = torch.optim.Adam(source.parameters(), lr=1e-4)
    path = tmp_path / "best.pth"

    save_checkpoint(
        path,
        source,
        optimizer,
        episode=9,
        global_step=321,
        best_eval_mean=88.0,
        config={"model": model_name},
    )
    restored = build_model(model_name)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4)
    metadata = load_checkpoint(path, restored, device, restored_optimizer)

    assert checkpoint_action_size(path) == 3
    assert metadata["config"]["model"] == model_name
    assert metadata["episode"] == 9
    assert metadata["global_step"] == 321
    assert metadata["best_eval_mean"] == 88.0
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

