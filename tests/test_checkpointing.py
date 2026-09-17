from __future__ import annotations

import torch

from checkpointing import load_checkpoint, save_checkpoint
from networks import build_model


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
        config={"model": "ffnn"},
    )
    restored = build_model("ffnn")
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4)
    metadata = load_checkpoint(path, restored, device, restored_optimizer)

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

