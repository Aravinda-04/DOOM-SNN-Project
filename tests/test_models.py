from __future__ import annotations

import pytest
import torch

from networks import build_model


@pytest.mark.parametrize("model_name", ["cnn", "ffnn", "snn", "rsnn"])
def test_model_factory_forward_contract(model_name):
    kwargs = {} if model_name in ("cnn", "ffnn") else {"num_steps": 2}
    model = build_model(model_name, action_size=3, **kwargs)

    q_values, spike_count = model(torch.rand(2, 1, 84, 84))

    assert q_values.shape == (2, 3)
    assert spike_count.ndim == 0


def test_model_factory_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported model"):
        build_model("typo")



@pytest.mark.parametrize("action_size", [3, 5])
def test_cnn_dqn_optimizer_step(action_size):
    model = build_model("cnn", action_size=action_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    frames = torch.rand(2, 1, 84, 84)
    q_values, spikes = model(frames)
    assert q_values.shape == (2, action_size)
    assert spikes.item() == 0
    assert spikes.device == q_values.device
    selected = q_values.gather(1, torch.tensor([[0], [action_size - 1]]))
    loss = torch.nn.functional.smooth_l1_loss(selected, torch.ones_like(selected))
    optimizer.zero_grad()
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    before = model.conv1.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.conv1.weight)


def test_cnn_dueling_centering_and_batch_independence():
    model = build_model("cnn", action_size=5).eval()
    frames = torch.rand(2, 1, 84, 84)
    with torch.no_grad():
        original, _ = model(frames)
        model.fc2.bias.add_(2.0)
        shifted, _ = model(frames)
        single, _ = model(frames[:1])
        model.value_fc2.bias.add_(3.0)
        raised, _ = model(frames)
    torch.testing.assert_close(shifted, original, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(single, shifted[:1], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(raised, shifted + 3.0)
