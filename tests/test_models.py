from __future__ import annotations

import pytest
import torch

from networks import build_model


@pytest.mark.parametrize("model_name", ["ffnn", "snn", "rsnn", "compact_snn", "compact_conv_snn"])
def test_model_factory_forward_contract(model_name):
    kwargs = {} if model_name == "ffnn" else {"num_steps": 2}
    model = build_model(model_name, action_size=3, **kwargs)

    q_values, spike_count = model(torch.rand(2, 1, 84, 84))

    assert q_values.shape == (2, 3)
    assert spike_count.ndim == 0


def test_model_factory_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported model"):
        build_model("typo")
