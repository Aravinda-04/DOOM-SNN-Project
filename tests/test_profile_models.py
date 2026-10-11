from __future__ import annotations

from networks.level_drqn import LevelDRQNQNetwork
from profile_models import profile


def test_profiler_counts_lstm_weights_and_state():
    model = LevelDRQNQNetwork(hidden_size=128)

    result = profile(model, in_channels=4)

    assert result["parameters"] == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert result["matrix_or_kernel_parameters"] > 1_600_000
    assert result["recurrent_state_values"] == 256
    assert result["dense_macs_per_decision"] > 0
