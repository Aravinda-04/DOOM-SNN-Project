"""Contract tests for the CRSNN and RSNNv2 networks.

Tests verify:
  - Output tensor shapes for single-frame and batched inputs.
  - Cross-frame hidden state is returned and accepted correctly.
  - Gradients flow through hidden state inside a training sequence.
  - Both networks share the same forward signature.
"""

from __future__ import annotations

import pytest
import torch

from networks.crsnn import CRSNNQNetwork
from networks.rsnn_v2 import RSNNv2QNetwork


ACTION_SIZE = 8
IN_CHANNELS = 4
BATCH = 2
H = W = 84


@pytest.fixture(params=["crsnn", "rsnn_v2"])
def model(request):
    cls = {"crsnn": CRSNNQNetwork, "rsnn_v2": RSNNv2QNetwork}[request.param]
    return cls(action_size=ACTION_SIZE, in_channels=IN_CHANNELS, num_steps=4)


def make_obs(batch: int = BATCH) -> torch.Tensor:
    return torch.rand(batch, IN_CHANNELS, H, W)


# ---------------------------------------------------------------------------

class TestForwardShapes:
    def test_q_values_shape(self, model):
        q, spikes, hidden = model(make_obs())
        assert q.shape == (BATCH, ACTION_SIZE), f"Expected ({BATCH},{ACTION_SIZE}), got {q.shape}"

    def test_hidden_tuple_returned(self, model):
        _, _, hidden = model(make_obs())
        assert isinstance(hidden, tuple) and len(hidden) == 2

    def test_hidden_shapes(self, model):
        _, _, (spk4, mem4) = model(make_obs())
        assert spk4.shape == (BATCH, 512)
        # mem4 may be a scalar (snntorch) or a tensor; just check it is not None.
        assert mem4 is not None

    def test_spikes_scalar(self, model):
        _, spikes, _ = model(make_obs())
        assert spikes.shape == torch.Size([]), "total_spikes must be a scalar tensor"

    def test_spike_opportunity_count_is_positive(self, model):
        assert model.spike_opportunities_per_sample > 0

    def test_single_frame_batch(self, model):
        q, spikes, hidden = model(make_obs(batch=1))
        assert q.shape == (1, ACTION_SIZE)

    def test_is_recurrent_flag(self, model):
        assert getattr(model, "is_recurrent", False) is True


class TestHiddenStateContinuity:
    def test_hidden_passed_across_frames(self, model):
        """Verify that a non-zero spk4 changes the recurrent current into lif4.

        We test the mechanism directly: the recurrent Linear applied to
        a non-zero spike vector must produce a different result than
        the same Linear applied to zeros.
        """
        spk4_zero = torch.zeros(BATCH, 512)
        spk4_nonzero = torch.rand(BATCH, 512)
        recurrent = (
            model.recurrent
            if hasattr(model, "recurrent")
            else model.lif4.recurrent
        )
        out_zero = recurrent(spk4_zero)
        out_nonzero = recurrent(spk4_nonzero)
        assert not torch.allclose(out_zero, out_nonzero), (
            "recurrent(non-zero spk4) must differ from recurrent(zeros)."
        )

    def test_recurrent_variants_are_architecturally_distinct(self, model):
        if isinstance(model, CRSNNQNetwork):
            assert isinstance(model.recurrent, torch.nn.Linear)
            assert model.recurrent.weight.shape == (512, 512)
        else:
            assert model.lif4.all_to_all is False
            recurrent_parameters = sum(
                parameter.numel()
                for parameter in model.lif4.recurrent.parameters()
            )
            assert recurrent_parameters < 512 * 512

    def test_none_hidden_resets(self, model):
        obs = make_obs()
        _, _, h1 = model(obs)
        # Passing None should behave identically to two independent calls.
        q_fresh1, _, _ = model(obs, hidden=None)
        q_fresh2, _, _ = model(obs, hidden=None)
        assert torch.allclose(q_fresh1, q_fresh2), (
            "Two calls with hidden=None and the same input must produce equal outputs."
        )

    def test_gradient_flows_across_frame_boundary(self, model):
        """A later-frame loss must depend on the preceding frame."""
        obs1 = make_obs(batch=1).requires_grad_(True)
        _, _, hidden = model(obs1, hidden=None)
        obs2 = make_obs(batch=1)
        q2, _, _ = model(obs2, hidden=hidden)

        grad = torch.autograd.grad(q2.sum(), obs1, allow_unused=True)[0]

        assert grad is not None
        assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grad).item() > 0

    def test_detached_hidden_defines_tbptt_boundary(self, model):
        obs1 = make_obs(batch=1).requires_grad_(True)
        _, _, hidden = model(obs1, hidden=None)
        boundary_hidden = tuple(value.detach() for value in hidden)
        q2, _, _ = model(make_obs(batch=1), hidden=boundary_hidden)

        grad = torch.autograd.grad(q2.sum(), obs1, allow_unused=True)[0]

        assert grad is None
