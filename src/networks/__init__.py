"""Network registry used by training, evaluation, and export commands."""

from .ffnn import FeedForwardQNetwork
from .rsnn import RSNNQNetwork
from .snn import SpikingQNetwork
from .compact_snn import CompactSpikingQNetwork
from .compact_conv_snn import CompactConvSpikingQNetwork


MODEL_CLASSES = {
    "ffnn": FeedForwardQNetwork,
    "rsnn": RSNNQNetwork,
    "snn": SpikingQNetwork,
    "compact_snn": CompactSpikingQNetwork,
    "compact_conv_snn": CompactConvSpikingQNetwork,
}
MODEL_NAMES = tuple(MODEL_CLASSES)


def build_model(model_name: str, action_size: int = 3, **kwargs):
    """Build a supported model or fail with an actionable error."""
    normalized_name = model_name.lower()
    try:
        model_class = MODEL_CLASSES[normalized_name]
    except KeyError as exc:
        choices = ", ".join(MODEL_NAMES)
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose one of: {choices}."
        ) from exc
    return model_class(action_size=action_size, **kwargs)


__all__ = [
    "CompactSpikingQNetwork",
    "CompactConvSpikingQNetwork",
    "FeedForwardQNetwork",
    "MODEL_CLASSES",
    "MODEL_NAMES",
    "RSNNQNetwork",
    "SpikingQNetwork",
    "build_model",
]
