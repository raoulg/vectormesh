"""VectorMesh components module."""

from .aggregation import (
    AttentionAggregator,
    BaseAggregator,
    MeanAggregator,
    RNNAggregator,
)
from .connectors import Concatenate2D
from .gating import Gate, Highway, Skip
from .padding import DynamicPadding, FixedPadding
from .pipelines import Parallel, Serial

__all__ = [
    # Pipelines
    "Serial",
    "Parallel",
    # Aggregation
    "BaseAggregator",
    "MeanAggregator",
    "AttentionAggregator",
    "RNNAggregator",
    # Connectors
    "Concatenate2D",
    # Padding
    "DynamicPadding",
    "FixedPadding",
    # Residual & Gating
    "Skip",
    "Gate",
    "Highway",
]
