"""VectorMesh components module."""

from .aggregation import (
    AttentionAggregator,
    MaskedMeanAggregator,
    MeanAggregator,
    RNNAggregator,
)
from .augmentation import GaussianNoise
from .connectors import Concatenate2D, Concatenate3D, Stack2D
from .contracts import (
    BaseAggregator,
    Connector,
    RankPolymorphicComponent,
    SequenceComponent,
)
from .gating import Gate, Highway, MoE, SparseMoE
from .metrics import RepresentationStd
from .neural import Attention, NeuralNet, Projection, TransformerBlock
from .padding import DynamicPadding, FixedPadding
from .pipelines import JEPA, Lift, Parallel, Serial, Siamese, Skip

__all__ = [
    "JEPA",
    "Attention",
    "AttentionAggregator",
    "BaseAggregator",
    "Concatenate2D",
    "Concatenate3D",
    "Connector",
    "DynamicPadding",
    "FixedPadding",
    "Gate",
    "GaussianNoise",
    "Highway",
    "Lift",
    "MaskedMeanAggregator",
    "MeanAggregator",
    "MoE",
    "NeuralNet",
    "Parallel",
    "Projection",
    "RNNAggregator",
    "RankPolymorphicComponent",
    "RepresentationStd",
    "SequenceComponent",
    "Serial",
    "Siamese",
    "Skip",
    "SparseMoE",
    "Stack2D",
    "TransformerBlock",
]
