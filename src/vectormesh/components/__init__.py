"""VectorMesh components module."""

from .aggregation import (
    AttentionAggregator,
    LastAggregator,
    MaskedMeanAggregator,
    MeanAggregator,
    RNNAggregator,
)
from .augmentation import GaussianNoise
from .connectors import Concatenate2D, Concatenate3D, ConcatenateSequence, Stack2D
from .contracts import (
    BaseAggregator,
    Connector,
    ImageComponent,
    RankPolymorphicComponent,
    SequenceComponent,
    SlidingComponent,
)
from .convolution import Conv1D, Conv2D, GlobalPool2D, Pool1D, Pool2D
from .gating import Gate, Highway, MoE, SparseMoE
from .metrics import RepresentationStd
from .neural import Attention, NeuralNet, Projection, TransformerBlock
from .normalization import Norm
from .padding import DynamicPadding, FixedPadding
from .pipelines import JEPA, Lift, Parallel, Serial, Siamese, Skip
from .recurrent import Recurrent

__all__ = [
    "JEPA",
    "Attention",
    "AttentionAggregator",
    "BaseAggregator",
    "Concatenate2D",
    "Concatenate3D",
    "ConcatenateSequence",
    "Connector",
    "Conv1D",
    "Conv2D",
    "DynamicPadding",
    "FixedPadding",
    "Gate",
    "GaussianNoise",
    "GlobalPool2D",
    "Highway",
    "ImageComponent",
    "LastAggregator",
    "Lift",
    "MaskedMeanAggregator",
    "MeanAggregator",
    "MoE",
    "NeuralNet",
    "Norm",
    "Parallel",
    "Pool1D",
    "Pool2D",
    "Projection",
    "RNNAggregator",
    "RankPolymorphicComponent",
    "Recurrent",
    "RepresentationStd",
    "SequenceComponent",
    "Serial",
    "Siamese",
    "Skip",
    "SlidingComponent",
    "SparseMoE",
    "Stack2D",
    "TransformerBlock",
]
