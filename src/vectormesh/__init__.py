from .types import VectorMeshComponent, OneDTensor, TwoDTensor, ThreeDTensor, NDTensor, VectorMeshError
from .utils import check_shapes
from .components.aggregation import (
    BaseAggregator,
    MeanAggregator,
    MaxAggregator,
    get_aggregator,
)
from .components.vectorizers import BaseVectorizer, TwoDVectorizer, ThreeDVectorizer
from .components.combinators import Serial, Parallel
from .components.connectors import GlobalConcat, GlobalStack
from .data import VectorCache
from .validation import validate_composition, validate_parallel, MorphismComposition, Morphism, TensorDimensionality
from . import zoo

__all__ = [
    "VectorMeshComponent",
    "OneDTensor",
    "TwoDTensor",
    "ThreeDTensor",
    "NDTensor",
    "check_shapes",
    "VectorMeshError",
    "BaseAggregator",
    "MeanAggregator",
    "MaxAggregator",
    "get_aggregator",
    "BaseVectorizer",
    "TwoDVectorizer",
    "ThreeDVectorizer",
    "Serial",
    "Parallel",
    "GlobalConcat",
    "GlobalStack",
    "VectorCache",
    "validate_composition",
    "validate_parallel",
    "MorphismComposition",
    "Morphism",
    "TensorDimensionality",
    "zoo",
]

def main() -> None:
    print("Hello from vectormesh!")
