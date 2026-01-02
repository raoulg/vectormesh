from .base import VectorMeshComponent
from .types import OneDTensor, TwoDTensor, ThreeDTensor
from .utils import check_shapes
from .errors import VectorMeshError
from .components.aggregation import (
    BaseAggregator,
    MeanAggregator,
    MaxAggregator,
    get_aggregator,
)
from .components.vectorizers import BaseVectorizer, TwoDVectorizer, ThreeDVectorizer, TextVectorizer
from .data import VectorCache
from . import zoo

__all__ = [
    "VectorMeshComponent",
    "OneDTensor",
    "TwoDTensor",
    "ThreeDTensor",
    "check_shapes",
    "VectorMeshError",
    "BaseAggregator",
    "MeanAggregator",
    "MaxAggregator",
    "get_aggregator",
    "BaseVectorizer",
    "TwoDVectorizer",
    "ThreeDVectorizer",
    "TextVectorizer",
    "VectorCache",
    "zoo",
]

def main() -> None:
    print("Hello from vectormesh!")
