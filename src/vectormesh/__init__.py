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
from .components.vectorizers import TextVectorizer
from .data import VectorCache

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
    "TextVectorizer",
    "VectorCache",
]

def main() -> None:
    print("Hello from vectormesh!")
