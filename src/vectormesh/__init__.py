from .types import VectorMeshComponent, OneDTensor, TwoDTensor, ThreeDTensor, NDTensor, VectorMeshError
from .utils import check_shapes
from .components.aggregation import (
    BaseAggregator,
    MeanAggregator,
    MaxAggregator,
    get_aggregator,
)
from .components.vectorizers import Vectorizer
from .components.combinators import Serial, Parallel
from .components.connectors import GlobalConcat, GlobalStack
from .visualization import visualize
from .data import VectorCache
from .validation import validate_composition, validate_parallel, MorphismComposition, Morphism, TensorDimensionality
from . import zoo
from loguru import logger
import sys

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
    "Vectorizer",
    "RegexVectorizer",
    "Serial",
    "Parallel",
    "GlobalConcat",
    "GlobalStack",
    "visualize",
    "VectorCache",
    "validate_composition",
    "validate_parallel",
    "MorphismComposition",
    "Morphism",
    "TensorDimensionality",
    "zoo",
]

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/dataset.log", rotation="10 MB", level="DEBUG")
