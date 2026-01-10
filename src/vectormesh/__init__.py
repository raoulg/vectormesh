import sys
from importlib.metadata import version

from loguru import logger

from . import zoo
from .components.combinators import Parallel, Serial
from .components.connectors import GlobalConcat, GlobalStack
from .components.vectorizers import Vectorizer
from .data import VectorCache
from .types import (
    NDTensor,
    OneDTensor,
    ThreeDTensor,
    TwoDTensor,
    VectorMeshComponent,
    VectorMeshError,
)
from .utils import check_shapes
from .validation import (
    Morphism,
    MorphismComposition,
    TensorDimensionality,
    validate_composition,
    validate_parallel,
)
from .visualization import visualize

__version__ = version("vectormesh")
__all__ = [
    "VectorMeshComponent",
    "OneDTensor",
    "TwoDTensor",
    "ThreeDTensor",
    "NDTensor",
    "check_shapes",
    "VectorMeshError",
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
