import sys
from importlib.metadata import version

from loguru import logger

from vectormesh.data import (
    BaseVectorizer,
    ImageVectorizer,
    PatchImageVectorizer,
    ChunkedRegexVectorizer,
    LabelEncoder,
    RegexVectorizer,
    VectorCache,
    Vectorizer,
    build,
)
from vectormesh.training import (
    EarlyStopping,
    Reporter,
    Step,
    Trainer,
    TrainerSettings,
    TrainResult,
)

__version__ = version("vectormesh")

__all__ = [
    "VectorCache",
    "LabelEncoder",
    "build",
    "BaseVectorizer",
    "Vectorizer",
    "ImageVectorizer",
    "PatchImageVectorizer",
    "RegexVectorizer",
    "ChunkedRegexVectorizer",
    "Trainer",
    "TrainerSettings",
    "EarlyStopping",
    "Reporter",
    "Step",
    "TrainResult",
]

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/dataset.log", rotation="10 MB", level="DEBUG")
