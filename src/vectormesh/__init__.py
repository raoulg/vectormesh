import sys
from importlib.metadata import version

from loguru import logger

from vectormesh.data import (
    BaseVectorizer,
    ChunkedRegexVectorizer,
    ChunkedVectorizer,
    HubUpload,
    ImageVectorizer,
    ItemVectorizer,
    LabelEncoder,
    PatchImageVectorizer,
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
    "BaseVectorizer",
    "ChunkedRegexVectorizer",
    "ChunkedVectorizer",
    "EarlyStopping",
    "HubUpload",
    "ImageVectorizer",
    "ItemVectorizer",
    "LabelEncoder",
    "PatchImageVectorizer",
    "RegexVectorizer",
    "Reporter",
    "Step",
    "TrainResult",
    "Trainer",
    "TrainerSettings",
    "VectorCache",
    "Vectorizer",
    "build",
]

logger.remove()
logger.add(sys.stderr, level="INFO")
