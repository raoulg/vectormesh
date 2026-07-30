import sys
from importlib.metadata import version

from loguru import logger

from vectormesh.data import (
    BaseVectorizer,
    ImageVectorizer,
    ChunkedRegexVectorizer,
    LabelEncoder,
    RegexVectorizer,
    VectorCache,
    Vectorizer,
    build,
)

__version__ = version("vectormesh")

__all__ = [
    "VectorCache",
    "LabelEncoder",
    "build",
    "BaseVectorizer",
    "Vectorizer",
    "ImageVectorizer",
    "RegexVectorizer",
    "ChunkedRegexVectorizer",
]

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/dataset.log", rotation="10 MB", level="DEBUG")
