from .base import VectorMeshComponent
from .types import OneDTensor, TwoDTensor, ThreeDTensor
from .utils import check_shapes
from .errors import VectorMeshError
from .components.vectorizers import TextVectorizer

__all__ = [
    "VectorMeshComponent",
    "OneDTensor",
    "TwoDTensor",
    "ThreeDTensor",
    "check_shapes",
    "VectorMeshError",
    "TextVectorizer",
]

def main() -> None:
    print("Hello from vectormesh!")
