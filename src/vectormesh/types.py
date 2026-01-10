"""
Centralized types, base classes, and errors for VectorMesh.

This module contains all foundational types and classes used throughout VectorMesh.
"""

from typing import Optional, Union

from jaxtyping import Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor

OneDTensor = Float[Tensor, "?batch"]
OneDTensor.__doc__ = "1D Tensor representing a single vector. Shape: (batch,)"

TwoDTensor = Float[Tensor, "batch tokens"]
TwoDTensor.__doc__ = "2D Tensor representing, eg (batch, tokens)"

ThreeDTensor = Float[Tensor, "batch tokens dim"]
ThreeDTensor.__doc__ = "3D Tensor, eg (batch, tokens, dim)"

FourDTensor = Float[Tensor, "batch chunks tokens embed"]
FourDTensor.__doc__ = "4D Tensor representing a batch of chunked token embeddings. Shape: (batch, chunks, tokens, embed)"


NDTensor = Union[TwoDTensor, ThreeDTensor]
NDTensor.__doc__ = """
Union of all supported tensor dimensions in VectorMesh.

Currently supports 2D and 3D tensors. When 4D support is added,
just update this single type definition to include FourDTensor.

Usage:
    def process(tensor: NDTensor) -> NDTensor: ...
    def aggregate(input: NDTensor) -> TwoDTensor: ...
"""


class VectorMeshComponent(BaseModel):
    """
    Base class for all VectorMesh components.

    Enforces strict validation and immutable configuration using Pydantic.
    All vectorizers, aggregators, and combinators inherit from this base.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class VectorMeshError(Exception):
    """
    Base exception for all VectorMesh errors.

    Includes educational hints and fixes to help users understand
    and resolve tensor flow and composition issues.

    Args:
        message: Primary error message
        hint: Educational hint about what went wrong
        fix: Suggested fix or next steps

    Example:
        raise VectorMeshError(
            message="Cannot compose 2D → 3D tensors",
            hint="Shape mismatch in pipeline",
            fix="Insert MeanAggregator() to convert 3D → 2D"
        )
    """

    def __init__(
        self, message: str, hint: Optional[str] = None, fix: Optional[str] = None
    ):
        super().__init__(message)
        self.hint = hint
        self.fix = fix
