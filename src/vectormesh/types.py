from typing import Optional

from pydantic import BaseModel, ConfigDict


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
