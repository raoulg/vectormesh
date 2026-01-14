from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

from beartype import beartype
from jaxtyping import Float, jaxtyped
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn


class VectorMeshError(Exception):
    """
    Base exception for all VectorMesh errors.

    Includes educational hints and fixes to help users understand
    and resolve tensor flow and composition issues.

    Args:
        message: Primary error message
        hint: Educational hint about what went wrong
        fix: Suggested fix or next steps
    """

    def __init__(
        self, message: str, hint: Optional[str] = None, fix: Optional[str] = None
    ):
        super().__init__(message)
        self.hint = hint
        self.fix = fix


class Cachable(BaseModel):
    """
    Base class for cachable components.
    Enforces strict validation and immutable configuration using Pydantic.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


TensorInput = Union[Float[Tensor, "..."], Tuple[Float[Tensor, "..."], ...]]


class BaseComponent(nn.Module, ABC):
    """Root class for all pipeline components."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def forward(self, tensors: TensorInput) -> Float[Tensor, "..."]: ...
