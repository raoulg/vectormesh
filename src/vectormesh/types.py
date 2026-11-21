import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache
from typing import Any, ClassVar

import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
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

    def __init__(self, message: str, hint: str | None = None, fix: str | None = None):
        super().__init__(message)
        self.hint = hint
        self.fix = fix


class Cachable(BaseModel):
    """
    Base class for cachable components.
    Enforces strict validation and immutable configuration using Pydantic.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


@jaxtyped(typechecker=beartype)
@dataclass(frozen=True)
class Padded:
    """A batch of padded sequences with the mask of its real positions.

    ``values`` is ``(batch, seq, ...)``; ``mask`` is ``(batch, seq)``, True where a
    position holds a real chunk and False where it is padding. `FixedPadding` and
    `DynamicPadding` return one, so the mask is known from the lengths, not guessed
    from the values.

    A component called on a `Padded` batch runs on ``values``. A component whose
    ``forward`` takes a ``mask`` argument (`MaskedMeanAggregator`, `RNNAggregator`,
    `AttentionAggregator`, `LastAggregator`, `Attention`, `TransformerBlock`,
    `Recurrent`, `SparseMoE`, and the forms that hold one: `Skip`, `Highway`, `MoE`)
    also gets ``mask``. The result is `Padded` again while it keeps the sequence
    axis: with the same mask when the length stays, with the component's
    ``output_mask(mask)`` when it changes the length (`Conv1D`, `Pool1D`). A result
    without the sequence axis (after an aggregator) is a plain tensor. `Serial`,
    `Parallel` and the connectors pass `Padded` batches on, so the mask travels
    through a whole pipeline.
    """

    values: Float[Tensor, "batch seq *rest"]
    mask: Bool[Tensor, "batch seq"]

    @property
    def shape(self) -> torch.Size:
        """The shape of ``values``."""
        return self.values.shape

    @property
    def lengths(self) -> Int[Tensor, "batch"]:
        """The number of real positions of every item."""
        return self.mask.sum(dim=1)

    def to(self, *args: Any, **kwargs: Any) -> "Padded":
        """``values`` and ``mask`` moved as ``Tensor.to`` moves them; the mask
        stays boolean."""
        values = self.values.to(*args, **kwargs)
        return Padded(values, self.mask.to(device=values.device))


TensorInput = Float[Tensor, "..."] | Padded | tuple[Float[Tensor, "..."] | Padded, ...]
"""Documented union for connector/wiring code that accepts either a single tensor
or a tuple of tensors, each of which may be a `Padded` batch. Concrete components
narrow to one branch and enforce it at runtime via @jaxtyped(typechecker=beartype)."""


@cache
def takes_mask(component: type[nn.Module]) -> bool:
    """Whether a component class's ``forward`` has a ``mask`` parameter."""
    return "mask" in inspect.signature(component.forward).parameters


def call_masked(
    module: nn.Module, tensors: Tensor, mask: Bool[Tensor, "batch seq"] | None
) -> Tensor:
    """``module(tensors)``, with ``mask`` passed on when there is one and
    ``module`` is a component (which then sees a `Padded` batch). For the forms
    that hold other components: `Skip`, `Highway`, `MoE`."""
    if mask is None or not isinstance(module, BaseComponent):
        return module(tensors)
    out = module(Padded(tensors, mask))
    return out.values if isinstance(out, Padded) else out


class BaseComponent(nn.Module, ABC):
    """Root class for all pipeline components.

    A component maps a tensor, or a tuple of tensors, to a tensor or a tuple of tensors
    (`TensorInput`); concrete components narrow that to the shapes they take and give.
    `Serial` and `Parallel` are components too, made of components.

    Called on a `Padded` batch, a component runs on its values and passes its mask
    on, as `Padded` describes. A class with ``takes_padded = True`` (`Serial`,
    `Parallel`, the connectors) gets the `Padded` batch itself.
    """

    takes_padded: ClassVar[bool] = False

    def __init__(self):
        super().__init__()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], Padded) and not self.takes_padded:
            return self._call_padded(args[0], *args[1:], **kwargs)
        return super().__call__(*args, **kwargs)

    def _call_padded(self, batch: Padded, *args: Any, **kwargs: Any) -> Any:
        if takes_mask(type(self)):
            kwargs["mask"] = batch.mask
        out = super().__call__(batch.values, *args, **kwargs)
        if not isinstance(out, Tensor) or out.ndim != batch.values.ndim:
            return out  # the sequence axis is gone: an aggregated result
        if out.shape[:2] == batch.mask.shape:
            return Padded(out, batch.mask)
        output_mask = getattr(self, "output_mask", None)
        if output_mask is None:
            raise VectorMeshError(
                f"{type(self).__name__} changed the sequence length of a padded batch "
                f"from {batch.mask.shape[1]} to {out.shape[1]}, and has no output_mask "
                "to say which of its output positions are padding.",
                hint="a Padded batch carries a mask over the sequence axis; a component "
                "that changes that axis has to map the mask along.",
                fix="define output_mask(mask) on the component, or pass "
                "Padded.values to it if the padding does not matter there.",
            )
        return Padded(out, output_mask(batch.mask))

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> TensorInput: ...
