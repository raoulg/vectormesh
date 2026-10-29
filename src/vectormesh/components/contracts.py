"""The component families: one abstract component per rung of the tensor ladder.

A family fixes the shape its members take and give, so the rung a component works on is
its class, not only its annotation:

- ``VectorComponent``, ``(batch, dim) -> (batch, out)``: ``Gate``, ``Highway``
- ``SequenceComponent``, ``(batch, seq, dim) -> (batch, seq, out)``: ``Attention``,
  ``TransformerBlock``
- ``RankPolymorphicComponent``, ``(..., dim) -> (..., out)``: ``NeuralNet``, ``Projection``,
  ``MoE``, ``SparseMoE``, ``GaussianNoise``, ``Skip``
- ``BaseAggregator``, ``(batch, _, dim) -> (batch, dim)``: the aggregators
- ``Connector``, a tuple of branches -> one tensor: ``Concatenate2D``, ``Concatenate3D``,
  ``Stack2D``

A member narrows its family's shape (``Gate`` keeps ``dim``; ``NeuralNet`` names its widths),
it never widens it. A component in no family (``Serial``, ``Parallel``, ``Lift``) takes its
shape from the components it holds.
"""

from abc import abstractmethod

from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from vectormesh.types import BaseComponent


class VectorComponent(BaseComponent):
    """Maps a batch of vectors to a batch of vectors: ``(batch, dim) -> (batch, out)``."""

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch dim"]
    ) -> Float[Tensor, "batch out"]: ...


class SequenceComponent(BaseComponent):
    """Maps a batch of sequences to a batch of sequences of the same length:
    ``(batch, seq, dim) -> (batch, seq, out)``."""

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch seq dim"]
    ) -> Float[Tensor, "batch seq out"]: ...


class RankPolymorphicComponent(BaseComponent):
    """Works at any rank on the last axis: ``(..., dim) -> (..., out)``.

    The leading axes pass through, so the same component is a head on ``(batch, dim)`` and a
    per-chunk transform on ``(batch, chunks, dim)``.
    """

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "... dim"]
    ) -> Float[Tensor, "... out"]: ...


class BaseAggregator(BaseComponent):
    """Reduces a batch of sequences to a batch of vectors: ``(batch, _, dim) -> (batch, dim)``.

    The sequence axis is anonymous: its length may differ from one batch to the next.
    """

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch _ dim"]
    ) -> Float[Tensor, "batch dim"]: ...


class Connector(BaseComponent):
    """Turns the tuple of branches a ``Parallel`` returns into one tensor.

    ``forward`` unpacks the tuple into ``connect(*tensors)``, and a member annotates
    ``connect``. beartype checks one random element of a ``tuple[X, ...]`` but every
    argument of ``*tensors``, so this is what makes every branch checked, and a named axis
    in ``connect``'s annotation bind across all branches and the output.
    """

    def forward(self, tensors: tuple[Tensor, ...]) -> Tensor:
        return self.connect(*tensors)

    @abstractmethod
    def connect(self, *tensors: Float[Tensor, "..."]) -> Float[Tensor, "..."]: ...
