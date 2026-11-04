"""The component families: one abstract component per rung of the tensor ladder.

A family fixes the shape its members take and give, so the rung a component works on is
its class, not only its annotation:

- ``RankPolymorphicComponent``, ``(*lead, dim) -> (*lead, out)``: ``NeuralNet``,
  ``Projection``, ``Gate``, ``Highway``, ``MoE``, ``SparseMoE``, ``GaussianNoise``
- ``SequenceComponent``, ``(batch, seq, dim) -> (batch, seq, out)``: ``Attention``,
  ``TransformerBlock``
- ``BaseAggregator``, ``(batch, _, dim) -> (batch, dim)``: the aggregators
- ``Connector``, a tuple of branches -> one tensor: ``Concatenate2D``, ``Concatenate3D``,
  ``Stack2D``

A member narrows its family's shape (``Gate`` keeps its width; ``NeuralNet`` names its
widths), it never widens it. A component in no family is an architecture form (``Serial``,
``Parallel``, ``Skip``, ``Lift``): it takes its shape from the components it holds.
"""

from abc import abstractmethod

from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from vectormesh.types import BaseComponent


class SequenceComponent(BaseComponent):
    """Maps a batch of sequences to a batch of sequences of the same length:
    ``(batch, seq, dim) -> (batch, seq, out)``."""

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch seq dim"]
    ) -> Float[Tensor, "batch seq out"]: ...


class RankPolymorphicComponent(BaseComponent):
    """Works at any rank on the last axis: ``(*lead, dim) -> (*lead, out)``.

    The leading axes pass through, so the same component is a head on ``(batch, dim)`` and a
    per-chunk transform on ``(batch, chunks, dim)``. ``*lead`` is a named variadic axis: it
    binds, so the output must have the leading axes the input had. ``...`` would not bind.
    """

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "*lead dim"]
    ) -> Float[Tensor, "*lead out"]: ...


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

    def forward(self, tensors: tuple[Tensor, ...]) -> Float[Tensor, "..."]:
        return self.connect(*tensors)

    @abstractmethod
    def connect(self, *tensors: Float[Tensor, "..."]) -> Float[Tensor, "..."]: ...
