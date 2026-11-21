"""The component families: one abstract component per rung of the tensor ladder.

A family fixes the shape its members take and give, so the rung a component works on is
its class, not only its annotation:

- ``RankPolymorphicComponent``, ``(*lead, dim) -> (*lead, out)``: ``NeuralNet``,
  ``Projection``, ``Gate``, ``Highway``, ``MoE``, ``SparseMoE``, ``GaussianNoise``
- ``SequenceComponent``, ``(batch, seq, dim) -> (batch, seq, out)``: ``Attention``,
  ``TransformerBlock``, ``Recurrent``
- ``SlidingComponent``, ``(batch, seq, dim) -> (batch, seq_out, out)``: ``Conv1D``,
  ``Pool1D``
- ``ImageComponent``, ``(batch, channels, height, width) -> (batch, out, height_out,
  width_out)``: ``Conv2D``, ``Pool2D``
- ``BaseAggregator``, ``(batch, _, dim) -> (batch, dim)``: the aggregators
- ``Connector``, a tuple of branches -> one tensor: ``Concatenate2D``, ``Concatenate3D``,
  ``Stack2D``

A member narrows its family's shape (``Gate`` keeps its width; ``NeuralNet`` names its
widths), it never widens it. A component in no family is an architecture form (``Serial``,
``Parallel``, ``Skip``, ``Lift``): it takes its shape from the components it holds. The
one other component in no family is ``GlobalPool2D``, the step from an image to a
vector, ``(batch, channels, height, width) -> (batch, channels)``.
"""

from abc import abstractmethod

import torch
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from torch import Tensor

from vectormesh.types import BaseComponent, Padded, VectorMeshError


class SequenceComponent(BaseComponent):
    """Maps a batch of sequences to a batch of sequences of the same length:
    ``(batch, seq, dim) -> (batch, seq, out)``."""

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch seq dim"]
    ) -> Float[Tensor, "batch seq out"]: ...


class SlidingComponent(BaseComponent):
    """Slides a window along the sequence axis: ``(batch, seq, dim) -> (batch, seq_out, out)``.

    The window may shorten the sequence, so the output length has its own name. The
    channels are the last axis, as everywhere else in the library.
    """

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch seq dim"]
    ) -> Float[Tensor, "batch seq_out out"]: ...


class ImageComponent(BaseComponent):
    """Maps a batch of images to a batch of images:
    ``(batch, channels, height, width) -> (batch, out, height_out, width_out)``.

    PyTorch's image layout: the channels are the second axis, not the last.
    """

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch channels height width"]
    ) -> Float[Tensor, "batch out height_out width_out"]: ...


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

    When a branch is a `Padded` batch, ``connect`` joins the values and
    ``connect_mask`` the masks, and the result is `Padded`. A branch without a mask
    counts as all real. A connector without ``connect_mask`` rejects `Padded` branches.
    """

    takes_padded = True

    def forward(
        self, tensors: tuple[Float[Tensor, "..."] | Padded, ...]
    ) -> Float[Tensor, "..."] | Padded:
        values = [t.values if isinstance(t, Padded) else t for t in tensors]
        if not any(isinstance(t, Padded) for t in tensors):
            return self.connect(*values)
        masks = [
            t.mask
            if isinstance(t, Padded)
            else torch.ones(t.shape[:2], dtype=torch.bool, device=t.device)
            for t in tensors
        ]
        mask = self.connect_mask(*masks)  # first: a connector without masks says so
        return Padded(self.connect(*values), mask)

    @abstractmethod
    def connect(self, *tensors: Float[Tensor, "..."]) -> Float[Tensor, "..."]: ...

    def connect_mask(self, *masks: Bool[Tensor, "batch _"]) -> Bool[Tensor, "batch _"]:
        """The mask of the joined branches, from the mask of each branch."""
        raise VectorMeshError(
            f"{type(self).__name__} got a Padded branch, but has no way to join masks.",
            hint="a padded sequence has to keep its sequence axis to keep its mask; "
            "this connector joins vectors.",
            fix="aggregate each padded branch (e.g. MaskedMeanAggregator) before "
            "this connector, or use Concatenate3D / ConcatenateSequence.",
        )
