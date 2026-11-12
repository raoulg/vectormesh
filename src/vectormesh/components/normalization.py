"""Normalisation on the last axis, at any rank, chosen by name."""

from typing import Literal

from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from vectormesh.components.contracts import RankPolymorphicComponent
from vectormesh.types import VectorMeshError

NormKind = Literal["layer", "batch", "rms", "none"]


class Norm(RankPolymorphicComponent):
    """Normalise the last axis: ``(*lead, hidden_size) -> (*lead, hidden_size)``.

    One component for the usual choices, so a search can switch between them by name:

    ==========  ==================================================================
    ``kind``    normalises
    ==========  ==================================================================
    ``layer``   each vector over its own features (``nn.LayerNorm``)
    ``batch``   each feature over the batch, and over every position at rank 3
                (``nn.BatchNorm1d``; running statistics at eval)
    ``rms``     each vector by its root mean square, no centring (``nn.RMSNorm``)
    ``none``    nothing: the identity, so "no norm" is a value of the same axis
    ==========  ==================================================================

    Args:
        hidden_size: width of the last axis.
        kind: which normalisation, see the table.
    """

    def __init__(self, hidden_size: int, kind: NormKind = "layer"):
        super().__init__()
        self.hidden_size = hidden_size
        self.kind = kind
        if kind == "layer":
            self.norm: nn.Module = nn.LayerNorm(hidden_size)
        elif kind == "batch":
            self.norm = nn.BatchNorm1d(hidden_size)
        elif kind == "rms":
            self.norm = nn.RMSNorm(hidden_size)
        elif kind == "none":
            self.norm = nn.Identity()
        else:
            raise VectorMeshError(
                f"Norm kind {kind!r} is not one of 'layer', 'batch', 'rms', 'none'",
                hint="pass kind='layer' for a per-vector norm, 'batch' for a per-feature one",
            )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "*lead {self.hidden_size}"]
    ) -> Float[Tensor, "*lead {self.hidden_size}"]:
        if self.kind != "batch":
            return self.norm(tensors)
        # BatchNorm1d wants the features on axis 1: fold every leading axis into the batch.
        flat = tensors.reshape(-1, self.hidden_size)
        return self.norm(flat).reshape(tensors.shape)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, kind={self.kind!r}"
