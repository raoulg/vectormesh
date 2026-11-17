"""Activations, chosen by name."""

from typing import Literal, get_args

from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from vectormesh.components.contracts import RankPolymorphicComponent
from vectormesh.types import VectorMeshError

ActivationKind = Literal["gelu", "relu", "tanh", "sigmoid", "none"]


class Activation(RankPolymorphicComponent):
    """Apply a nonlinearity elementwise: ``(*lead, hidden) -> (*lead, hidden)``.

    One component for the usual choices, so a search can switch between them by name:

    ===========  ==========================================
    ``kind``     applies
    ===========  ==========================================
    ``gelu``     ``nn.GELU``
    ``relu``     ``nn.ReLU``
    ``tanh``     ``nn.Tanh``
    ``sigmoid``  ``nn.Sigmoid``
    ``none``     nothing: the identity, so "no activation" is a value of the same axis
    ===========  ==========================================

    Args:
        kind: which activation, see the table.
    """

    def __init__(self, kind: ActivationKind = "gelu"):
        super().__init__()
        self.kind = kind
        kinds: dict[str, nn.Module] = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "none": nn.Identity(),
        }
        if kind not in kinds:
            listed = ", ".join(repr(k) for k in get_args(ActivationKind))
            raise VectorMeshError(
                f"Activation kind {kind!r} is not one of {listed}",
                hint="pass kind='gelu' for the default, or 'none' for no activation",
            )
        self.activation = kinds[kind]

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "*lead hidden"]
    ) -> Float[Tensor, "*lead hidden"]:
        return self.activation(tensors)

    def extra_repr(self) -> str:
        return f"kind={self.kind!r}"
