import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from vectormesh.types import BaseComponent


class Concatenate2D(BaseComponent):
    """Concatenate tuples from parallel branches at last dimension.

    input: ((batch dim1), (batch dim2), ...)
    output: (batch ndim)

    where ndim = dim1 + dim2 + ...
    """

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: tuple[Float[Tensor, "batch dim"], ...]
    ) -> Float[Tensor, "batch ndim"]:
        return torch.cat(tensors, dim=-1)


class Concatenate3D(BaseComponent):
    """Concatenate parallel (batch, chunks, dim_i) tensors along the feature axis.

    input : ((batch chunks dim1), (batch chunks dim2), ...)
    output: (batch chunks ndim)   where ndim = dim1 + dim2 + ...

    Branches must share the same chunk dimension (e.g. both padded to max_chunks
    with FixedPadding). A padded chunk stays all-zero across the concatenation, so
    downstream padding masks (MaskedMeanAggregator, TransformerBlock) still fire.
    """

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: tuple[Float[Tensor, "batch chunks _"], ...]
    ) -> Float[Tensor, "batch chunks ndim"]:
        return torch.cat(tensors, dim=-1)


class Stack2D(BaseComponent):
    """Stack n tuples from parallel branches, default 1st dimension.

    input : ((batch dim1), (batch dim1), ...)
    output: (batch nstack dim1)

    where nstack is the number of tensors in the tuple
    """

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: tuple[Float[Tensor, "batch dim1"], ...]
    ) -> Float[Tensor, "batch nstack dim1"]:
        return torch.stack(tensors, dim=1)
