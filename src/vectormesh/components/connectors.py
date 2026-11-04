import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from vectormesh.components.contracts import Connector


class Concatenate2D(Connector):
    """Concatenate tuples from parallel branches at last dimension.

    input: ((batch dim1), (batch dim2), ...)
    output: (batch total)

    where total = dim1 + dim2 + ...

    The branch widths do NOT have to match -- fusing a 384-dim image embedding with a
    512-dim one is the ordinary case. Every branch is checked as an argument of
    `connect` (see `Connector`), so a named axis binds across branches: `"batch dim"`
    would require every branch to be the same width and reject the very case this
    component exists for. Hence the anonymous `_`. `Concatenate3D` does the same;
    `Stack2D` genuinely does need equal widths, because `torch.stack` does, and its
    named axis is therefore correct.
    """

    @jaxtyped(typechecker=beartype)
    def connect(
        self, *tensors: Float[Tensor, "batch _"]
    ) -> Float[Tensor, "batch total"]:
        return torch.cat(tensors, dim=-1)


class Concatenate3D(Connector):
    """Concatenate parallel (batch, chunks, dim_i) tensors along the feature axis.

    input : ((batch chunks dim1), (batch chunks dim2), ...)
    output: (batch chunks total)   where total = dim1 + dim2 + ...

    Branches must share the same chunk dimension (e.g. both padded to max_chunks
    with FixedPadding). A padded chunk stays all-zero across the concatenation, so
    downstream padding masks (MaskedMeanAggregator, TransformerBlock) still fire.
    """

    @jaxtyped(typechecker=beartype)
    def connect(
        self, *tensors: Float[Tensor, "batch chunks _"]
    ) -> Float[Tensor, "batch chunks total"]:
        return torch.cat(tensors, dim=-1)


class ConcatenateSequence(Connector):
    """Concatenate parallel (batch, chunks_i, dim) tensors along the chunk axis.

    input : ((batch chunks1 dim), (batch chunks2 dim), ...)
    output: (batch total dim)   where total = chunks1 + chunks2 + ...

    Puts several sequences one after the other, so a `TransformerBlock` attends over all
    of them at once: text chunks followed by image patches, or the chunks of two fields.
    The branches must share the width, so a branch usually ends in a `Projection`; the
    chunk counts may differ. Where `Concatenate3D` lengthens the feature axis, this
    lengthens the sequence axis; the rank stays 3 either way.

    Padding from each branch ends up in the middle of the joined sequence. The padding
    masks of `MaskedMeanAggregator`, `RNNAggregator` and `TransformerBlock` find all-zero
    rows at any position.
    """

    @jaxtyped(typechecker=beartype)
    def connect(
        self, *tensors: Float[Tensor, "batch _ dim"]
    ) -> Float[Tensor, "batch total dim"]:
        return torch.cat(tensors, dim=1)


class Stack2D(Connector):
    """Stack n tuples from parallel branches, default 1st dimension.

    input : ((batch dim1), (batch dim1), ...)
    output: (batch nstack dim1)

    where nstack is the number of tensors in the tuple
    """

    @jaxtyped(typechecker=beartype)
    def connect(
        self, *tensors: Float[Tensor, "batch dim1"]
    ) -> Float[Tensor, "batch nstack dim1"]:
        return torch.stack(tensors, dim=1)
