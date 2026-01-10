from abc import abstractmethod

from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class BaseAggregator:
    """Base class for aggregating 3D -> 2D tensors."""

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def __call__(
        self, embeddings: Float[Tensor, "batch _ dim"]
    ) -> Float[Tensor, "batch dim"]:
        """Aggregate from (batch, chunks, dim) to (batch, dim)."""
        ...


class MeanAggregator(BaseAggregator):
    """Aggregate by taking mean over chunks."""

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, embeddings: Float[Tensor, "batch _ dim"]
    ) -> Float[Tensor, "batch dim"]:
        """Mean over chunks dimension."""
        return embeddings.mean(dim=1)


class DynamicPadding:
    @jaxtyped(typechecker=beartype)
    def __call__(
        self, embeddings: list[Float[Tensor, "chunks dim"]]
    ) -> Float[Tensor, "batch max_chunks dim"]:
        """Pad sequences to the maximum length in the batch."""
        return pad_sequence(
            embeddings,
            batch_first=True,
        )
