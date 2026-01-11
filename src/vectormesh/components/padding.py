import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class DynamicPadding:
    @jaxtyped(typechecker=beartype)
    def __call__(
        self, embeddings: list[Float[Tensor, "_ dim"]]
    ) -> Float[Tensor, "batch _ dim"]:
        """Pad sequences to the maximum length in the batch."""
        return pad_sequence(
            embeddings,
            batch_first=True,
        )


class FixedPadding:
    def __init__(self, max_chunks: int):
        self.max_chunks = max_chunks

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, embeddings: list[Float[Tensor, "_ dim"]]
    ) -> Float[Tensor, "batch {self.max_chunks} dim"]:
        padded = pad_sequence(embeddings, batch_first=True)

        current = padded.shape[1]
        if current < self.max_chunks:
            return F.pad(padded, (0, 0, 0, self.max_chunks - current))
        else:
            return padded[:, : self.max_chunks, :]
