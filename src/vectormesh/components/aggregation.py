import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

from vectormesh.components.contracts import BaseAggregator


class MeanAggregator(BaseAggregator):
    """Aggregate by taking mean over chunks.
    no learnable parameters.
    """

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch _ dim"]
    ) -> Float[Tensor, "batch dim"]:
        """Mean over chunks dimension."""
        return tensors.mean(dim=1)


class MaskedMeanAggregator(BaseAggregator):
    """Mean aggregator that ignores zero-padded chunks (from FixedPadding).
    Detects padding by checking if all values in a chunk are zero.
    """

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch _ dim"]
    ) -> Float[Tensor, "batch dim"]:
        mask = tensors.abs().sum(dim=-1) > 0  # (batch, chunks), False for padded
        mask_f = mask.unsqueeze(-1).float()  # (batch, chunks, 1)
        summed = (tensors * mask_f).sum(dim=1)
        count = mask_f.sum(dim=1).clamp(min=1)
        return summed / count


class AttentionAggregator(BaseAggregator):
    """Aggregate using learnable attention over chunks.
    Because attention does not handle variable-length sequences,
    we actually get (batch, chunks, dim) where chunks is fixed.
    """

    def __init__(self, hidden_size: int):
        """initialize learnable parameters."""
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch _ dim"]
    ) -> Float[Tensor, "batch dim"]:
        # attention_weights: (batch, _, 1)
        attention_weights = torch.softmax(self.attention(tensors), dim=1)
        return (tensors * attention_weights).sum(dim=1)


class RNNAggregator(BaseAggregator):
    """Aggregate by running a GRU over the chunks and returning its final hidden state.

    Padding is skipped: a chunk that is all zeros is padding, as in `MaskedMeanAggregator`.
    The real chunks of each item move to the front in their own order, so padding at the
    end (after a padder) and gaps in the middle (after concatenating padded sequences)
    are handled alike. The GRU then runs over each item's real chunks only, and the final
    hidden state is the one after the item's last real chunk. An item that is all padding
    counts as one zero chunk.
    """

    def __init__(self, hidden_size: int):
        """initialize learnable parameters."""
        super().__init__()
        self.rnn = torch.nn.GRU(
            input_size=hidden_size, hidden_size=hidden_size, batch_first=True
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch _ dim"]
    ) -> Float[Tensor, "batch dim"]:
        mask = tensors.abs().sum(dim=-1) > 0  # (batch, chunks), False for padding
        # a stable sort on "is padding" puts the real chunks first, in their own order
        order = torch.sort((~mask).int(), dim=1, stable=True).indices
        compact = tensors.gather(1, order.unsqueeze(-1).expand_as(tensors))
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(
            compact, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)  # (num_layers, batch, dim), in input order
        return hidden[-1]
