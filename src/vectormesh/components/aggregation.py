import torch
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

from vectormesh.components.contracts import BaseAggregator
from vectormesh.types import VectorMeshError


def real_first(
    tensors: Float[Tensor, "batch seq dim"], mask: Bool[Tensor, "batch seq"]
) -> tuple[Float[Tensor, "batch seq dim"], Tensor, Tensor]:
    """Move every item's real positions to the front, in their own order.

    Returns the reordered values, the order (``compact[:, j] = tensors[:, order[:,
    j]]``) and the number of real positions per item, at least 1, on the CPU, as
    ``pack_padded_sequence`` wants it. Padding at the end and gaps in the middle
    (after `ConcatenateSequence`) are handled alike.
    """
    # a stable sort on "is padding" puts the real positions first, in their own order
    order = torch.sort((~mask).int(), dim=1, stable=True).indices
    compact = tensors.gather(1, order.unsqueeze(-1).expand_as(tensors))
    lengths = mask.sum(dim=1).clamp(min=1).cpu()
    return compact, order, lengths


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


class LastAggregator(BaseAggregator):
    """Aggregate by taking the last step. No learnable parameters.

    After a `Recurrent` this is the state after the whole sequence. With a mask (a
    `Padded` batch) it is each item's last real step; an item that is all padding
    gives its first step. Without a mask it is the last step.
    """

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        tensors: Float[Tensor, "batch seq dim"],
        mask: Bool[Tensor, "batch seq"] | None = None,
    ) -> Float[Tensor, "batch dim"]:
        if mask is None:
            return tensors[:, -1]
        position = torch.arange(1, mask.shape[1] + 1, device=mask.device)
        last = ((position * mask).amax(dim=1) - 1).clamp(min=0)  # (batch,)
        return tensors[torch.arange(tensors.shape[0], device=tensors.device), last]


class MaskedMeanAggregator(BaseAggregator):
    """Mean over the real chunks of each item, padding left out.

    Needs the mask: call it on a `Padded` batch (from `FixedPadding` or
    `DynamicPadding`, through any components before it) or pass ``mask``. Without a
    mask it raises, since it cannot tell padding from data. An item that is all
    padding gives a zero vector.
    """

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        tensors: Float[Tensor, "batch seq dim"],
        mask: Bool[Tensor, "batch seq"] | None = None,
    ) -> Float[Tensor, "batch dim"]:
        if mask is None:
            raise VectorMeshError(
                "MaskedMeanAggregator got a plain tensor, without a padding mask.",
                hint="the mask comes with a Padded batch, as FixedPadding and "
                "DynamicPadding return it; padding is not guessed from the values.",
                fix="pad with FixedPadding/DynamicPadding and keep the Padded batch "
                "(Serial passes it on), pass mask=..., or use MeanAggregator.",
            )
        mask_f = mask.unsqueeze(-1).float()  # (batch, chunks, 1)
        summed = (tensors * mask_f).sum(dim=1)
        count = mask_f.sum(dim=1).clamp(min=1)
        return summed / count


class AttentionAggregator(BaseAggregator):
    """Aggregate by a learned attention weight per chunk: a softmax over the chunks
    of each item, then the weighted sum.

    With a mask (a `Padded` batch) padded chunks get weight 0; an item that is all
    padding is weighted over all its chunks.
    """

    def __init__(self, hidden_size: int):
        """initialize learnable parameters."""
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        tensors: Float[Tensor, "batch seq dim"],
        mask: Bool[Tensor, "batch seq"] | None = None,
    ) -> Float[Tensor, "batch dim"]:
        scores = self.attention(tensors)  # (batch, seq, 1)
        if mask is not None:
            keep = mask | ~mask.any(dim=1, keepdim=True)
            scores = scores.masked_fill(~keep.unsqueeze(-1), float("-inf"))
        attention_weights = torch.softmax(scores, dim=1)
        return (tensors * attention_weights).sum(dim=1)


class RNNAggregator(BaseAggregator):
    """Aggregate by running a GRU over the chunks and returning its final hidden state.

    With a mask (a `Padded` batch) padding is skipped: the real chunks of each item move
    to the front in their own order, so padding at the end (after a padder) and gaps in
    the middle (after `ConcatenateSequence`) are handled alike. The GRU then runs over
    each item's real chunks only, and the final hidden state is the one after the item's
    last real chunk. An item that is all padding counts as one chunk. Without a mask
    every chunk is real.
    """

    def __init__(self, hidden_size: int):
        """initialize learnable parameters."""
        super().__init__()
        self.rnn = torch.nn.GRU(
            input_size=hidden_size, hidden_size=hidden_size, batch_first=True
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        tensors: Float[Tensor, "batch seq dim"],
        mask: Bool[Tensor, "batch seq"] | None = None,
    ) -> Float[Tensor, "batch dim"]:
        if mask is None:
            _, hidden = self.rnn(tensors)
            return hidden[-1]
        compact, _, lengths = real_first(tensors, mask)
        packed = pack_padded_sequence(
            compact, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)  # (num_layers, batch, dim), in input order
        return hidden[-1]
