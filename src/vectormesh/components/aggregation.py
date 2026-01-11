from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor


class BaseAggregator(ABC, nn.Module):
    """Base class for aggregating 3D -> 2D tensors.
    We use "forward" to be compatible with nn.Module
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def forward(
        self, embeddings: Float[Tensor, "batch _ dim"]
    ) -> Float[Tensor, "batch dim"]:
        """Aggregate from (batch, chunks, dim) to (batch, dim)."""
        ...


class MeanAggregator(BaseAggregator):
    """Aggregate by taking mean over chunks.
    no learnable parameters.
    """

    @jaxtyped(typechecker=beartype)
    def forward(
        self, embeddings: Float[Tensor, "batch _ dim"]
    ) -> Float[Tensor, "batch dim"]:
        """Mean over chunks dimension."""
        return embeddings.mean(dim=1)


class AttentionAggregator(BaseAggregator):
    """Aggregate using learnable attention over chunks."""

    def __init__(self, hidden_size: int):
        """initialize learnable parameters."""
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, embeddings: Float[Tensor, "batch _ dim"]
    ) -> Float[Tensor, "batch dim"]:
        # attention_weights: (batch, _, 1)
        attention_weights = torch.softmax(self.attention(embeddings), dim=1)
        return (embeddings * attention_weights).sum(dim=1)


class RNNAggregator(BaseAggregator):
    """Aggregate using RNN over chunks.
    return final hidden state.
    """

    def __init__(self, hidden_size: int):
        """initialize learnable parameters."""
        super().__init__()
        self.rnn = torch.nn.GRU(
            input_size=hidden_size, hidden_size=hidden_size, batch_first=True
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, embeddings: Float[Tensor, "batch _ dim"]
    ) -> Float[Tensor, "batch dim"]:
        output, _ = self.rnn(embeddings)
        return output[:, -1, :]
