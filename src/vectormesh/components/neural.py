import torch.nn as nn
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from torch import Tensor

from vectormesh.types import BaseComponent


class NeuralNet(BaseComponent):
    """Two-layer feedforward network with GELU activation.

    Acts on the last dimension, so it accepts any leading shape: ``(batch, hidden)``
    or ``(batch, seq, hidden)`` (e.g. as a position-wise expert inside MoE).
    """

    def __init__(self, hidden_size: int, out_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_size)
        self.activation = nn.GELU()

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "... {self.hidden_size}"]
    ) -> Float[Tensor, "... {self.out_size}"]:
        return self.fc2(self.activation(self.fc1(tensors)))


class Projection(BaseComponent):
    """Linear projection layer."""

    def __init__(self, hidden_size: int, out_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.proj = nn.Linear(hidden_size, out_size)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "... {self.hidden_size}"]
    ) -> Float[Tensor, "... {self.out_size}"]:
        return self.proj(tensors)


class Attention(nn.Module):
    """Multi-head self-attention using PyTorch's implementation."""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch seq {self.hidden_size}"]
    ) -> Float[Tensor, "batch seq {self.hidden_size}"]:
        attn_output, _ = self.attn(tensors, tensors, tensors, need_weights=False)
        return attn_output


class TransformerBlock(BaseComponent):
    """A minimal pre-norm transformer block: attention + feed-forward, each with
    a residual connection.

        x = x + Attention(norm1(x))
        x = x + FFN(norm2(x))

    Shape-preserving: both residual additions require the block to return the same
    width it received, so there is a single ``hidden_size`` and no output size to
    choose. Put a ``Projection`` before or after the block to change width.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ff = NeuralNet(hidden_size, hidden_size)

    @staticmethod
    def _pad_mask(tensors: Float[Tensor, "batch seq dim"]) -> Bool[Tensor, "batch seq"]:
        """Reconstruct the key padding mask from all-zero (padded) positions.
        This avoids the need to pass the mask through the pipeline

        A fully-padded row would make attention's softmax produce NaNs,
        so such a (pathological) row is treated as fully valid instead.
        """
        pad_mask = tensors.abs().sum(dim=-1) == 0  # (batch, seq)
        return pad_mask & ~pad_mask.all(dim=1, keepdim=True)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch seq {self.hidden_size}"]
    ) -> Float[Tensor, "batch seq {self.hidden_size}"]:
        pad_mask = self._pad_mask(tensors)

        normed = self.norm1(tensors)
        attn_out, _ = self.attn(
            normed,
            normed,
            normed,
            key_padding_mask=pad_mask,
            need_weights=False,
        )
        tensors = tensors + attn_out
        tensors = tensors + self.ff(self.norm2(tensors))
        return tensors
