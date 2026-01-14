import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from vectormesh.types import BaseComponent


class NeuralNet(BaseComponent):
    """Two-layer feedforward network with GELU activation."""

    def __init__(self, hidden_size: int, out_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_size)
        self.activation = nn.GELU()

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch {self.hidden_size}"]
    ) -> Float[Tensor, "batch {self.out_size}"]:
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
        self, tensors: Float[Tensor, "batch {self.hidden_size}"]
    ) -> Float[Tensor, "batch {self.out_size}"]:
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
        # Self-attention: query, key, value all come from tensors
        attn_output, _ = self.attn(tensors, tensors, tensors, need_weights=False)
        return attn_output
