import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor


class NeuralNet(nn.Module):
    """Two-layer feedforward network with GELU activation."""

    def __init__(self, hidden_size: int, out_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_size)
        self.activation = nn.GELU()

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "batch dim1"]) -> Float[Tensor, "batch dim2"]:
        return self.fc2(self.activation(self.fc1(x)))


class Projection(nn.Module):
    """Linear projection layer."""

    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.proj = nn.Linear(in_size, out_size)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "batch dim1"]) -> Float[Tensor, "batch dim2"]:
        return self.proj(x)


class Attention(nn.Module):
    """Multi-head self-attention using PyTorch's implementation."""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "batch seq dim"]
    ) -> Float[Tensor, "batch seq dim"]:
        # Self-attention: query, key, value all come from x
        attn_output, _ = self.attn(x, x, x, need_weights=False)
        return attn_output
