import torch.nn as nn
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from torch import Tensor

from vectormesh.types import BaseComponent, VectorMeshError


class NeuralNet(BaseComponent):
    """Feedforward network with GELU activations, two layers by default.

    Acts on the last dimension, so it accepts any leading shape: ``(batch, hidden)``
    or ``(batch, seq, hidden)`` (e.g. as a position-wise expert inside MoE).

    ``hidden_size`` takes either an ``int`` (today's default: one hidden layer the
    same width as the input, `NeuralNet(384, 47)`, unchanged) or a ``list[int]``
    naming the *whole* width chain from the input dimension through every hidden
    layer -- `NeuralNet([384, 128], 47)` is `Linear(384, 128) -> GELU ->
    Linear(128, 47)`, `NeuralNet([384, 128, 128], 47)` adds another `128 -> 128`
    hidden layer, and so on. A single-element list, `NeuralNet([384], 47)`, means
    exactly what the bare int does: both spellings are accepted precisely so nothing
    forces a choice.

    Depth belongs here, as *one* component, rather than as several composed
    `NeuralNet`s. That composition is a trap: `NeuralNet` ends in a bare `Linear`
    with no trailing activation, so `Serial([NeuralNet(d, w), NeuralNet(w, k)])`
    puts two `Linear` layers back-to-back with nothing nonlinear between them --
    the exact "depth without nonlinearity buys nothing" collapse this course's own
    notebook 02 proves. `hidden_size=[d, w]` builds `Linear(d, w) -> GELU ->
    Linear(w, out_size)` instead, so a GELU sits between every pair of `Linear`s and
    the collapse cannot happen by construction.
    """

    def __init__(self, hidden_size: "int | list[int]", out_size: int):
        super().__init__()
        widths = (
            [hidden_size, hidden_size]
            if isinstance(hidden_size, int)
            else list(hidden_size)
        )
        if not widths:
            raise VectorMeshError(
                "hidden_size=[] has no input dimension to build from -- pass at "
                "least one width (the input dimension)."
            )
        if len(widths) == 1:
            # A lone width doubles as both the input dimension and the one hidden
            # layer's width, matching what the bare-int form has always meant.
            widths = widths * 2

        self.hidden_size = widths[0]  # the true input dim, for the shape contract below
        self.out_size = out_size
        self.widths = widths  # the full input->hidden chain, excluding out_size

        all_widths = [*widths, out_size]
        self.layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(all_widths, all_widths[1:])
        )
        self.activation = nn.GELU()

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "... {self.hidden_size}"]
    ) -> Float[Tensor, "... {self.out_size}"]:
        layers = list(self.layers)
        x = tensors
        for layer in layers[:-1]:
            x = self.activation(layer(x))
        return layers[-1](x)


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
