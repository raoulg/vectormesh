"""Residual and gating components for skip connections and gated transformations."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor


class Skip(nn.Module):
    """Residual skip connection: output = batchnorm(transform(x) + projection(x))"""

    transform: nn.Module
    projection: Optional[nn.Module]
    layernorm: nn.LayerNorm

    def __init__(
        self,
        transform: nn.Module,
        in_size: int,
        projection: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.transform = transform
        self.projection = projection  # type: ignore[unresolved-attribute]
        self.layernorm = nn.LayerNorm(in_size)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        # pre-norm (instead of post-norm) improves stability
        x = self.layernorm(x)
        residual = self.projection(x) if self.projection else x
        transformed = self.transform(x)
        return transformed + residual


class Gate(nn.Module):
    """Simple gating: output = sigmoid(W·x) * x"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.project = nn.Linear(hidden_size, hidden_size)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "batch dim"]) -> Float[Tensor, "batch dim"]:
        return F.sigmoid(self.project(x)) * x


class Highway(nn.Module):
    """Highway network: G * T(x) + (1-G) * x"""

    def __init__(self, transform: nn.Module, hidden_size: int):
        super().__init__()
        self.transform = transform
        self.project = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "batch dim"]) -> Float[Tensor, "batch dim"]:
        # pre-norm (instead of post-norm) improves stability
        x = self.norm(x)
        gate = F.sigmoid(self.project(x))
        transformed = self.transform(x)
        return gate * transformed + (1 - gate) * x


class MoE(nn.Module):
    """Mixture of Experts with top-k routing."""

    experts: nn.ModuleList
    num_experts: int
    top_k: int
    router: nn.Linear
    norm: nn.LayerNorm
    out_size: int

    def __init__(
        self,
        experts: list[nn.Module],
        hidden_size: int,
        out_size: int,
        top_k: int,
    ):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)  # type: ignore[unresolved-attribute]
        self.top_k = top_k  # type: ignore[unresolved-attribute]
        self.router = nn.Linear(hidden_size, self.num_experts)
        self.norm = nn.LayerNorm(hidden_size)
        self.out_size = out_size  # type: ignore[unresolved-attribute]

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensor: Float[Tensor, "batch dim"]
    ) -> Float[Tensor, "batch out_dim"]:
        # pre-norm (instead of post-norm) improves stability
        tensor = self.norm(tensor)
        top_k_probs, top_k_indices = self._select_top_k_experts(tensor)

        output_shape = tensor.shape[:-1] + (self.out_size,)
        output = torch.zeros(output_shape, device=tensor.device)

        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]  # (batch,)
            expert_weights = top_k_probs[:, i].unsqueeze(-1)  # (batch, 1)

            # Route each batch item to its selected expert
            for expert_id in range(self.num_experts):
                mask = expert_idx == expert_id
                if mask.any():
                    expert_input = tensor[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += expert_weights[mask] * expert_output

        return output

    def _select_top_k_experts(self, tensor: Tensor) -> tuple[Tensor, Tensor]:
        """Select top-k experts and renormalize their probabilities.
        eg
          router_probs are [0.4, 0.3, 0.2, 0.1]
          select topk=2 -> probs=[0.4, 0.3], indices=[0, 1]
          renormalized -> probs=[0.57, 0.43]
        """
        router_logits = self.router(tensor)  # (batch, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        return top_k_probs, top_k_indices
