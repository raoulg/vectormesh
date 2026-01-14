"""Residual and gating components for skip connections and gated transformations."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from vectormesh.types import BaseComponent


class Skip(BaseComponent):
    """Residual skip connection: output = batchnorm(transform(x) + projection(x))
    - transform is the pipeline we want to apply to the input
    - in_size is the dimensionality of the input; we need this for the layernorm
    - projection is an optional pipeline, eg a Linear(in_size, out_size) if the
    transform changes the dimensionality.
    """

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
        self.projection = projection
        self.layernorm = nn.LayerNorm(in_size)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        # pre-norm (instead of post-norm) improves stability
        x = self.layernorm(x)
        residual = self.projection(x) if self.projection else x
        transformed = self.transform(x)
        return transformed + residual


class Gate(BaseComponent):
    """Simple gating: output = sigmoid(W·x) * x"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.project = nn.Linear(hidden_size, hidden_size)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "batch dim"]) -> Float[Tensor, "batch dim"]:
        return F.sigmoid(self.project(x)) * x


class Highway(BaseComponent):
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


class MoE(BaseComponent):
    """
    See https://arxiv.org/abs/1701.06538 for paper
    """

    def __init__(self, experts, hidden_size, out_size, top_k, noisy_gating=True):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.router = nn.Linear(hidden_size, len(experts))

        self.w_noise = nn.Linear(hidden_size, len(experts))
        self.noisy_gating = noisy_gating
        self.top_k = top_k
        self.num_experts = len(experts)
        self.out_size = out_size

    def forward(self, x):
        clean_logits = self.router(x)

        # self.training is automatically managed by .eval() and .train()
        if self.noisy_gating and self.training:
            raw_noise_stddev = self.w_noise(x)
            noise_stddev = F.softplus(raw_noise_stddev) + 1e-2
            noise = torch.randn_like(clean_logits) * noise_stddev
            noisy_logits = clean_logits + noise
        else:
            noisy_logits = clean_logits

        # We set non-top-k logits to -inf so Softmax drives them to absolute zero
        top_logits, top_indices = noisy_logits.topk(self.top_k, dim=1)
        full_logits = torch.full_like(noisy_logits, float("-inf"))
        full_logits.scatter_(1, top_indices, top_logits)

        router_probs = F.softmax(full_logits, dim=1)

        final_output = torch.zeros(x.size(0), self.out_size, device=x.device)

        for i in range(self.num_experts):
            mask = (top_indices == i).any(dim=1)

            if mask.any():
                expert_input = x[mask]
                expert_output = self.experts[i](expert_input)

                expert_weights = router_probs[mask, i].unsqueeze(-1)

                final_output[mask] += expert_output * expert_weights

        # TODO: the paper implements importance loss
        # to encourage balanced expert usage
        #
        # importance = router_probs.sum(dim=0)
        # imp_loss = (importance.std() / (importance.mean() + 1e-10)).pow(2)
        # return final_output, imp_loss

        return final_output
