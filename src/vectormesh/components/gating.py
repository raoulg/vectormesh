"""Residual and gating components for skip connections and gated transformations."""

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from vectormesh.types import BaseComponent, VectorMeshError


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
    def forward(self, tensors: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        # pre-norm (instead of post-norm) improves stability
        tensors = self.layernorm(tensors)
        residual = self.projection(tensors) if self.projection else tensors
        transformed = self.transform(tensors)
        return transformed + residual


class Gate(BaseComponent):
    """Simple gating: output = sigmoid(W·x) * x"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.project = nn.Linear(hidden_size, hidden_size)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch dim"]
    ) -> Float[Tensor, "batch dim"]:
        return F.sigmoid(self.project(tensors)) * tensors


class Highway(BaseComponent):
    """Highway network: G * T(x) + (1-G) * x"""

    def __init__(self, transform: nn.Module, hidden_size: int):
        super().__init__()
        self.transform = transform
        self.project = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch dim"]
    ) -> Float[Tensor, "batch dim"]:
        # pre-norm (instead of post-norm) improves stability
        tensors = self.norm(tensors)
        gate = F.sigmoid(self.project(tensors))
        transformed = self.transform(tensors)
        return gate * transformed + (1 - gate) * tensors


class MoE(BaseComponent):
    """Dense mixture of experts: a softmax-weighted blend of all experts.

    The natural "multi-gate" generalisation of the gating family in this module::

        Gate:     sigmoid(Wx) * x                    # 1 gate,  1 transform
        Highway:  g * T(x) + (1 - g) * x             # 1 gate,  2 experts (T, identity)
        MoE:      sum_i softmax(Wx)_i * expert_i(x)  # N gates, N experts

    This is essentially the original Jacobs & Jordan (Adaptive mixtures of local experts, 1991) formulation, not optimized for
    thousands of experts like the Shazeer et al. (2017) paper.

    Routing is per position, so the layer accepts both (batch, hidden) and
    (batch, seq, hidden) inputs: the same softmax-blend is applied along the
    last axis at every position.

    Experts must map (..., hidden_size) to (..., out_size)

    Args:
        experts: modules mapping (..., hidden_size) -> (..., out_size).
        hidden_size: input dimensionality (also the router input size).
        out_size: output dimensionality of each expert.
    """

    def __init__(
        self, experts: list[nn.Module], hidden_size: int, out_size: int
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.router = nn.Linear(hidden_size, len(experts))
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_experts = len(experts)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "... {self.hidden_size}"]
    ) -> Float[Tensor, "... {self.out_size}"]:
        # Per-position gates over experts; works for 2D and 3D alike.
        gates = F.softmax(self.router(tensors), dim=-1)  # (..., num_experts)
        # stack experts on dim -2: (..., num_experts, out_size)
        expert_outputs = torch.stack([e(tensors) for e in self.experts], dim=-2)
        # sum over experts weighted by gates: (..., out_size)
        return (gates.unsqueeze(-1) * expert_outputs).sum(dim=-2)


class SparseMoE(BaseComponent):
    """Sparse (top-k) mixture of experts: each example activates only its top-k
    experts, and -- unlike `MoE` above -- only those experts actually run on it.

    `MoE` is the Jacobs & Jordan (1991) formulation: every expert runs on every
    input and the outputs are softmax-blended, so compute per example is `O(N)` in
    the number of experts regardless of how many are actually useful for that
    example. `SparseMoE` is the Shazeer et al. (2017, *Outrageously Large Neural
    Networks*) / Switch Transformer (Fedus et al., 2021) formulation: the router
    picks the top-`k` experts per example, each example is *gathered* into only
    those experts' input batches, and the outputs are *scattered* back weighted by
    the (renormalised, over just the top-k) softmax score. An expert an example did
    not route to never sees that example's tensor -- there is no "compute
    everything then mask" step, which would still cost `O(N)` and defeat the point.

    That gather/scatter is what decouples capacity from compute: total parameters
    scale with `N` (add experts freely), compute per example scales with `k`
    (usually 1 or 2), so a model can hold far more capacity than it spends compute
    on any single input. `k=1` is Switch-style (one expert per example, the simplest
    and cheapest routing); `k>1` blends the top-`k` experts, as in Shazeer 2017 and
    Mixtral.

    Experts must map (..., hidden_size) to (..., out_size); the router looks at the
    last dim. Accepts any leading shape -- `(batch, hidden)` or `(batch, seq,
    hidden)` -- by flattening every leading dim into one routing axis and restoring
    it afterwards.

    Args:
        experts: modules mapping (..., hidden_size) -> (..., out_size).
        hidden_size: input dimensionality (also the router input size).
        out_size: output dimensionality of each expert.
        k: number of experts activated per example (default 1, Switch-style). Must
            be between 1 and `len(experts)`.
    """

    def __init__(
        self,
        experts: Sequence[nn.Module],
        hidden_size: int,
        out_size: int,
        k: int = 1,
    ) -> None:
        super().__init__()
        experts = list(experts)
        if not (1 <= k <= len(experts)):
            raise VectorMeshError(
                f"k={k} is not a valid number of experts to route to out of "
                f"{len(experts)} experts.",
                hint="k selects how many of the N experts each example activates; "
                "it has to be at least 1 and cannot exceed N.",
                fix=f"pass 1 <= k <= {len(experts)}.",
            )
        self.experts = nn.ModuleList(experts)
        self.router = nn.Linear(hidden_size, len(experts))
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_experts = len(experts)
        self.k = k

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "... {self.hidden_size}"]
    ) -> Float[Tensor, "... {self.out_size}"]:
        leading_shape = tensors.shape[:-1]
        flat = tensors.reshape(-1, self.hidden_size)  # (n, hidden) -- one routing axis
        n = flat.shape[0]

        logits = self.router(flat)  # (n, num_experts)
        topk_logits, topk_idx = logits.topk(self.k, dim=-1)  # (n, k) each
        # renormalise over just the top-k, not the full softmax over all N experts --
        # that is what keeps the weights a distribution while only touching k logits.
        topk_weights = F.softmax(topk_logits, dim=-1)

        # Flatten the (n, k) routing decisions into n*k independent (example, expert,
        # weight) assignments, one per example-slot pair, then sort by expert id so
        # every expert's assignments become one contiguous slice. This is what keeps
        # the loop below cheap as num_experts grows: the per-expert cost is a slice
        # (free) plus one matmul over exactly its assigned rows, not an O(num_experts)
        # scan of the whole batch looking for that expert's hits every iteration.
        assigned_expert = topk_idx.reshape(-1)  # (n*k,)
        assigned_weight = topk_weights.reshape(-1)  # (n*k,)
        assigned_example = (
            torch.arange(n, device=tensors.device)
            .unsqueeze(1)
            .expand(-1, self.k)
            .reshape(-1)
        )  # (n*k,) -- which original row each assignment belongs to

        sort_order = assigned_expert.argsort()
        sorted_example = assigned_example[sort_order]
        sorted_weight = assigned_weight[sort_order]

        counts = torch.bincount(assigned_expert, minlength=self.num_experts)
        offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])

        output = flat.new_zeros(n, self.out_size)
        # only experts that actually received an assignment this batch ever run --
        # an expert with count 0 never appears in this loop at all.
        for expert_id in counts.nonzero(as_tuple=True)[0].tolist():
            start, end = int(offsets[expert_id]), int(offsets[expert_id + 1])
            example_idx = sorted_example[start:end]
            weight = sorted_weight[start:end].unsqueeze(-1)
            # gather: only the examples routed to this expert ever enter it
            gathered = self.experts[expert_id](flat[example_idx])
            # scatter-add: accumulate weighted expert output back at each example's row
            output.index_add_(0, example_idx, gathered * weight)

        return output.reshape(*leading_shape, self.out_size)
