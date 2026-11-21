"""Gating components: a learned weight decides how much of a transform gets through."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from torch import Tensor, nn

from vectormesh.components.contracts import RankPolymorphicComponent
from vectormesh.types import VectorMeshError, call_masked


class Gate(RankPolymorphicComponent):
    """Simple gating: output = sigmoid(W·x) * x, on the last axis at any rank."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.project = nn.Linear(hidden_size, hidden_size)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "*lead {self.hidden_size}"]
    ) -> Float[Tensor, "*lead {self.hidden_size}"]:
        return F.sigmoid(self.project(tensors)) * tensors


class Highway(RankPolymorphicComponent):
    """Highway network: G * T(x) + (1-G) * x, on the last axis at any rank.

    The transform must keep the width, ``(*lead, hidden_size) -> (*lead, hidden_size)``,
    since its output is blended with the input. On a `Padded` batch the transform gets
    the mask too.
    """

    def __init__(self, transform: nn.Module, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.transform = transform
        self.project = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        tensors: Float[Tensor, "*lead {self.hidden_size}"],
        mask: Bool[Tensor, "*mlead"] | None = None,
    ) -> Float[Tensor, "*lead {self.hidden_size}"]:
        # pre-norm (instead of post-norm) improves stability
        tensors = self.norm(tensors)
        gate = F.sigmoid(self.project(tensors))
        transformed = call_masked(self.transform, tensors, mask)
        return gate * transformed + (1 - gate) * tensors


class MoE(RankPolymorphicComponent):
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

    Experts must map (..., hidden_size) to (..., out_size). On a `Padded` batch an
    expert that is a component gets the mask too, so `TransformerBlock` experts do
    not attend to padding.

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
        self,
        tensors: Float[Tensor, "*lead {self.hidden_size}"],
        mask: Bool[Tensor, "*mlead"] | None = None,
    ) -> Float[Tensor, "*lead {self.out_size}"]:
        # Per-position gates over experts; works for 2D and 3D alike.
        gates = F.softmax(self.router(tensors), dim=-1)  # (..., num_experts)
        # stack experts on dim -2: (..., num_experts, out_size)
        expert_outputs = torch.stack(
            [call_masked(e, tensors, mask) for e in self.experts], dim=-2
        )
        # sum over experts weighted by gates: (..., out_size)
        return (gates.unsqueeze(-1) * expert_outputs).sum(dim=-2)


class SparseMoE(RankPolymorphicComponent):
    """Sparse (top-k) mixture of experts: each example activates only its top-k
    experts, and only those experts run on it.

    `MoE` is the Jacobs & Jordan (1991) formulation: every expert runs on every
    input and the outputs are softmax-blended, so compute per example is `O(N)` in
    the number of experts. `SparseMoE` is the Shazeer et al. (2017, *Outrageously
    Large Neural Networks*) / Switch Transformer (Fedus et al., 2021) formulation:
    the router picks the top-`k` experts per example, each example is *gathered*
    into only those experts' input batches, and the outputs are *scattered* back.
    An expert an example did not route to never sees that example's tensor.

    Each chosen expert's output is weighted by its probability under the softmax
    over all `N` router logits, as in the Switch Transformer. That probability
    depends on the router weights also at `k=1`, so the router learns. With
    ``normalize=True`` the `k` chosen probabilities are rescaled to sum to 1, as in
    Shazeer et al. (2017) and Mixtral; at `k=1` that weight is always 1 and the
    router would get no gradient, so ``normalize=True`` needs ``k >= 2``.

    After every forward pass `aux_loss` holds the Switch Transformer
    load-balancing loss, ``N * sum_i f_i * P_i``: `f_i` is the fraction of routing
    assignments that went to expert `i`, `P_i` the mean router probability of
    expert `i`. It is 1 when routing is uniform and grows as the router
    concentrates on few experts. Nothing adds it to the training loss on its own;
    `load_balancing_loss(model)` sums it over every `SparseMoE` in a model, for a
    custom `Step`.

    Experts must map (..., hidden_size) to (..., out_size); the router looks at the
    last dim. Accepts any leading shape -- `(batch, hidden)` or `(batch, seq,
    hidden)` -- by flattening every leading dim into one routing axis and restoring
    it afterwards. With a `mask` (True for real positions, as a `Padded` batch
    carries it), padded positions are not routed: no expert runs on them, they
    count towards neither `f_i` nor `P_i`, and their output is zero.

    Args:
        experts: modules mapping (..., hidden_size) -> (..., out_size).
        hidden_size: input dimensionality (also the router input size).
        out_size: output dimensionality of each expert.
        k: number of experts activated per example (default 1, Switch-style). Must
            be between 1 and `len(experts)`.
        normalize: rescale the `k` chosen probabilities to sum to 1. Needs `k >= 2`.
    """

    def __init__(
        self,
        experts: Sequence[nn.Module],
        hidden_size: int,
        out_size: int,
        k: int = 1,
        normalize: bool = False,
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
        if normalize and k == 1:
            raise VectorMeshError(
                "normalize=True with k=1 gives every example the weight 1, so the "
                "router would receive no gradient and never learn.",
                hint="normalize rescales the k chosen router probabilities to sum "
                "to 1; with one chosen expert that sum is the probability itself.",
                fix="pass normalize=False (Switch-style) for k=1, or k >= 2.",
            )
        self.experts = nn.ModuleList(experts)
        self.router = nn.Linear(hidden_size, len(experts))
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_experts = len(experts)
        self.k = k
        self.normalize = normalize
        self.aux_loss: Tensor = torch.zeros(())

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        tensors: Float[Tensor, "*lead {self.hidden_size}"],
        mask: Bool[Tensor, "*lead"] | None = None,
    ) -> Float[Tensor, "*lead {self.out_size}"]:
        leading_shape = tensors.shape[:-1]
        flat = tensors.reshape(-1, self.hidden_size)  # (n, hidden): one routing axis
        output = flat.new_zeros(flat.shape[0], self.out_size)
        if mask is None:
            real = None
            routed = flat
        else:
            real = mask.reshape(-1).nonzero(as_tuple=True)[0]
            routed = flat[real]
        n = routed.shape[0]
        if n == 0:
            self.aux_loss = flat.new_zeros(())
            return output.reshape(*leading_shape, self.out_size)

        probs = F.softmax(self.router(routed), dim=-1)  # (n, num_experts)
        topk_probs, topk_idx = probs.topk(self.k, dim=-1)  # (n, k) each
        if self.normalize:
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # n*k (example, expert, weight) assignments, sorted by expert so that every
        # expert's assignments are one contiguous slice.
        assigned_expert = topk_idx.reshape(-1)  # (n*k,)
        sort_order = assigned_expert.argsort(stable=True)
        sorted_example = sort_order // self.k  # the routed row each assignment reads
        sorted_weight = topk_probs.reshape(-1)[sort_order].unsqueeze(-1)
        counts = torch.bincount(assigned_expert, minlength=self.num_experts)

        # one host sync for all experts; an expert with no assignment never runs
        sizes = counts.tolist()
        mixed = routed.new_zeros(n, self.out_size)
        for expert, example_idx, weight in zip(
            self.experts, sorted_example.split(sizes), sorted_weight.split(sizes)
        ):
            if example_idx.numel():
                # gather the rows routed to this expert, scatter-add its weighted output
                mixed.index_add_(0, example_idx, expert(routed[example_idx]) * weight)

        fraction = counts.to(probs.dtype) / (n * self.k)
        self.aux_loss = self.num_experts * (fraction * probs.mean(dim=0)).sum()

        if real is None:
            output = mixed
        else:
            output = output.index_copy(0, real, mixed)
        return output.reshape(*leading_shape, self.out_size)


def load_balancing_loss(model: nn.Module) -> Tensor:
    """The sum of `aux_loss` over every `SparseMoE` in `model`, from its last
    forward pass: the Switch Transformer load-balancing loss, for a custom `Step`.

    >>> step = lambda model, x, y: loss_fn(model(x), y) + 0.01 * load_balancing_loss(model)

    Fedus et al. (2021) use a weight of 0.01. A model without a `SparseMoE` gives 0.
    """
    losses = [m.aux_loss for m in model.modules() if isinstance(m, SparseMoE)]
    if not losses:
        return torch.zeros(())
    return torch.stack(losses).sum()
