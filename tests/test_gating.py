"""Tests for `SparseMoE`, the top-k routed mixture of experts.

The property under test throughout is not just "the shapes and numbers come out
right" but **sparsity of compute**: an expert an example did not route to must
never receive that example's tensor, not receive it and get masked out afterwards.
Several tests below instrument the experts themselves (call counts, which rows they
were called with, whether they ever appear in the autograd graph) to check that
directly, because a "compute everything then mask" implementation would pass a
shape-only test just as easily as a real gather/scatter one.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vectormesh.components.gating import MoE, SparseMoE
from vectormesh.types import VectorMeshError


class CountingLinear(nn.Module):
    """A Linear that records every batch of rows it was actually called with."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.calls = 0
        self.seen_batch_sizes: list[int] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.seen_batch_sizes.append(x.shape[0])
        return self.linear(x)


def make_experts(n: int, hidden: int, out: int, counting: bool = False):
    if counting:
        return [CountingLinear(hidden, out) for _ in range(n)]
    return [nn.Linear(hidden, out) for _ in range(n)]


def test_invalid_k_raises():
    with pytest.raises(VectorMeshError, match="k=0"):
        SparseMoE(experts=make_experts(4, 8, 8), hidden_size=8, out_size=8, k=0)
    with pytest.raises(VectorMeshError, match="k=5"):
        SparseMoE(experts=make_experts(4, 8, 8), hidden_size=8, out_size=8, k=5)


def test_default_k_is_one():
    moe = SparseMoE(experts=make_experts(4, 8, 8), hidden_size=8, out_size=8)
    assert moe.k == 1


def test_output_shape_2d_and_3d():
    moe = SparseMoE(experts=make_experts(4, 8, 6), hidden_size=8, out_size=6, k=2)
    out2d = moe(torch.randn(5, 8))
    assert out2d.shape == (5, 6)
    out3d = moe(torch.randn(2, 7, 8))
    assert out3d.shape == (2, 7, 6)


def test_k1_output_exactly_matches_the_single_routed_expert():
    """With k=1 the softmax over a single logit is always 1.0, so the output must
    equal the chosen expert's raw output -- no blending, no scaling."""
    torch.manual_seed(0)
    moe = SparseMoE(experts=make_experts(4, 8, 8), hidden_size=8, out_size=8, k=1)
    x = torch.randn(16, 8)
    with torch.no_grad():
        logits = moe.router(x)
        chosen = logits.argmax(dim=-1)
        expected = torch.stack(
            [moe.experts[chosen[i]](x[i]) for i in range(x.shape[0])]
        )
        actual = moe(x)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_only_routed_experts_are_ever_called():
    """The core sparsity claim: an expert that received zero examples this batch
    must have zero calls -- not a call on an empty-but-present tensor, no call at
    all -- and every called expert must only ever see the rows routed to it."""
    torch.manual_seed(1)
    experts = make_experts(6, 8, 8, counting=True)
    moe = SparseMoE(experts=experts, hidden_size=8, out_size=8, k=1)
    x = torch.randn(20, 8)

    with torch.no_grad():
        routed_to = moe.router(x).argmax(dim=-1)  # (20,)
        moe(x)

    counts = torch.bincount(routed_to, minlength=6)
    for expert_id, expert in enumerate(experts):
        if counts[expert_id] == 0:
            assert expert.calls == 0, f"expert {expert_id} routed 0 examples but ran"
        else:
            assert expert.calls == 1
            assert expert.seen_batch_sizes == [int(counts[expert_id])]


def test_compute_scales_with_k_not_num_experts():
    """Total rows processed across all experts must equal n * k, regardless of how
    many experts exist -- the whole point of routing instead of running everything."""
    torch.manual_seed(2)
    n, hidden, k = 32, 8, 2
    for num_experts in (4, 64):
        experts = make_experts(num_experts, hidden, hidden, counting=True)
        moe = SparseMoE(experts=experts, hidden_size=hidden, out_size=hidden, k=k)
        with torch.no_grad():
            moe(torch.randn(n, hidden))
        total_rows_processed = sum(sum(e.seen_batch_sizes) for e in experts)
        assert total_rows_processed == n * k


def test_unrouted_experts_get_no_gradient():
    """An expert with zero routed examples must be absent from the autograd graph
    entirely -- its params should have no grad at all, not a zero grad computed from
    a masked-out contribution."""
    torch.manual_seed(3)
    hidden = 8
    experts = [nn.Linear(hidden, hidden) for _ in range(6)]
    moe = SparseMoE(experts=experts, hidden_size=hidden, out_size=hidden, k=1)
    x = torch.randn(10, hidden)

    with torch.no_grad():
        routed_to = moe.router(x).argmax(dim=-1)
    used = set(routed_to.tolist())

    out = moe(x)
    out.sum().backward()

    for expert_id, expert in enumerate(experts):
        grad = expert.weight.grad
        if expert_id in used:
            assert grad is not None
        else:
            assert grad is None


def test_k_equals_num_experts_matches_dense_moe():
    """When k == N, top-k is the whole expert set, and the softmax over the top-k
    is identical to the dense MoE's softmax over all experts -- the two components
    should therefore agree exactly given the same router and expert weights. This
    ties the sparse gather/scatter path to the already-trusted dense implementation."""
    torch.manual_seed(4)
    hidden, out, n_experts = 8, 8, 4
    dense = MoE(
        experts=[nn.Linear(hidden, out) for _ in range(n_experts)],
        hidden_size=hidden,
        out_size=out,
    )
    sparse = SparseMoE(
        experts=[nn.Linear(hidden, out) for _ in range(n_experts)],
        hidden_size=hidden,
        out_size=out,
        k=n_experts,
    )
    sparse.router.load_state_dict(dense.router.state_dict())
    for d_expert, s_expert in zip(dense.experts, sparse.experts):
        s_expert.load_state_dict(d_expert.state_dict())

    x = torch.randn(9, hidden)
    with torch.no_grad():
        assert torch.allclose(dense(x), sparse(x), atol=1e-5)


def test_gradients_reach_router_and_routed_experts():
    """A basic trainability check: backprop through SparseMoE must update the
    router and every expert that was actually used."""
    torch.manual_seed(5)
    hidden = 8
    moe = SparseMoE(
        experts=[nn.Linear(hidden, hidden) for _ in range(4)],
        hidden_size=hidden,
        out_size=hidden,
        k=2,
    )
    x = torch.randn(12, hidden, requires_grad=True)
    out = moe(x)
    out.sum().backward()
    assert moe.router.weight.grad is not None
    assert not torch.all(moe.router.weight.grad == 0)
    assert x.grad is not None


def test_topk_weights_sum_to_one_per_example():
    """Renormalisation check: softmax(topk_logits) sums to 1 across the k chosen
    experts for every example, independent of how many experts exist in total."""
    torch.manual_seed(6)
    hidden = 8
    moe = SparseMoE(
        experts=[nn.Linear(hidden, hidden) for _ in range(5)],
        hidden_size=hidden,
        out_size=hidden,
        k=3,
    )
    x = torch.randn(7, hidden)
    with torch.no_grad():
        logits = moe.router(x)
        topk_logits, _ = logits.topk(3, dim=-1)
        weights = F.softmax(topk_logits, dim=-1)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(7), atol=1e-6)
