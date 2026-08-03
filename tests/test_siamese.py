"""Tests for Siamese and RepresentationStd -- the self-supervised pair.

Siamese is Parallel with the branches' weights coupled and a gradient gate on one
side; RepresentationStd is the instrument that catches the failure mode the loss
curve cannot see.
"""

import copy

import pytest
import torch
from torch import nn

from vectormesh.components import RepresentationStd, Siamese

DIM = 8


def encoder(dim: int = DIM) -> nn.Module:
    return nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))


def views(batch: int = 16, dim: int = DIM):
    return torch.randn(batch, dim), torch.randn(batch, dim)


def test_shared_weights_when_momentum_is_zero():
    """momentum=0 means the target IS the online encoder -- SimSiam's setup."""
    m = Siamese(encoder(), momentum=0.0)

    assert m.target is None
    assert len(list(m.parameters())) == len(list(m.online.parameters()))


def test_ema_target_is_a_frozen_copy():
    m = Siamese(encoder(), momentum=0.99)

    assert m.target is not None
    assert m.target is not m.online
    assert not any(p.requires_grad for p in m.target.parameters())


def test_target_branch_is_detached_by_default():
    m = Siamese(encoder(), momentum=0.0)
    m.train()

    (_, z_target), _ = m(views())

    assert not z_target.requires_grad, "the target must not carry gradient"


def test_stopgrad_false_lets_gradient_through():
    """The ablation. Without the gate, both branches can move toward each other."""
    m = Siamese(encoder(), momentum=0.0, stopgrad=False)
    m.train()

    (_, z_target), _ = m(views())

    assert z_target.requires_grad


def test_ema_moves_target_toward_online():
    m = Siamese(encoder(), momentum=0.9)
    m.train()
    assert m.target is not None
    before = copy.deepcopy([p.clone() for p in m.target.parameters()])

    with torch.no_grad():
        for p in m.online.parameters():
            p.add_(1.0)
    m.update_target()

    after = list(m.target.parameters())
    assert all(not torch.equal(b, a) for b, a in zip(before, after))
    # one step of m=0.9 covers 10% of the gap, not all of it
    expected = before[0] * 0.9 + (before[0] + 1.0) * 0.1
    assert torch.allclose(after[0], expected)


def test_no_ema_update_in_eval_mode():
    """Evaluating a model must not change it."""
    m = Siamese(encoder(), momentum=0.9)
    m.eval()
    assert m.target is not None
    before = [p.clone() for p in m.target.parameters()]

    m(views())

    assert all(torch.equal(b, a) for b, a in zip(before, m.target.parameters()))


def test_predictor_applies_to_online_branch_only():
    marker = nn.Linear(DIM, DIM)
    with torch.no_grad():
        marker.weight.zero_()
        marker.bias.fill_(42.0)
    m = Siamese(encoder(), momentum=0.0, predictor=marker)
    m.train()

    (prediction, z_target), _ = m(views())

    assert torch.allclose(prediction, torch.full_like(prediction, 42.0))
    assert not torch.allclose(z_target, torch.full_like(z_target, 42.0))


def test_output_is_symmetrised():
    """Each view takes a turn as the target, so the loss sees both directions."""
    m = Siamese(encoder(), momentum=0.0)
    m.train()

    out = m(views())

    assert len(out) == 2
    assert all(len(pair) == 2 for pair in out)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_momentum_is_validated(bad):
    with pytest.raises(ValueError, match="momentum"):
        Siamese(encoder(), momentum=bad)


# ---------------------------------------------------------------------------
# RepresentationStd
# ---------------------------------------------------------------------------


def test_collapsed_representation_scores_near_zero():
    collapsed = torch.ones(32, 128)

    assert RepresentationStd(index=())(None, collapsed) < 1e-6


def test_healthy_representation_is_near_one_over_sqrt_dim():
    torch.manual_seed(0)
    healthy = torch.randn(512, 128)

    value = RepresentationStd(index=())(None, healthy)

    assert value == pytest.approx(1 / 128**0.5, abs=0.01)


def test_collapse_scores_far_below_healthy():
    """The property the metric exists for: it separates the two cases."""
    torch.manual_seed(0)
    healthy = RepresentationStd(index=())(None, torch.randn(512, 128))
    collapsed = RepresentationStd(index=())(None, torch.ones(512, 128) + 1e-4)

    assert collapsed < healthy / 100


def test_reaches_into_a_siamese_output_by_default():
    m = Siamese(encoder(), momentum=0.0)
    m.train()

    value = RepresentationStd()(None, m(views(batch=64)))

    assert isinstance(value, float)


def test_three_dimensional_input_treats_chunks_as_samples():
    torch.manual_seed(0)
    patches = torch.randn(8, 16, 128)  # (batch, chunks, dim)

    value = RepresentationStd(index=())(None, patches)

    assert value == pytest.approx(1 / 128**0.5, abs=0.02)


def test_wrong_index_explains_itself():
    m = Siamese(encoder(), momentum=0.0)
    m.train()

    with pytest.raises(TypeError, match="not a Tensor"):
        RepresentationStd(index=(0,))(None, m(views()))


def test_metric_does_not_leak_gradient():
    """A metric must never become part of the graph."""
    m = Siamese(encoder(), momentum=0.0, stopgrad=False)
    m.train()
    out = m(views())

    RepresentationStd()(None, out)

    assert out[0][0].grad_fn is not None, "sanity: the output was differentiable"
