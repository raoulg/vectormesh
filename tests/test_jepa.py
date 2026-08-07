"""Tests for JEPA -- Siamese's sibling for masked-position prediction.

JEPA shares Siamese's EMA-target-lives-in-forward trick, but the predictor needs
*where* to predict (context/target indices), not just *what*, so it is a separate
component rather than a Siamese configuration.
"""

import copy

import pytest
import torch
from torch import nn

from vectormesh.components import JEPA, RepresentationStd

DIM = 8
SEQ = 6


class ToyEncoder(nn.Module):
    """Position-aware encoder over a fixed-length sequence, with a `keep` kwarg
    selecting which positions to encode -- the contract JEPA's online branch needs."""

    def __init__(self, dim: int = DIM, seq: int = SEQ):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, seq, dim))
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, keep: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.proj(x + self.pos)
        return h if keep is None else h[:, keep, :]


class ToyPredictor(nn.Module):
    """Pools the context and broadcasts a prediction to every target position --
    enough to exercise the (context, ctx_idx, tgt_idx) -> prediction contract."""

    def __init__(self, dim: int = DIM):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, context, ctx_idx, tgt_idx):
        pooled = context.mean(dim=1, keepdim=True)
        return self.lin(pooled).expand(-1, len(tgt_idx), -1)


def inputs(batch: int = 4, dim: int = DIM, seq: int = SEQ):
    x = torch.randn(batch, seq, dim)
    ctx_idx = torch.tensor([0, 1, 2, 3])
    tgt_idx = torch.tensor([4, 5])
    return x, ctx_idx, tgt_idx


def test_shared_weights_when_momentum_is_zero():
    m = JEPA(ToyEncoder(), ToyPredictor(), momentum=0.0)

    assert m.target is None
    assert len(list(m.parameters())) == len(
        list(m.online.parameters()) + list(m.predictor.parameters())
    )


def test_ema_target_is_a_frozen_copy():
    m = JEPA(ToyEncoder(), ToyPredictor(), momentum=0.99)

    assert m.target is not None
    assert m.target is not m.online
    assert not any(p.requires_grad for p in m.target.parameters())


def test_target_branch_is_detached_by_default():
    m = JEPA(ToyEncoder(), ToyPredictor(), momentum=0.99)
    m.train()

    _, target = m(inputs())

    assert not target.requires_grad, "the target must not carry gradient"


def test_stopgrad_false_lets_gradient_through():
    """momentum=0.0 so the target IS the online encoder -- its parameters require
    grad, so this isolates what stopgrad itself controls."""
    m = JEPA(ToyEncoder(), ToyPredictor(), momentum=0.0, stopgrad=False)
    m.train()

    _, target = m(inputs())

    assert target.requires_grad


def test_ema_moves_target_toward_online():
    m = JEPA(ToyEncoder(), ToyPredictor(), momentum=0.9)
    m.train()
    assert m.target is not None
    before = copy.deepcopy([p.clone() for p in m.target.parameters()])

    with torch.no_grad():
        for p in m.online.parameters():
            p.add_(1.0)
    m.update_target()

    after = list(m.target.parameters())
    assert all(not torch.equal(b, a) for b, a in zip(before, after))
    expected = before[0] * 0.9 + (before[0] + 1.0) * 0.1
    assert torch.allclose(after[0], expected)


def test_no_ema_update_in_eval_mode():
    m = JEPA(ToyEncoder(), ToyPredictor(), momentum=0.9)
    m.eval()
    assert m.target is not None
    before = [p.clone() for p in m.target.parameters()]

    m(inputs())

    assert all(torch.equal(b, a) for b, a in zip(before, m.target.parameters()))


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_momentum_is_validated(bad):
    with pytest.raises(ValueError, match="momentum"):
        JEPA(ToyEncoder(), ToyPredictor(), momentum=bad)


def test_online_only_encodes_context_positions(monkeypatch):
    """The online branch must never see the target positions -- that is the
    entire point of the mask. Patched at the class level (not the instance) so
    `copy.deepcopy`-ing the target inside `JEPA.__init__` does not drag the spy
    along with it; calls are attributed back to the instance that made them."""
    calls = []
    real_forward = ToyEncoder.forward

    def spy(self, x, keep=None):
        calls.append((self, keep))
        return real_forward(self, x, keep=keep)

    monkeypatch.setattr(ToyEncoder, "forward", spy)

    encoder = ToyEncoder()
    m = JEPA(encoder, ToyPredictor(), momentum=0.99)
    m.train()

    x, ctx_idx, tgt_idx = inputs()
    m((x, ctx_idx, tgt_idx))

    online_calls = [keep for instance, keep in calls if instance is m.online]
    assert len(online_calls) == 1
    assert torch.equal(online_calls[0], ctx_idx)


def test_output_shapes_match_target_positions():
    m = JEPA(ToyEncoder(), ToyPredictor(), momentum=0.99)
    m.train()
    x, ctx_idx, tgt_idx = inputs(batch=4)

    prediction, target = m((x, ctx_idx, tgt_idx))

    assert prediction.shape == (4, len(tgt_idx), DIM)
    assert target.shape == (4, len(tgt_idx), DIM)


def test_target_is_the_target_positions_of_the_full_encoding():
    """With stopgrad and momentum=0.0 (target IS online), the target output must
    equal encoding the full sequence and slicing at tgt_idx."""
    m = JEPA(ToyEncoder(), ToyPredictor(), momentum=0.0)
    m.eval()  # eval mode: no dropout/BN drift, so a second forward pass matches
    x, ctx_idx, tgt_idx = inputs()

    _, target = m((x, ctx_idx, tgt_idx))
    with torch.no_grad():
        expected = m.online(x)[:, tgt_idx, :]

    assert torch.allclose(target, expected)


def test_reaches_into_a_jepa_output_with_index_zero():
    """JEPA returns a flat (prediction, target) pair, not Siamese's nested one --
    RepresentationStd needs index=(0,), not the Siamese default of (0, 0)."""
    m = JEPA(ToyEncoder(), ToyPredictor(), momentum=0.0)
    m.train()

    value = RepresentationStd(index=(0,))(None, m(inputs()))

    assert isinstance(value, float)
