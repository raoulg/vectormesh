"""Tests for the aggregators in `components/aggregation.py`."""

import torch

from vectormesh.components import LastAggregator, RNNAggregator
from vectormesh.types import Padded

DIM = 4


def _rnn() -> RNNAggregator:
    torch.manual_seed(0)
    return RNNAggregator(DIM).eval()


def _chunks(n: int) -> torch.Tensor:
    torch.manual_seed(1)
    return (
        torch.rand(n, DIM) + 0.1
    )  # strictly non-zero, so no real chunk reads as padding


def _padded(*items: torch.Tensor) -> Padded:
    """Items of shape ``(seq, DIM)`` with a mask that is False where an item is zero."""
    values = torch.stack(items)
    return Padded(values, values.abs().sum(-1) > 0)


def test_rnn_ignores_padding_at_the_end():
    rnn, real = _rnn(), _chunks(3)
    padded = torch.cat([real, torch.full((2, DIM), 7.0)])
    mask = torch.tensor([[True, True, True, False, False]])
    expected = rnn(real.unsqueeze(0))
    assert torch.allclose(rnn(Padded(padded.unsqueeze(0), mask)), expected, atol=1e-6)


def test_rnn_ignores_gaps_in_the_middle():
    """Concatenating two padded sequences leaves padding between them; the real chunks
    keep their order."""
    rnn, real = _rnn(), _chunks(4)
    gapped = torch.cat([real[:2], torch.zeros(2, DIM), real[2:], torch.zeros(1, DIM)])
    expected = rnn(real.unsqueeze(0))
    assert torch.allclose(rnn(_padded(gapped)), expected, atol=1e-6)


def test_rnn_keeps_items_apart_in_a_batch():
    rnn, real = _rnn(), _chunks(5)
    short = torch.cat([real[:2], torch.zeros(3, DIM)])
    out = rnn(_padded(real, short))
    assert torch.allclose(out[0], rnn(real.unsqueeze(0))[0], atol=1e-6)
    assert torch.allclose(out[1], rnn(real[:2].unsqueeze(0))[0], atol=1e-6)


def test_rnn_is_order_sensitive():
    rnn, real = _rnn(), _chunks(3)
    assert not torch.allclose(rnn(real.unsqueeze(0)), rnn(real.flip(0).unsqueeze(0)))


def test_rnn_takes_an_all_padding_item():
    out = _rnn()(Padded(torch.zeros(2, 3, DIM), torch.zeros(2, 3, dtype=torch.bool)))
    assert out.shape == (2, DIM)
    assert torch.isfinite(out).all()


def test_last_returns_the_last_step():
    x = torch.randn(2, 5, DIM)
    assert torch.equal(LastAggregator()(x), x[:, -1])


def test_last_reads_padding_as_a_step_without_a_mask():
    real = _chunks(3).unsqueeze(0)
    padded = torch.cat([real, torch.zeros(1, 2, DIM)], dim=1)
    assert torch.equal(LastAggregator()(padded), torch.zeros(1, DIM))


def test_last_takes_the_last_real_step_with_a_mask():
    real = _chunks(3)
    batch = _padded(torch.cat([real, torch.zeros(2, DIM)]), torch.cat([real, real[:2]]))
    out = LastAggregator()(batch)
    assert torch.equal(out[0], real[2])
    assert torch.equal(out[1], real[1])
