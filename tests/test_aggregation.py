"""Tests for the aggregators in `components/aggregation.py`."""

import torch

from vectormesh.components import RNNAggregator

DIM = 4


def _rnn() -> RNNAggregator:
    torch.manual_seed(0)
    return RNNAggregator(DIM).eval()


def _chunks(n: int) -> torch.Tensor:
    torch.manual_seed(1)
    return (
        torch.rand(n, DIM) + 0.1
    )  # strictly non-zero, so no real chunk reads as padding


def test_rnn_ignores_padding_at_the_end():
    rnn, real = _rnn(), _chunks(3)
    padded = torch.cat([real, torch.zeros(2, DIM)])
    expected = rnn(real.unsqueeze(0))
    assert torch.allclose(rnn(padded.unsqueeze(0)), expected, atol=1e-6)


def test_rnn_ignores_gaps_in_the_middle():
    """Concatenating two padded sequences leaves padding between them; the real chunks
    keep their order."""
    rnn, real = _rnn(), _chunks(4)
    gapped = torch.cat([real[:2], torch.zeros(2, DIM), real[2:], torch.zeros(1, DIM)])
    expected = rnn(real.unsqueeze(0))
    assert torch.allclose(rnn(gapped.unsqueeze(0)), expected, atol=1e-6)


def test_rnn_keeps_items_apart_in_a_batch():
    rnn, real = _rnn(), _chunks(5)
    short = torch.cat([real[:2], torch.zeros(3, DIM)])
    batch = torch.stack([real, short])
    out = rnn(batch)
    assert torch.allclose(out[0], rnn(real.unsqueeze(0))[0], atol=1e-6)
    assert torch.allclose(out[1], rnn(real[:2].unsqueeze(0))[0], atol=1e-6)


def test_rnn_is_order_sensitive():
    rnn, real = _rnn(), _chunks(3)
    assert not torch.allclose(rnn(real.unsqueeze(0)), rnn(real.flip(0).unsqueeze(0)))


def test_rnn_takes_an_all_padding_item():
    out = _rnn()(torch.zeros(2, 3, DIM))
    assert out.shape == (2, DIM)
    assert torch.isfinite(out).all()
