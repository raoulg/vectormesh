"""Tests for metrics."""

import pytest
import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from vectormesh.components.metrics import MASE, Accuracy


def _logits():
    # 4 samples, 3 classes; argmax -> predicted classes [2, 0, 1, 2]
    return torch.tensor(
        [
            [0.1, 0.2, 0.7],
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.3, 0.3, 0.4],
        ]
    )


def test_accuracy_integer_targets():
    yhat = _logits()
    y = torch.tensor([2, 0, 1, 0])  # last one wrong -> 3/4
    assert Accuracy()(y, yhat) == 0.75


def test_accuracy_one_hot_targets():
    yhat = _logits()
    # same labels as above but one-hot (batch, num_classes), like OneHot + Collate
    y = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    assert Accuracy()(y, yhat) == 0.75


def test_accuracy_one_hot_all_correct():
    yhat = _logits()
    y = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert Accuracy()(y, yhat) == 1.0


# --- MASE -------------------------------------------------------------------
W, H = 8, 2


def _windows(n: int = 32, ndim: int = 2):
    """(x, y) where the naive forecast is off by exactly 1.0, so the scale is 1.0."""
    torch.manual_seed(0)
    x = torch.randn(n, W)
    y = x[:, -H:] + 1.0
    return (x.unsqueeze(-1) if ndim == 3 else x), y


def test_mase_from_dataloader_2d():
    x, y = _windows()
    mase = MASE(DataLoader(TensorDataset(x, y), batch_size=8), horizon=H)
    assert torch.isclose(mase.scale, torch.tensor(1.0), atol=1e-6)
    # a forecast twice as wrong as the naive one scores 2.0
    assert abs(mase(y, y - 2.0) - 2.0) < 1e-5


def test_mase_accepts_a_feature_axis():
    """(batch, time, 1) gives the same scale as (batch, time)."""
    x3, y = _windows(ndim=3)
    x2, _ = _windows(ndim=2)
    scale3 = MASE(DataLoader(TensorDataset(x3, y), batch_size=8), horizon=H).scale
    scale2 = MASE(DataLoader(TensorDataset(x2, y), batch_size=8), horizon=H).scale
    assert torch.isclose(scale3, scale2)


def test_mase_consumes_every_batch():
    """Regression: the old loop re-read the first batch len(train) times."""
    x, y = _windows(n=16)
    y[8:] += 2.0  # the second half is much harder than the first
    both = MASE(DataLoader(TensorDataset(x, y), batch_size=8), horizon=H).scale
    first = MASE(DataLoader(TensorDataset(x[:8], y[:8]), batch_size=8), horizon=H).scale
    assert both > first


def test_mase_streamer_still_works_but_warns():
    class LegacyStreamer:
        def __init__(self, batches):
            self.batches = batches

        def __len__(self):
            return len(self.batches)

        def stream(self):
            return iter(self.batches)

    x, y = _windows(n=16)
    streamer = LegacyStreamer([(x[:8], y[:8]), (x[8:], y[8:])])

    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        mase = MASE(streamer, horizon=H)
    finally:
        logger.remove(sink)

    assert torch.isclose(mase.scale, torch.tensor(1.0), atol=1e-6)
    assert any("deprecated" in m for m in messages), messages


def test_mase_rejects_no_batches():
    empty = TensorDataset(torch.empty(0, W), torch.empty(0, H))
    with pytest.raises(ValueError, match="at least one batch"):
        MASE(DataLoader(empty, batch_size=8), horizon=H)


def test_mase_rejects_a_zero_scale():
    x = torch.randn(8, W)
    perfect = TensorDataset(x, x[:, -H:])  # naive is exact -> every score infinite
    with pytest.raises(ValueError, match="scale is zero"):
        MASE(DataLoader(perfect, batch_size=8), horizon=H)
