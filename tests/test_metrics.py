"""Tests for metrics."""

import torch

from vectormesh.components.metrics import Accuracy


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
