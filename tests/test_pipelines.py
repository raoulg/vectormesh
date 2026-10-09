"""Serial and Parallel are components made of components."""

import torch
from torch import nn

from vectormesh.components import Lift, NeuralNet, Parallel, Serial
from vectormesh.types import BaseComponent


def test_a_pipeline_is_a_component_made_of_components():
    inner = Serial([NeuralNet(hidden_size=8, out_size=4)])
    outer = Parallel([inner, Serial([])])
    assert isinstance(inner, BaseComponent)
    assert isinstance(outer, BaseComponent)
    assert all(isinstance(branch, BaseComponent) for branch in outer.branches)


def test_a_plain_module_is_lifted_into_a_component():
    linear = nn.Linear(8, 4)
    pipeline = Serial([linear])
    assert isinstance(pipeline.components[0], Lift)
    x = torch.randn(2, 8)
    assert torch.equal(pipeline(x), linear(x))


def test_an_empty_serial_is_the_identity():
    x = torch.randn(2, 8)
    assert torch.equal(Serial([])(x), x)
