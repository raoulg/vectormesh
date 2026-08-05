"""Tests for `NeuralNet`, in particular the `hidden_size: int | list[int]` depth API.

Stacking two `NeuralNet`s to "add a hidden layer" (`Serial([NeuralNet(d, w),
NeuralNet(w, k)])`) puts a bare `Linear -> Linear` seam in the middle -- notebook 02 of
the course spends a whole section proving that costs nothing over a single `Linear`.
`hidden_size` accepting a list exists so a caller never has to compose `NeuralNet`s to
get depth: the list names the whole input->hidden width chain in one place.
"""

import pytest
import torch
import torch.nn as nn

from vectormesh.components.neural import NeuralNet
from vectormesh.types import VectorMeshError


def test_default_int_matches_original_two_layer_shape():
    torch.manual_seed(0)
    net = NeuralNet(hidden_size=8, out_size=3)
    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    assert len(linears) == 2
    assert linears[0].in_features == 8 and linears[0].out_features == 8
    assert linears[1].in_features == 8 and linears[1].out_features == 3


def test_single_element_list_is_equivalent_to_the_bare_int():
    torch.manual_seed(0)
    from_int = NeuralNet(hidden_size=8, out_size=3)
    torch.manual_seed(0)
    from_list = NeuralNet(hidden_size=[8], out_size=3)

    int_widths = [
        (m.in_features, m.out_features)
        for m in from_int.modules()
        if isinstance(m, nn.Linear)
    ]
    list_widths = [
        (m.in_features, m.out_features)
        for m in from_list.modules()
        if isinstance(m, nn.Linear)
    ]
    assert int_widths == list_widths

    x = torch.randn(4, 8)
    assert torch.allclose(from_int(x), from_list(x))


def test_default_forward_is_linear_gelu_linear():
    torch.manual_seed(0)
    net = NeuralNet(hidden_size=8, out_size=3)
    x = torch.randn(5, 8)
    layers = list(net.layers)
    expected = layers[1](net.activation(layers[0](x)))
    assert torch.allclose(net(x), expected)


def test_two_element_list_is_one_hidden_layer_at_that_width():
    """[384, 128] is what the search's 'width' axis should build: input 384, one
    hidden layer of 128, not the old two-NeuralNet composition."""
    net = NeuralNet(hidden_size=[384, 128], out_size=47)
    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    widths = [(lin.in_features, lin.out_features) for lin in linears]
    assert widths == [(384, 128), (128, 47)]


def test_longer_list_adds_more_hidden_layers():
    net = NeuralNet(hidden_size=[8, 16, 32], out_size=3)
    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    widths = [(lin.in_features, lin.out_features) for lin in linears]
    assert widths == [(8, 16), (16, 32), (32, 3)]


def test_empty_list_raises():
    with pytest.raises(VectorMeshError, match="no input dimension"):
        NeuralNet(hidden_size=[], out_size=3)


def test_runs_and_shapes_correctly():
    net = NeuralNet(hidden_size=[8, 16], out_size=3)
    out = net(torch.randn(5, 8))
    assert out.shape == (5, 3)


def test_accepts_any_leading_shape():
    """Matches the class docstring's claim: acts on the last dim only."""
    net = NeuralNet(hidden_size=[8, 16], out_size=3)
    out = net(torch.randn(2, 7, 8))
    assert out.shape == (2, 7, 3)


def test_no_double_linear_seam_for_any_depth():
    """The property the whole feature exists for: two Linears never sit back-to-back
    with nothing nonlinear between them, however deep the list goes."""
    net = NeuralNet(hidden_size=[4, 6, 6, 6], out_size=2)
    layers = list(net.layers)
    x = torch.randn(3, 4)
    manual = x
    for layer in layers[:-1]:
        manual = net.activation(layer(manual))
    manual = layers[-1](manual)
    assert torch.allclose(net(x), manual)
    assert len(layers) == len(
        net.widths
    )  # widths (chain) + out_size -> len(widths) Linears
