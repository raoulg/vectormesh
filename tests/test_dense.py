"""Dense and Activation: a hidden layer chosen by name, at any rank."""

import pytest
import torch
from torch import nn

from vectormesh.components import Activation, Dense, Norm
from vectormesh.types import VectorMeshError


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("gelu", nn.GELU),
        ("relu", nn.ReLU),
        ("tanh", nn.Tanh),
        ("sigmoid", nn.Sigmoid),
        ("none", nn.Identity),
    ],
)
def test_activation_applies_the_named_function(kind, expected):
    layer = Activation(kind=kind)
    assert isinstance(layer.activation, expected)
    x = torch.randn(3, 5)
    assert torch.equal(layer(x), expected()(x))


def test_activation_refuses_an_unknown_kind_and_lists_the_kinds():
    with pytest.raises(VectorMeshError) as error:
        Activation(kind="swish")  # ty: ignore[invalid-argument-type]
    message = str(error.value)
    for kind in ("gelu", "relu", "tanh", "sigmoid", "none"):
        assert repr(kind) in message


@pytest.mark.parametrize("shape", [(4, 8), (4, 6, 8)])
def test_dense_changes_the_last_axis_at_any_rank(shape):
    out = Dense(8, 3)(torch.randn(*shape))
    assert out.shape == (*shape[:-1], 3)


def test_dense_is_linear_norm_activation_dropout():
    dense = Dense(8, 3, norm="layer", activation="relu", dropout=0.5)
    parts = list(dense.layers)
    assert [type(p) for p in parts] == [nn.Linear, Norm, Activation, nn.Dropout]
    assert parts[3].p == 0.5


def test_dense_defaults_are_a_linear_layer_and_gelu():
    dense = Dense(8, 3)
    x = torch.randn(2, 8)
    assert torch.allclose(dense(x), nn.functional.gelu(dense.layers[0](x)))


def test_dense_checks_the_width_it_takes():
    with pytest.raises(Exception, match="in_size|8"):
        Dense(8, 3)(torch.randn(2, 7))
