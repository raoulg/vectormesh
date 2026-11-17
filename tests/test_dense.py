"""Dense and Activation: a hidden layer chosen by name, at any rank."""

import pytest
import torch
from jaxtyping import TypeCheckError
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
    assert parts[1].kind == "layer"
    assert parts[2].kind == "relu"
    assert parts[3].p == 0.5


@pytest.mark.parametrize("norm", ["layer", "batch", "rms"])
@pytest.mark.parametrize("shape", [(6, 8), (6, 4, 8)])
def test_dense_in_train_mode_is_the_four_layers_composed_by_hand(norm, shape):
    dense = Dense(8, 3, norm=norm, activation="tanh", dropout=0.5).train()
    linear, normalise, activate, drop = dense.layers
    x = torch.randn(*shape)

    torch.manual_seed(0)
    got = dense(x)
    torch.manual_seed(0)
    expected = drop(activate(normalise(linear(x))))

    assert torch.equal(got, expected)
    assert (got == 0).any()  # the dropout acted
    assert not torch.equal(got, drop(activate(linear(x))))  # the norm acted


def test_dense_defaults_are_a_linear_layer_and_gelu():
    dense = Dense(8, 3)
    x = torch.randn(2, 8)
    assert torch.allclose(dense(x), nn.functional.gelu(dense.layers[0](x)))


def test_dense_checks_the_width_it_takes_and_the_width_it_gives():
    with pytest.raises(TypeCheckError):
        Dense(8, 3)(torch.randn(2, 7))

    wrong = Dense(8, 3)
    wrong.layers[0] = nn.Linear(8, 5)
    with pytest.raises(TypeCheckError):
        wrong(torch.randn(2, 8))
