"""`Recurrent`: the three cells by name, the state at every step, the step order."""

import pytest
import torch
from jaxtyping import TypeCheckError
from torch import nn

from vectormesh.components import Recurrent
from vectormesh.types import VectorMeshError

KINDS = ["rnn", "gru", "lstm"]


@pytest.mark.parametrize(
    ("kind", "cell"), [("rnn", nn.RNN), ("gru", nn.GRU), ("lstm", nn.LSTM)]
)
def test_returns_the_cell_state_at_every_step(kind, cell):
    """The output is the cell's own output sequence: one tensor, not the tuple the
    PyTorch cell returns."""
    torch.manual_seed(0)
    rec = Recurrent(3, 5, kind=kind, n_layers=2)
    assert isinstance(rec.rnn, cell)
    x = torch.randn(2, 7, 3)
    out = rec(x)
    assert isinstance(out, torch.Tensor)
    expected, _ = rec.rnn(x)
    assert torch.equal(out, expected)


@pytest.mark.parametrize("kind", KINDS)
def test_the_last_step_is_the_final_hidden_state(kind):
    rec = Recurrent(3, 5, kind=kind, n_layers=2)
    x = torch.randn(2, 7, 3)
    _, hidden = rec.rnn(x)
    h_n = hidden[0] if kind == "lstm" else hidden
    assert torch.allclose(rec(x)[:, -1], h_n[-1], atol=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_a_step_sees_only_the_steps_before_it(kind):
    rec = Recurrent(3, 5, kind=kind)
    x = torch.randn(1, 8, 3)
    later = x.clone()
    later[0, 5:] = torch.randn(3, 3)
    out, out_later = rec(x), rec(later)
    assert torch.allclose(out[:, :5], out_later[:, :5])
    assert not torch.allclose(out[:, 5:], out_later[:, 5:])


def test_layers_are_stacked():
    assert Recurrent(3, 5, n_layers=3).rnn.num_layers == 3


def test_unknown_kind_is_refused():
    with pytest.raises(VectorMeshError, match="transformer") as error:
        Recurrent(3, 5, kind="transformer")  # ty: ignore[invalid-argument-type]
    assert error.value.hint is not None


def test_wrong_width_is_refused():
    with pytest.raises(TypeCheckError):
        Recurrent(3, 5)(torch.randn(2, 7, 4))


def test_repr_names_the_settings():
    assert "kind='lstm', n_layers=2" in repr(Recurrent(3, 5, "lstm", 2))
