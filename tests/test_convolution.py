"""Convolution and pooling: the shape each component announces is the shape it returns,
and the values are those of the PyTorch operation on the moved axis."""

import itertools

import pytest
import torch
import torch.nn.functional as F
from jaxtyping import TypeCheckError

from vectormesh.components import (
    Conv1D,
    Conv2D,
    GlobalPool2D,
    ImageComponent,
    Pool1D,
    Pool2D,
    SlidingComponent,
)
from vectormesh.types import VectorMeshError

GRID = list(itertools.product([1, 2, 3, 5], [1, 2, 3], [0, 1, 2], [1, 2]))


@pytest.mark.parametrize(
    ("member", "family"),
    [
        (Conv1D, SlidingComponent),
        (Pool1D, SlidingComponent),
        (Conv2D, ImageComponent),
        (Pool2D, ImageComponent),
    ],
)
def test_member_is_in_its_family_only(member, family):
    assert issubclass(member, family)
    other = ImageComponent if family is SlidingComponent else SlidingComponent
    assert not issubclass(member, other)


@pytest.mark.parametrize(("kernel", "stride", "padding", "dilation"), GRID)
def test_conv1d_returns_the_length_it_announces(kernel, stride, padding, dilation):
    conv = Conv1D(3, 4, kernel, stride=stride, padding=padding, dilation=dilation)
    for length in (11, 20):
        out = conv(torch.randn(2, length, 3))
        assert out.shape == (2, conv.output_length(length), 4)


@pytest.mark.parametrize(("kernel", "stride", "padding", "dilation"), GRID)
def test_conv2d_returns_the_size_it_announces(kernel, stride, padding, dilation):
    conv = Conv2D(3, 4, kernel, stride=stride, padding=padding, dilation=dilation)
    out = conv(torch.randn(2, 3, 11, 16))
    assert out.shape == (2, 4, *conv.output_size(11, 16))


@pytest.mark.parametrize("kernel", [1, 2, 3, 4])
@pytest.mark.parametrize("kind", ["max", "mean"])
def test_pools_return_the_shape_they_announce(kernel, kind):
    pool1, pool2 = Pool1D(kernel, kind), Pool2D(kernel, kind)
    assert pool1(torch.randn(2, 11, 3)).shape == (2, pool1.output_length(11), 3)
    assert pool2(torch.randn(2, 3, 11, 9)).shape == (2, 3, *pool2.output_size(11, 9))


def test_conv1d_convolves_along_the_sequence():
    """The channels move to axis 1 and back: the values are `F.conv1d` on the
    channels-first sequence, not on a reshaped one."""
    conv = Conv1D(3, 4, 5, stride=2, padding=1, dilation=2)
    x = torch.randn(2, 20, 3)
    expected = F.conv1d(
        x.transpose(1, 2),
        conv.conv.weight,
        conv.conv.bias,
        stride=2,
        padding=1,
        dilation=2,
    ).transpose(1, 2)
    assert torch.allclose(conv(x), expected, atol=1e-6)


def test_conv1d_is_local_along_the_sequence():
    """A change at one step reaches only the outputs whose window covers it."""
    conv = Conv1D(3, 4, 3)
    x = torch.randn(1, 20, 3)
    moved = x.clone()
    moved[0, 10] += 1.0
    changed = (conv(moved) - conv(x)).abs().sum(-1)[0] > 0
    assert changed.nonzero().flatten().tolist() == [8, 9, 10]


def test_conv2d_is_nn_conv2d():
    conv = Conv2D(3, 4, 3, padding=1)
    x = torch.randn(2, 3, 8, 8)
    expected = F.conv2d(x, conv.conv.weight, conv.conv.bias, padding=1)
    assert torch.allclose(conv(x), expected, atol=1e-6)


def test_pool1d_pools_each_channel_over_its_window():
    x = torch.arange(24.0).reshape(1, 8, 3)  # steps 0..7, three channels
    windows = x.reshape(1, 4, 2, 3)  # four windows of two steps
    assert torch.equal(Pool1D(2, "max")(x), windows.amax(dim=2))
    assert torch.allclose(Pool1D(2, "mean")(x), windows.mean(dim=2))


def test_pool2d_pools_each_channel_over_its_window():
    x = torch.randn(2, 3, 4, 6)
    windows = x.reshape(2, 3, 2, 2, 3, 2)  # (b, c, h/2, 2, w/2, 2)
    assert torch.equal(Pool2D(2, "max")(x), windows.amax(dim=(3, 5)))
    assert torch.allclose(Pool2D(2, "mean")(x), windows.mean(dim=(3, 5)), atol=1e-6)


def test_global_pool_reduces_each_channel_over_the_image():
    x = torch.randn(2, 3, 5, 7)
    assert torch.allclose(GlobalPool2D()(x), x.mean(dim=(2, 3)))
    assert torch.equal(GlobalPool2D("max")(x), x.flatten(2).amax(-1))


@pytest.mark.parametrize("component", [Pool1D, Pool2D, GlobalPool2D])
def test_unknown_kind_is_refused(component):
    args = () if component is GlobalPool2D else (2,)
    with pytest.raises(VectorMeshError, match="median") as error:
        component(*args, kind="median")
    assert error.value.hint is not None


def test_wrong_width_is_refused_at_the_boundary():
    with pytest.raises(TypeCheckError):
        Conv1D(3, 4, 3)(torch.randn(2, 20, 5))


def test_channels_first_sequence_is_refused():
    """A sequence in PyTorch's `(batch, channels, seq)` layout fails on the width."""
    with pytest.raises(TypeCheckError):
        Conv1D(3, 4, 3)(torch.randn(2, 3, 20))


def test_wrong_channels_are_refused_at_the_boundary():
    with pytest.raises(TypeCheckError):
        Conv2D(3, 4, 3)(torch.randn(2, 4, 8, 8))


def test_repr_names_the_settings():
    assert "kernel_size=5" in repr(Conv1D(3, 4, 5))
    assert "dilation=2" in repr(Conv2D(3, 4, 3, dilation=2))
    assert "kind='mean'" in repr(Pool1D(2, "mean"))
    assert "kind='max'" in repr(GlobalPool2D("max"))
