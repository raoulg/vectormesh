"""Convolution and pooling: a window that slides along a sequence or over an image.

The 1D components take sequences with the channels on the last axis, as every other
component does, and move the axis for PyTorch themselves. The 2D components take
PyTorch's image layout, ``(batch, channels, height, width)``.

Every component tells the shape it will return before it runs: ``output_length(length)``
for a sequence, ``output_size(height, width)`` for an image.
"""

from typing import Literal

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from vectormesh.components.contracts import ImageComponent, SlidingComponent
from vectormesh.types import BaseComponent, VectorMeshError

PoolKind = Literal["max", "mean"]


def _sliding_length(
    component: str,
    axis: str,
    length: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """The number of window positions along one axis, as PyTorch computes it.

    Raises ``VectorMeshError`` when the window does not fit once: the axis is shorter
    than the window reaches.
    """
    positions = (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    if positions < 1:
        raise VectorMeshError(
            f"{component} gets a {axis} of length {length}, shorter than its window: "
            f"kernel_size={kernel_size}, dilation={dilation}, padding={padding}, "
            f"stride={stride} leave {positions} positions",
            hint="use a smaller kernel_size, more padding, or pool less before this "
            "component",
        )
    return positions


def _check_kind(component: str, kind: str) -> None:
    if kind not in ("max", "mean"):
        raise VectorMeshError(
            f"{component} kind {kind!r} is not one of 'max', 'mean'",
            hint="pass kind='max' to keep the strongest response in each window, "
            "'mean' to average the window",
        )


class Conv1D(SlidingComponent):
    """A 1D convolution along the sequence: ``(batch, seq, in_size) -> (batch, seq_out,
    out_size)``.

    Wraps ``nn.Conv1d``, which wants the channels on axis 1; the permute happens here,
    so the sequence keeps the library's layout on both sides. The length changes as
    ``output_length`` says: ``padding=(kernel_size - 1) // 2 * dilation`` with
    ``stride=1`` and an odd kernel keeps it.

    Args:
        in_size: channels of the input, its last axis.
        out_size: channels of the output, the number of filters.
        kernel_size: width of the window, in steps.
        stride: steps between two window positions.
        padding: zeros added at both ends of the sequence.
        dilation: spacing between the steps the window reads.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_size,
            out_size,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def output_length(self, length: int) -> int:
        """The length of the sequence this component returns for one of ``length``."""
        return _sliding_length(
            "Conv1D",
            "seq",
            length,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch seq {self.in_size}"]
    ) -> Float[Tensor, "batch seq_out {self.out_size}"]:
        # a window that does not fit raises here, naming this component
        self.output_length(tensors.shape[1])
        return self.conv(tensors.transpose(1, 2)).transpose(1, 2)

    def extra_repr(self) -> str:
        return (
            f"in_size={self.in_size}, out_size={self.out_size}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}"
        )


class Pool1D(SlidingComponent):
    """Pool non-overlapping windows along the sequence: ``(batch, seq, dim) -> (batch,
    seq_out, dim)``.

    Each window of ``kernel_size`` steps becomes one step, so the sequence gets
    ``kernel_size`` times shorter (``output_length``) and the width stays. A tail shorter
    than a window is dropped.

    Args:
        kernel_size: steps per window, which is also the step between windows.
        kind: ``"max"`` keeps the largest value of each channel in the window,
            ``"mean"`` averages it.
    """

    def __init__(self, kernel_size: int, kind: PoolKind = "max"):
        super().__init__()
        _check_kind("Pool1D", kind)
        self.kernel_size = kernel_size
        self.kind = kind
        self.pool: nn.Module = (
            nn.MaxPool1d(kernel_size) if kind == "max" else nn.AvgPool1d(kernel_size)
        )

    def output_length(self, length: int) -> int:
        """The length of the sequence this component returns for one of ``length``."""
        return _sliding_length(
            "Pool1D", "seq", length, self.kernel_size, stride=self.kernel_size
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch seq dim"]
    ) -> Float[Tensor, "batch seq_out dim"]:
        # a window that does not fit raises here, naming this component
        self.output_length(tensors.shape[1])
        return self.pool(tensors.transpose(1, 2)).transpose(1, 2)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, kind={self.kind!r}"


class Conv2D(ImageComponent):
    """A 2D convolution over an image: ``(batch, in_size, height, width) -> (batch,
    out_size, height_out, width_out)``.

    Wraps ``nn.Conv2d`` with a square kernel, and the same stride, padding and dilation
    on both axes. ``output_size`` gives the height and width it returns:
    ``padding=(kernel_size - 1) // 2 * dilation`` with ``stride=1`` and an odd kernel
    keeps them.

    Args:
        in_size: channels of the input, its second axis.
        out_size: channels of the output, the number of filters.
        kernel_size: height and width of the window, in pixels.
        stride: pixels between two window positions.
        padding: zeros added at every border.
        dilation: spacing between the pixels the window reads.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.conv = nn.Conv2d(
            in_size,
            out_size,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def output_size(self, height: int, width: int) -> tuple[int, int]:
        """The ``(height, width)`` this component returns for an image of that size."""
        args = (self.kernel_size, self.stride, self.padding, self.dilation)
        return (
            _sliding_length("Conv2D", "height", height, *args),
            _sliding_length("Conv2D", "width", width, *args),
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch {self.in_size} height width"]
    ) -> Float[Tensor, "batch {self.out_size} height_out width_out"]:
        # a window that does not fit raises here, naming this component
        self.output_size(tensors.shape[2], tensors.shape[3])
        return self.conv(tensors)

    def extra_repr(self) -> str:
        return (
            f"in_size={self.in_size}, out_size={self.out_size}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}"
        )


class Pool2D(ImageComponent):
    """Pool non-overlapping square windows of an image: ``(batch, channels, height,
    width) -> (batch, channels, height_out, width_out)``.

    Each ``kernel_size`` by ``kernel_size`` window becomes one pixel, so height and width
    get ``kernel_size`` times smaller (``output_size``) and the channels stay.

    Args:
        kernel_size: pixels per window side, which is also the step between windows.
        kind: ``"max"`` or ``"mean"``, as in ``Pool1D``.
    """

    def __init__(self, kernel_size: int, kind: PoolKind = "max"):
        super().__init__()
        _check_kind("Pool2D", kind)
        self.kernel_size = kernel_size
        self.kind = kind
        self.pool: nn.Module = (
            nn.MaxPool2d(kernel_size) if kind == "max" else nn.AvgPool2d(kernel_size)
        )

    def output_size(self, height: int, width: int) -> tuple[int, int]:
        """The ``(height, width)`` this component returns for an image of that size."""
        k = self.kernel_size
        return (
            _sliding_length("Pool2D", "height", height, k, stride=k),
            _sliding_length("Pool2D", "width", width, k, stride=k),
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch channels height width"]
    ) -> Float[Tensor, "batch channels height_out width_out"]:
        # a window that does not fit raises here, naming this component
        self.output_size(tensors.shape[2], tensors.shape[3])
        return self.pool(tensors)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, kind={self.kind!r}"


class GlobalPool2D(BaseComponent):
    """Pool each channel over the whole image: ``(batch, channels, height, width) ->
    (batch, channels)``.

    The step from an image to a vector, as an aggregator is the step from a sequence to
    a vector. It belongs to no family: its input is an image and its output a vector.

    Args:
        kind: ``"mean"`` averages each channel over all pixels, ``"max"`` keeps its
            largest value.
    """

    def __init__(self, kind: PoolKind = "mean"):
        super().__init__()
        _check_kind("GlobalPool2D", kind)
        self.kind = kind

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch channels height width"]
    ) -> Float[Tensor, "batch channels"]:
        if self.kind == "max":
            return torch.amax(tensors, dim=(2, 3))
        return tensors.mean(dim=(2, 3))

    def extra_repr(self) -> str:
        return f"kind={self.kind!r}"
