"""Three small models built from library components, to test how the convolution and
recurrent components compose with the pipelines, gates and connectors.

Each model has a shape test and a training test: a few seeded steps on synthetic data
with a learnable signal, after which the loss must have fallen. Where a model needs
something the library does not have, the spot is marked ``WORKAROUND`` with what is
missing, and the workaround is the smallest one: a plain PyTorch module, lifted.
"""

import math

import pytest
import torch
from torch import Tensor, nn

from vectormesh.components import (
    Concatenate3D,
    Conv1D,
    Conv2D,
    Gate,
    GlobalPool2D,
    LastAggregator,
    Lift,
    MeanAggregator,
    NeuralNet,
    Parallel,
    Pool1D,
    Projection,
    Recurrent,
    Serial,
)

CLASSES = 3
BATCH = 48


def _waves(n: int, length: int, seed: int) -> tuple[Tensor, Tensor]:
    """``(n, length, 3)`` sequences whose first channel is a noisy sine; the class sets
    its frequency. The other two channels are noise."""
    gen = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, CLASSES, (n,), generator=gen)
    steps = torch.arange(length).float()
    freq = (labels.float() + 1).unsqueeze(1) * 2 * math.pi / 25
    phase = torch.rand(n, 1, generator=gen) * 2 * math.pi
    x = 0.3 * torch.randn(n, length, 3, generator=gen)
    x[:, :, 0] += torch.sin(freq * steps + phase)
    return x, labels


def _images(n: int, seed: int) -> tuple[Tensor, Tensor]:
    """``(n, 3, 32, 32)`` noise images; the class is the channel that is brighter."""
    gen = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, CLASSES, (n,), generator=gen)
    x = torch.randn(n, 3, 32, 32, generator=gen)
    x[torch.arange(n), labels] += 0.5
    return x, labels


def _losses(model: nn.Module, x: Tensor, y: Tensor, steps: int) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def _falls(losses: list[float]) -> bool:
    """The mean of the last three losses is well under that of the first three."""
    return sum(losses[-3:]) < 0.7 * sum(losses[:3])


# --- A: a sequence through convolution, pooling and a recurrent layer ---------------


def sequence_model() -> Serial:
    """``(batch, 200, 3) -> (batch, CLASSES)``, in one `Serial`."""
    return Serial(
        [
            Conv1D(3, 16, kernel_size=5),  # (B, 200, 3) -> (B, 196, 16)
            Pool1D(4),  # -> (B, 49, 16)
            Recurrent(16, 32, kind="gru"),  # -> (B, 49, 32)
            LastAggregator(),  # -> (B, 32)
            Gate(32),
            Projection(32, CLASSES),
        ]
    )


def test_sequence_model_shape():
    torch.manual_seed(0)
    assert sequence_model()(torch.randn(4, 200, 3)).shape == (4, CLASSES)


def test_sequence_model_learns():
    torch.manual_seed(0)
    x, y = _waves(BATCH, 200, seed=1)
    assert _falls(_losses(sequence_model(), x, y, steps=30))


# --- B: an image through a residual convolution block and a squeeze-excite gate -----


class _Residual(nn.Module):
    """``x + block(x)`` on an image.

    WORKAROUND: `Skip` normalises and checks the last axis (``"*lead {in_size}"``). On
    ``(batch, channels, height, width)`` that is the width: ``Skip(block, channels)``
    is refused, and ``Skip(block, width)`` runs and layer-normalises each row of pixels.
    Missing: a residual whose norm and check can be put on the channel axis, or one
    without a norm.
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class _SqueezeExcite(nn.Module):
    """Scale each channel by a gate computed from the channel means.

    WORKAROUND: `Gate` computes its gate from the tensor it multiplies, on the last
    axis, so it cannot take its gate from the ``(batch, channels)`` means and apply it
    to the ``(batch, channels, height, width)`` image. The pieces exist (`GlobalPool2D`
    for the means, `NeuralNet` for the excitation); missing is the broadcast multiply
    that puts a ``(batch, channels)`` gate on whole channels, and a sigmoid.
    """

    def __init__(self, channels: int, reduced: int):
        super().__init__()
        self.squeeze = GlobalPool2D("mean")
        self.excite = NeuralNet([channels, reduced], channels)

    def forward(self, x: Tensor) -> Tensor:
        gate = torch.sigmoid(self.excite(self.squeeze(x)))  # (B, C)
        return x * gate[:, :, None, None]


def image_model() -> Serial:
    """``(batch, 3, 32, 32) -> (batch, CLASSES)``."""
    return Serial(
        [
            Conv2D(3, 8, kernel_size=3, padding=1),  # (B, 3, 32, 32) -> (B, 8, 32, 32)
            # WORKAROUND: the library has no activation component, so ReLU is plain
            # PyTorch, lifted by `Serial`.
            nn.ReLU(),
            Lift(
                _Residual(
                    Serial(
                        [
                            Conv2D(8, 8, kernel_size=3, padding=1),
                            nn.ReLU(),  # WORKAROUND: no activation component
                            Conv2D(8, 8, kernel_size=3, padding=1),
                        ]
                    )
                )
            ),
            Lift(_SqueezeExcite(8, reduced=4)),
            GlobalPool2D("mean"),  # -> (B, 8)
            Projection(8, CLASSES),
        ]
    )


def test_image_model_shape():
    torch.manual_seed(0)
    assert image_model()(torch.randn(4, 3, 32, 32)).shape == (4, CLASSES)


def test_image_model_learns():
    torch.manual_seed(0)
    x, y = _images(BATCH, seed=2)
    assert _falls(_losses(image_model(), x, y, steps=20))


# --- C: three kernel sizes in parallel on one sequence -----------------------------


class _Copies(nn.Module):
    """The same tensor ``n`` times, as a tuple.

    WORKAROUND: `Parallel` gives branch *i* element *i* of its input, so three branches
    on one sequence need three copies of it. Missing: a fan-out that hands one input to
    every branch. Given the tensor itself, `Parallel` zips the branches with its batch
    rows.
    """

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        return (x,) * self.n


def parallel_kernels_model() -> Serial:
    """``(batch, 100, 3) -> (batch, CLASSES)``: kernels 3, 5 and 7 side by side."""
    branches = [
        # padding k // 2 keeps the length, so `Concatenate3D` finds one chunk axis.
        # WORKAROUND: no activation component, so ReLU is plain PyTorch.
        Serial([Conv1D(3, 8, kernel_size=k, padding=k // 2), nn.ReLU()])
        for k in (3, 5, 7)
    ]
    return Serial(
        [
            Lift(_Copies(3)),  # (B, 100, 3) -> three times (B, 100, 3)
            Parallel(branches),  # -> three times (B, 100, 8)
            Concatenate3D(),  # -> (B, 100, 24)
            MeanAggregator(),  # -> (B, 24)
            Projection(24, CLASSES),
        ]
    )


def test_parallel_kernels_model_shape():
    torch.manual_seed(0)
    assert parallel_kernels_model()(torch.randn(4, 100, 3)).shape == (4, CLASSES)


def test_parallel_kernels_model_learns():
    torch.manual_seed(0)
    x, y = _waves(BATCH, 100, seed=3)
    assert _falls(_losses(parallel_kernels_model(), x, y, steps=30))


# --- every model: a gradient reaches every parameter --------------------------------


@pytest.mark.parametrize(
    ("build", "shape"),
    [
        (sequence_model, (4, 200, 3)),
        (image_model, (4, 3, 32, 32)),
        (parallel_kernels_model, (4, 100, 3)),
    ],
    ids=["sequence", "image", "parallel_kernels"],
)
def test_every_parameter_gets_a_gradient(build, shape):
    torch.manual_seed(0)
    model = build()
    model(torch.randn(*shape)).pow(2).sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.abs().sum() > 0, name
