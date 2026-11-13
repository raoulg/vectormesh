"""Three kernel sizes side by side on one sequence, joined on the feature axis.

Each stage is a method whose signature states the shape it takes and the shape it gives.
The activation is plain PyTorch, because the library has no component for it. Run from
the repository root:

    uv run python scripts/design_parallel_kernels.py
"""

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from vectormesh.components import Concatenate3D, Conv1D, MeanAggregator, Projection


class ParallelKernels(nn.Module):
    """``(batch, seq, sensors) -> (batch, classes)``."""

    def __init__(self, n_sensors: int, channels: int, n_classes: int) -> None:
        super().__init__()
        self.n_sensors = n_sensors
        self.channels = channels
        self.joined = 3 * channels
        self.n_classes = n_classes
        # padding kernel_size // 2 keeps the length, so the three branches line up
        self.conv_narrow = Conv1D(n_sensors, channels, kernel_size=3, padding=1)
        self.conv_medium = Conv1D(n_sensors, channels, kernel_size=5, padding=2)
        self.conv_wide = Conv1D(n_sensors, channels, kernel_size=7, padding=3)
        self.concatenate = Concatenate3D()
        self.mean = MeanAggregator()
        self.head = Projection(self.joined, n_classes)

    @jaxtyped(typechecker=beartype)
    def narrow(
        self, series: Float[Tensor, "batch seq {self.n_sensors}"]
    ) -> Float[Tensor, "batch seq {self.channels}"]:
        """A window of 3 steps."""
        return torch.relu(self.conv_narrow(series))

    @jaxtyped(typechecker=beartype)
    def medium(
        self, series: Float[Tensor, "batch seq {self.n_sensors}"]
    ) -> Float[Tensor, "batch seq {self.channels}"]:
        """A window of 5 steps."""
        return torch.relu(self.conv_medium(series))

    @jaxtyped(typechecker=beartype)
    def wide(
        self, series: Float[Tensor, "batch seq {self.n_sensors}"]
    ) -> Float[Tensor, "batch seq {self.channels}"]:
        """A window of 7 steps."""
        return torch.relu(self.conv_wide(series))

    @jaxtyped(typechecker=beartype)
    def join(
        self,
        narrow: Float[Tensor, "batch seq {self.channels}"],
        medium: Float[Tensor, "batch seq {self.channels}"],
        wide: Float[Tensor, "batch seq {self.channels}"],
    ) -> Float[Tensor, "batch seq {self.joined}"]:
        """Every step keeps what all three windows saw."""
        return self.concatenate((narrow, medium, wide))

    @jaxtyped(typechecker=beartype)
    def summarise(
        self, joined: Float[Tensor, "batch seq {self.joined}"]
    ) -> Float[Tensor, "batch {self.joined}"]:
        """From a sequence to a vector: the mean over the steps."""
        return self.mean(joined)

    @jaxtyped(typechecker=beartype)
    def classify(
        self, vector: Float[Tensor, "batch {self.joined}"]
    ) -> Float[Tensor, "batch {self.n_classes}"]:
        return self.head(vector)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, series: Float[Tensor, "batch seq {self.n_sensors}"]
    ) -> Float[Tensor, "batch {self.n_classes}"]:
        narrow = self.narrow(series)
        medium = self.medium(series)
        wide = self.wide(series)
        joined = self.join(narrow, medium, wide)
        vector = self.summarise(joined)
        return self.classify(vector)


@jaxtyped(typechecker=beartype)
def random_series(
    batch_size: int, seq: int, n_sensors: int
) -> Float[Tensor, "{batch_size} {seq} {n_sensors}"]:
    return torch.randn(batch_size, seq, n_sensors)


def main() -> None:
    model = ParallelKernels(n_sensors=3, channels=8, n_classes=3)
    series = random_series(batch_size=4, seq=100, n_sensors=model.n_sensors)
    logits = model.forward(series)
    print(model)
    print(f"{tuple(series.shape)} -> {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
