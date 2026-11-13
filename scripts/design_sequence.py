"""A sequence classifier: a kernel finds local patterns, a recurrent layer reads them in order.

Each stage is a method whose signature states the shape it takes and the shape it gives.
Run from the repository root:

    uv run python scripts/design_sequence.py
"""

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from vectormesh.components import (
    Conv1D,
    Gate,
    LastAggregator,
    Pool1D,
    Projection,
    Recurrent,
)


class SequenceClassifier(nn.Module):
    """``(batch, seq, sensors) -> (batch, classes)``."""

    def __init__(
        self,
        n_sensors: int,
        channels: int,
        kernel_size: int,
        pool_size: int,
        hidden_size: int,
        n_classes: int,
    ) -> None:
        super().__init__()
        self.n_sensors = n_sensors
        self.channels = channels
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.conv = Conv1D(n_sensors, channels, kernel_size=kernel_size)
        self.pool = Pool1D(pool_size)
        self.recurrent = Recurrent(channels, hidden_size, kind="gru")
        self.last = LastAggregator()
        self.gate = Gate(hidden_size)
        self.head = Projection(hidden_size, n_classes)

    @jaxtyped(typechecker=beartype)
    def local_patterns(
        self, series: Float[Tensor, "batch seq {self.n_sensors}"]
    ) -> Float[Tensor, "batch windows {self.channels}"]:
        """A kernel slides along time: one vector per window of steps."""
        return self.conv(series)

    @jaxtyped(typechecker=beartype)
    def shorten(
        self, patterns: Float[Tensor, "batch windows {self.channels}"]
    ) -> Float[Tensor, "batch pooled {self.channels}"]:
        """Keep the strongest response in every run of windows."""
        return self.pool(patterns)

    @jaxtyped(typechecker=beartype)
    def read_in_order(
        self, pooled: Float[Tensor, "batch pooled {self.channels}"]
    ) -> Float[Tensor, "batch pooled {self.hidden_size}"]:
        """A state carried from step to step: one state per step."""
        return self.recurrent(pooled)

    @jaxtyped(typechecker=beartype)
    def final_state(
        self, states: Float[Tensor, "batch pooled {self.hidden_size}"]
    ) -> Float[Tensor, "batch {self.hidden_size}"]:
        """The state after the whole sequence."""
        return self.last(states)

    @jaxtyped(typechecker=beartype)
    def select(
        self, state: Float[Tensor, "batch {self.hidden_size}"]
    ) -> Float[Tensor, "batch {self.hidden_size}"]:
        """A gate scales every feature of the state by a value between 0 and 1."""
        return self.gate(state)

    @jaxtyped(typechecker=beartype)
    def classify(
        self, state: Float[Tensor, "batch {self.hidden_size}"]
    ) -> Float[Tensor, "batch {self.n_classes}"]:
        return self.head(state)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, series: Float[Tensor, "batch seq {self.n_sensors}"]
    ) -> Float[Tensor, "batch {self.n_classes}"]:
        patterns = self.local_patterns(series)
        pooled = self.shorten(patterns)
        states = self.read_in_order(pooled)
        state = self.final_state(states)
        selected = self.select(state)
        return self.classify(selected)


@jaxtyped(typechecker=beartype)
def random_series(
    batch_size: int, seq: int, n_sensors: int
) -> Float[Tensor, "{batch_size} {seq} {n_sensors}"]:
    return torch.randn(batch_size, seq, n_sensors)


def main() -> None:
    model = SequenceClassifier(
        n_sensors=3,
        channels=16,
        kernel_size=5,  # 200 steps -> 196 windows
        pool_size=4,  # 196 windows -> 49
        hidden_size=32,
        n_classes=3,
    )
    series = random_series(batch_size=4, seq=200, n_sensors=model.n_sensors)
    logits = model.forward(series)
    print(model)
    print(f"{tuple(series.shape)} -> {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
