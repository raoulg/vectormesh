"""A recurrent layer over a sequence, the kind of cell chosen by name."""

from typing import Literal

from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from vectormesh.components.contracts import SequenceComponent
from vectormesh.types import VectorMeshError

RecurrentKind = Literal["rnn", "gru", "lstm"]


class Recurrent(SequenceComponent):
    """Run a recurrent layer over the sequence: ``(batch, seq, in_size) -> (batch, seq,
    hidden_size)``.

    Returns the state at every step and nothing else: the final hidden state (and an
    LSTM's cell state) are not returned, so the component composes in a ``Serial``. Put
    ``LastAggregator`` after it for the state after the last step, or any other
    aggregator to reduce the steps another way.

    ========  ===========================================================
    ``kind``  cell
    ========  ===========================================================
    ``rnn``   a plain tanh recurrence (``nn.RNN``)
    ``gru``   a gated recurrent unit (``nn.GRU``)
    ``lstm``  a long short-term memory, with its own cell state (``nn.LSTM``)
    ========  ===========================================================

    Every step is computed, padding included, so it suits sequences without padding.
    For padded chunks, ``RNNAggregator`` runs a GRU over the real chunks only.

    Args:
        in_size: width of the input's last axis.
        hidden_size: width of the state, the output's last axis.
        kind: which cell, see the table.
        n_layers: recurrent layers stacked on each other; the output is the top one's.
    """

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        kind: RecurrentKind = "gru",
        n_layers: int = 1,
    ):
        super().__init__()
        if kind not in ("rnn", "gru", "lstm"):
            raise VectorMeshError(
                f"Recurrent kind {kind!r} is not one of 'rnn', 'gru', 'lstm'",
                hint="pass kind='gru' for a gated cell, 'lstm' for one with a separate "
                "cell state, 'rnn' for the plain recurrence",
            )
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.kind = kind
        self.n_layers = n_layers
        cell = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind]
        self.rnn: nn.RNNBase = cell(
            in_size, hidden_size, num_layers=n_layers, batch_first=True
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tensors: Float[Tensor, "batch seq {self.in_size}"]
    ) -> Float[Tensor, "batch seq {self.hidden_size}"]:
        states, _ = self.rnn(tensors)
        return states

    def extra_repr(self) -> str:
        return (
            f"in_size={self.in_size}, hidden_size={self.hidden_size}, "
            f"kind={self.kind!r}, n_layers={self.n_layers}"
        )
