"""A recurrent layer over a sequence, the kind of cell chosen by name."""

from typing import Literal

import torch
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from vectormesh.components.aggregation import real_first
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

    With a mask (a `Padded` batch) the recurrence runs over each item's real steps
    only, in their order, also when padding sits between them; a padded step's
    output is zero. ``LastAggregator`` after it then takes the state after the last
    real step. Without a mask every step is real.

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
        self,
        tensors: Float[Tensor, "batch seq {self.in_size}"],
        mask: Bool[Tensor, "batch seq"] | None = None,
    ) -> Float[Tensor, "batch seq {self.hidden_size}"]:
        if mask is None:
            states, _ = self.rnn(tensors)
            return states
        seq = tensors.shape[1]
        compact, order, lengths = real_first(tensors, mask)
        packed = pack_padded_sequence(
            compact, lengths, batch_first=True, enforce_sorted=False
        )
        states, _ = pad_packed_sequence(
            self.rnn(packed)[0], batch_first=True, total_length=seq
        )
        states = states * mask.sum(dim=1).gt(0).view(-1, 1, 1)  # all padding: zeros
        # compact step j is step order[:, j] of the item; steps past its length are padding
        out = torch.zeros_like(states)
        return out.scatter(1, order.unsqueeze(-1).expand_as(states), states)

    def extra_repr(self) -> str:
        return (
            f"in_size={self.in_size}, hidden_size={self.hidden_size}, "
            f"kind={self.kind!r}, n_layers={self.n_layers}"
        )
