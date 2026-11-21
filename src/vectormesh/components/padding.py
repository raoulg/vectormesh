"""Padders: a list of ``(chunks_i, dim)`` tensors to one `Padded` batch."""

from typing import Any

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Bool, jaxtyped
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from vectormesh.types import Padded


def _mask(
    lengths: list[int], seq: int, device: torch.device
) -> Bool[Tensor, "batch seq"]:
    """True for the first ``lengths[i]`` positions of item ``i``."""
    length = torch.tensor(lengths, device=device)
    return torch.arange(seq, device=device).unsqueeze(0) < length.unsqueeze(1)


class DynamicPadding:
    """Pad every item with zero chunks to the longest item of the batch.

    Returns a `Padded` batch: the zero-padded values, ``(batch, max(chunks_i), dim)``,
    and the mask of the real chunks.
    """

    @jaxtyped(typechecker=beartype)
    def __call__(self, tensors: list[Any]) -> Padded:
        padded = pad_sequence(tensors, batch_first=True)
        lengths = [len(t) for t in tensors]
        return Padded(padded, _mask(lengths, padded.shape[1], padded.device))


class FixedPadding:
    """Pad or cut every item to ``max_chunks`` chunks.

    Returns a `Padded` batch: the values, ``(batch, max_chunks, dim)``, zero where an
    item has fewer chunks, and the mask of the real chunks. An item with more than
    ``max_chunks`` chunks keeps its first ``max_chunks``.
    """

    def __init__(self, max_chunks: int):
        self.max_chunks = max_chunks

    @jaxtyped(typechecker=beartype)
    def __call__(self, tensors: list[Any]) -> Padded:
        padded = pad_sequence(tensors, batch_first=True)
        current = padded.shape[1]
        if current < self.max_chunks:
            padded = F.pad(padded, (0, 0, 0, self.max_chunks - current))
        else:
            padded = padded[:, : self.max_chunks, :]
        lengths = [min(len(t), self.max_chunks) for t in tensors]
        return Padded(padded, _mask(lengths, self.max_chunks, padded.device))
