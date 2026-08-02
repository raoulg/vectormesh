from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Iterator, Protocol, Union, runtime_checkable

import numpy as np
import torch
from loguru import logger

NumericType = Union[torch.Tensor, np.ndarray]

#: one (x, y) batch, as a DataLoader or a legacy streamer yields it
Batch = tuple[torch.Tensor, torch.Tensor]


@runtime_checkable
class Streamer(Protocol):
    """The mltrainer ``BaseDatastreamer`` interface.

    Kept only so metrics fitted on training data can still accept one during the
    deprecation window; ``stream()`` returns an unbounded generator, which is why
    callers need ``__len__`` to know when one pass is done.

    .. deprecated:: 0.4.3
       Removed in 0.5. Pass a ``torch.utils.data.DataLoader`` instead.
    """

    def __len__(self) -> int: ...

    def stream(self) -> Iterator[Batch]: ...


#: what a fitted metric accepts as its training data
Batches = Union[Iterable[Batch], Streamer]


class Metric(ABC):
    """Base class for all metrics with proper device and type handling"""

    @abstractmethod
    def _compute(self, y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
        """Actual metric computation - implemented by subclasses"""
        pass

    def __call__(self, y: NumericType, yhat: NumericType) -> float:
        """Handle device/type conversion and return final metric"""
        # Convert to tensors if needed
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        if isinstance(yhat, np.ndarray):
            yhat = torch.from_numpy(yhat)

        # Ensure we're working with the same device
        device = y.device
        y = y.to(device)
        yhat = yhat.to(device)

        # Compute metric
        result = self._compute(y, yhat)

        # Return as float for consistency
        return float(result.cpu().detach())

    @abstractmethod
    def __repr__(self) -> str:
        pass


class MAE(Metric):
    def _compute(self, y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(y - yhat))

    def __repr__(self) -> str:
        return "MAE"


class MASE(Metric):
    """Mean Absolute Scaled Error: MAE divided by a naive forecaster's MAE.

    The scale is fitted once, on training batches, so the metric is comparable
    across series with different units and does not move when the evaluation
    split changes.

    ``train`` is any iterable of ``(x, y)`` batches — a ``torch.utils.data.DataLoader``
    is the expected type. ``x`` may be ``(batch, time)`` or ``(batch, time, 1)``;
    ``y`` is ``(batch, horizon)``.

    .. deprecated:: 0.4.3
       Passing an mltrainer ``BaseDatastreamer`` (anything exposing ``.stream()``)
       still works but logs a warning and will be removed in 0.5. Pass a
       ``DataLoader`` instead.
    """

    def __init__(self, train: Batches, horizon: int) -> None:
        self.horizon = horizon
        with torch.no_grad():
            self.scale = self._calculate_scale(train)

    def _batches(self, train: Batches) -> Iterator[Batch]:
        """Yield ``(x, y)`` batches from a DataLoader, or from a legacy streamer."""
        if isinstance(train, Streamer):
            logger.warning(
                "MASE received a streamer (an object with .stream()). Support for this "
                "is deprecated and will be removed in vectormesh 0.5 — pass a "
                "torch.utils.data.DataLoader instead."
            )
            stream = train.stream()
            for _ in range(len(train)):  # stream() is unbounded; one pass only
                yield next(stream)
            return
        yield from train

    def _calculate_scale(self, train: Batches) -> torch.Tensor:
        errors = [
            torch.mean(torch.abs(y - self._naive_predict(x)))
            for x, y in self._batches(train)
        ]
        if not errors:
            raise ValueError(
                "MASE needs at least one batch to fit its scale, got none."
            )
        scale = torch.mean(torch.stack(errors))
        if scale == 0:
            raise ValueError(
                "MASE scale is zero: the naive forecast is exact on the training data, "
                "so every scaled error would be infinite."
            )
        return scale

    def _naive_predict(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat the last ``horizon`` observed steps — the seasonal-naive forecast.

        Accepts ``(batch, time)`` and ``(batch, time, 1)``; the trailing singleton
        feature axis is dropped so the result lines up with a ``(batch, horizon)``
        target either way.
        """
        naive = x[..., -self.horizon :, :] if x.ndim == 3 else x[..., -self.horizon :]
        return naive.squeeze(-1) if naive.ndim == 3 and naive.shape[-1] == 1 else naive

    def _compute(self, y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
        mae = torch.mean(torch.abs(y - yhat))
        return mae / self.scale

    def __repr__(self) -> str:
        return f"MASE(scale={self.scale:.3f})"


class Accuracy(Metric):
    """Top-1 accuracy.

    Accepts either integer class targets ``y`` of shape ``(batch,)`` or one-hot /
    soft targets of shape ``(batch, num_classes)`` (as produced by ``OneHot`` +
    ``Collate``). In the latter case the target class is taken as ``argmax``.
    """

    def _compute(self, y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
        predictions = yhat.argmax(dim=-1)
        if y.shape == yhat.shape:  # one-hot / soft targets -> class indices
            y = y.argmax(dim=-1)
        return (predictions == y).float().mean()

    def __repr__(self) -> str:
        return "Accuracy"


class F1Score(Metric):
    def __init__(self, average="micro", threshold=0.5, epsilon=1e-7):
        self.average = average
        self.threshold = threshold
        self.epsilon = epsilon

    def _compute(self, y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
        y_prob = torch.sigmoid(yhat)
        y_pred = (y_prob > self.threshold).float()

        tp = y_pred * y
        fp = y_pred * (1 - y)
        fn = (1 - y_pred) * y

        if self.average == "micro":
            tp_sum = tp.sum()
            fp_sum = fp.sum()
            fn_sum = fn.sum()
            f1 = (2 * tp_sum) / (2 * tp_sum + fp_sum + fn_sum + self.epsilon)

        elif self.average == "macro":
            tp_sum = tp.sum(dim=0)
            fp_sum = fp.sum(dim=0)
            fn_sum = fn.sum(dim=0)
            f1_per_class = (2 * tp_sum) / (2 * tp_sum + fp_sum + fn_sum + self.epsilon)
            f1 = f1_per_class.mean()

        else:
            raise ValueError("Average must be 'micro' or 'macro'")

        return f1

    def __repr__(self) -> str:
        return f"F1-{self.average}"
