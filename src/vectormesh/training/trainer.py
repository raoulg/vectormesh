from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable

import torch
from loguru import logger
from torch.optim import Optimizer
from tqdm import tqdm

from .settings import TrainerSettings

OptimizerType = TypeVar("OptimizerType", bound=torch.optim.Optimizer)

# A batch input is either a single tensor or a tuple of tensors (e.g. packed
# sequences); the target is always a tensor.
BatchTensor = torch.Tensor | tuple[torch.Tensor, ...]


@runtime_checkable
class Reporter(Protocol):
    """Anything shaped like ``report(epoch, train_loss, test_loss, metric_dict)``.

    This is a structural (``Protocol``) type, not an ABC -- there is nothing to
    subclass and nothing to import from vectormesh to satisfy it. A bare
    function or lambda with this signature already qualifies:

    >>> def to_ray(epoch, train_loss, test_loss, metric_dict):
    ...     from ray import tune
    ...     tune.report({"train_loss": train_loss, "test_loss": test_loss, **metric_dict})

    vectormesh ships zero framework-specific reporters -- no mlflow, ray, or
    tensorboard import anywhere in this package -- so using any of them costs
    you exactly their own dependency, never vectormesh's. See
    ``docs/10-reporters.md`` for copy-paste adapters (mlflow, ray, tensorboard,
    a plain webhook).
    """

    def __call__(
        self,
        epoch: int,
        train_loss: float,
        test_loss: float,
        metric_dict: dict[str, float],
    ) -> None: ...


def _timestamped_dir(log_dir: Path | None = None) -> Path:
    """Compute -- but do not create -- a per-run subdirectory path.

    Nothing here touches the filesystem, so a `Trainer` that never ends up
    writing anything (the common case for a ray trial with early stopping
    disabled) pays nothing for it -- see `EarlyStopping.save_checkpoint` for
    the one place a directory actually gets created.

    A short random suffix, not just the second-resolution timestamp, keeps two
    trials that start in the same wall-clock second (routine under ray, which
    launches trials in parallel processes) from computing the *same* path.
    """
    log_dir = Path(log_dir) if log_dir is not None else Path(".")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return log_dir / f"{timestamp}-{uuid.uuid4().hex[:6]}"


@dataclass
class TrainResult:
    """What one `Trainer.loop()` produced: the final epoch's `train_loss`,
    `test_loss` and `metric_dict`. Lets a caller -- a ray `trainable(config)`,
    a plain script, a notebook cell -- read the result directly rather than
    reaching into trainer internals.
    """

    epoch: int
    train_loss: float
    test_loss: float
    metric_dict: dict[str, float] = field(default_factory=dict)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        settings: TrainerSettings,
        loss_fn: Callable,
        optimizer: type[Optimizer],
        traindataloader: Iterable,
        validdataloader: Iterable,
        scheduler: Callable | None = None,
        device: str | None = None,
        reporters: Sequence[Reporter] = (),
        progress: bool = True,
    ) -> None:
        """
        Args:
            reporters: called once per epoch with
                ``(epoch, train_loss, test_loss, metric_dict)``. Anything
                matching that signature works -- see `Reporter`. Empty by
                default: a bare `Trainer` logs to loguru only and touches no
                logging framework.
            progress: show `tqdm` bars for epochs and training batches. Set to
                `False` inside a ray trial (or any loop that runs many
                `Trainer`s back to back) -- ray's own status table already
                shows progress across trials, and nested per-trial bars are
                pure overhead there.
        """
        self.model = model
        self.settings = settings
        # Computed, not created -- see _timestamped_dir. The directory shows
        # up on disk the first time something actually writes into it
        # (currently: EarlyStopping.save_checkpoint).
        self.log_dir = _timestamped_dir(settings.logdir)
        self.loss_fn = loss_fn
        self.traindataloader = traindataloader
        self.validdataloader = validdataloader
        self._train_iter: Iterator | None = None
        self._valid_iter: Iterator | None = None
        self.device = device
        self.progress = progress
        self.reporters = list(reporters)

        if self.device:
            self.model.to(self.device)

        self.optimizer = optimizer(model.parameters(), **settings.optimizer_kwargs)
        self.last_epoch = 0

        if scheduler:
            if settings.scheduler_kwargs is None:
                raise ValueError("Missing 'scheduler_kwargs' in TrainerSettings.")
            self.scheduler = scheduler(self.optimizer, **settings.scheduler_kwargs)
        else:
            self.scheduler = None

        if settings.earlystop_kwargs is not None:
            logger.info(
                "Found earlystop_kwargs in settings."
                "Set to None if you dont want earlystopping."
            )
            self.early_stopping: EarlyStopping | None = EarlyStopping(
                self.log_dir, **settings.earlystop_kwargs
            )
        else:
            self.early_stopping = None

    def _bar(self, iterable: Iterable, **kwargs) -> Iterable:
        """`tqdm(iterable, **kwargs)` when `self.progress`, else `iterable` untouched."""
        return tqdm(iterable, **kwargs) if self.progress else iterable

    def _to_device(
        self, x: BatchTensor, y: torch.Tensor
    ) -> tuple[BatchTensor, torch.Tensor]:
        """Move a batch onto self.device, handling tuple inputs (e.g. packed
        sequences) as well as single tensors."""
        if not self.device:
            return x, y
        if isinstance(x, tuple):
            x = tuple(xx.to(self.device) for xx in x)
        else:
            x = x.to(self.device)
        return x, y.to(self.device)

    def _next_batch(
        self, loader: Iterable, cache: str
    ) -> tuple[BatchTensor, torch.Tensor]:
        """Pull one batch, keeping the iterator alive across steps.

        `next(iter(loader))` per step is only correct for an *iterator* -- a
        streamer -- because `iter()` on an iterator returns itself. A torch
        DataLoader is an *iterable*: it hands out a fresh iterator (and, with
        num_workers > 0, a fresh worker pool) on every call, so a training
        loader would reshuffle and replay its first batch every step, and a
        validation loader would score the same first batch `valid_steps` times.

        Holding the iterator and re-`iter`ing it only when it is exhausted
        works for both: an infinite streamer never raises StopIteration, and a
        finite DataLoader wraps around into its next epoch.
        """
        it = getattr(self, cache)
        if it is None:
            it = iter(loader)
            setattr(self, cache, it)
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            setattr(self, cache, it)
            return next(it)

    def loop(self) -> TrainResult:
        epoch = 0
        result: TrainResult | None = None
        for epoch in self._bar(range(self.settings.epochs), colour="#1e4706"):
            train_loss = self.trainbatches()
            metric_dict, test_loss = self.evalbatches()
            resolved_epoch = self.report(epoch, train_loss, test_loss, metric_dict)
            result = TrainResult(resolved_epoch, train_loss, test_loss, metric_dict)

            if self.early_stopping:
                self.early_stopping(test_loss, self.model)

            if self.early_stopping is not None and self.early_stopping.early_stop:
                logger.info("Interrupting loop due to early stopping patience.")
                self.last_epoch = epoch
                if self.early_stopping.save:
                    self.model = self.early_stopping.get_best()
                else:
                    logger.info(
                        "early_stopping_save was false, using latest model."
                        "Set to true to retrieve best model."
                    )
                break
        self.last_epoch += epoch
        assert result is not None  # settings.epochs >= 1 is the caller's contract
        return result

    def trainbatches(self) -> float:
        self.model.train()
        train_loss: float = 0.0
        train_steps = self.settings.train_steps
        for _ in self._bar(range(train_steps), colour="#1e4706"):
            x, y = self._next_batch(self.traindataloader, "_train_iter")
            x, y = self._to_device(x, y)
            self.optimizer.zero_grad()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            loss.backward()
            self.optimizer.step()
            train_loss += float(loss.cpu().detach())
        train_loss /= train_steps

        return train_loss

    def evalbatches(self) -> tuple[dict[str, float], float]:
        """Evaluate model on validation data with proper device handling"""
        self.model.eval()
        valid_steps = self.settings.valid_steps
        test_loss: float = 0.0
        metric_dict: dict[str, float] = {}

        with torch.no_grad():  # Prevent gradient computation during evaluation
            for _ in range(valid_steps):
                x, y = self._next_batch(self.validdataloader, "_valid_iter")
                x, y = self._to_device(x, y)

                # Forward pass
                yhat = self.model(x)

                # Calculate loss (already handled by loss_fn)
                test_loss += float(self.loss_fn(yhat, y).cpu())

                # Calculate metrics (now handled by metric classes)
                for m in self.settings.metrics:
                    metric_dict[str(m)] = metric_dict.get(str(m), 0.0) + m(y, yhat)

        # Average the results
        test_loss /= valid_steps
        for key in metric_dict:
            metric_dict[key] = metric_dict[key] / valid_steps

        # Handle scheduler
        if self.scheduler:
            if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                self.scheduler.step(test_loss)
            else:
                self.scheduler.step()

        return metric_dict, test_loss

    def _resolve_epoch(self, epoch: int) -> int:
        """Offset the epoch counter when resuming a previous training run."""
        if self.last_epoch != 0 and epoch == 0:
            self.last_epoch += 1
            logger.info(f"Resuming epochs from previous training at {self.last_epoch}")
        if self.last_epoch != 0:
            epoch += self.last_epoch
        return epoch

    def report(
        self,
        epoch: int,
        train_loss: float,
        test_loss: float,
        metric_dict: dict[str, float],
    ) -> int:
        """Resolve the epoch, log it, and call every reporter. Returns the
        resolved epoch, so `loop()` can hand it straight to `TrainResult`."""
        epoch = self._resolve_epoch(epoch)
        self.test_loss = test_loss

        for reporter in self.reporters:
            reporter(epoch, train_loss, test_loss, metric_dict)

        metric_scores = [f"{v:.4f}" for v in metric_dict.values()]
        logger.info(
            f"Epoch {epoch} train {train_loss:.4f} test {test_loss:.4f} "
            f"metric {metric_scores}"
        )
        return epoch


class EarlyStopping:
    def __init__(
        self,
        log_dir: Path,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0.0,
        save: bool = False,
    ) -> None:
        """
        Args:
            log_dir (Path): location to save checkpoint to.
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss
            improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as
            an improvement. Default: 0.0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.path = Path(log_dir) / "checkpoint.pt"
        self.save = save

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        # first epoch best_loss is still None
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.save:
                self.save_checkpoint(val_loss, model)
        elif val_loss + self.delta >= self.best_loss:
            # we minimize loss. If current loss did not improve
            # the previous best (with a delta) it is considered not to improve.
            self.counter += 1
            logger.info(
                f"best loss: {self.best_loss:.4f}, current loss {val_loss:.4f}."
                f"Counter {self.counter}/{self.patience}."
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # if not the first run, and val_loss is smaller, we improved.
            if self.save:
                self.save_checkpoint(val_loss, model)
            self.best_loss = val_loss
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module) -> None:
        """Saves model when validation loss decrease."""
        if self.verbose:
            logger.info(
                f"Validation loss ({self.best_loss:.4f} --> {val_loss:.4f})."
                f"Saving {self.path} ..."
            )
        # The only place a Trainer run creates a directory on disk, and only
        # when save=True and a checkpoint is actually about to be written --
        # never as a side effect of just constructing a Trainer or an
        # EarlyStopping. exist_ok=True: two ray trials with save=True could in
        # principle share a colliding log_dir second, though _timestamped_dir's
        # random suffix already makes that vanishingly unlikely.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, self.path)
        self.val_loss_min = val_loss

    def get_best(self) -> torch.nn.Module:
        if self.verbose:
            logger.info(f"retrieving best model from {self.path}")
        return torch.load(self.path, weights_only=False)
