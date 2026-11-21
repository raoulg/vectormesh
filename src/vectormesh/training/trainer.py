import uuid
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

import torch
from jaxtyping import Float, Shaped
from loguru import logger
from torch.optim import Optimizer
from tqdm import tqdm

from .settings import TrainerSettings

OptimizerType = TypeVar("OptimizerType", bound=torch.optim.Optimizer)

# A batch input is either a single tensor or a tuple of tensors (e.g. packed
# sequences); the target is always a tensor.
BatchTensor = (
    Shaped[torch.Tensor, "batch ..."] | tuple[Shaped[torch.Tensor, "..."], ...]
)
#: the target of a batch, one row per example
TargetTensor = Shaped[torch.Tensor, "batch ..."]


@runtime_checkable
class Reporter(Protocol):
    """Anything shaped like ``report(epoch, train_loss, test_loss, metric_dict)``.

    This is a structural (``Protocol``) type. A bare
    function or lambda with this signature qualifies:

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


@runtime_checkable
class Step(Protocol):
    """Computes a scalar loss from a model, a batch, and its target.

    Used for both the training step and the validation step, so `test_loss`
    stays coherent for whatever consumes it downstream -- early stopping, the
    LR scheduler, every `Reporter`.

    Not given: `Trainer` builds one from `loss_fn` as
    ``lambda model, x, y: loss_fn(model(x), y)``, exactly today's single
    forward pass. Override when a loss needs more than one `model(x)` call to
    make sense -- a contrastive loss run on two augmented views, a VAE
    combining a reconstruction and a KL term. `metrics` still assume a plain
    `model(x)` produces a `(y, yhat)` pair worth scoring; pass `metrics=[]`
    when that is not true for your `step` (a contrastive pair usually has no
    such pair at all).
    """

    def __call__(
        self, model: torch.nn.Module, x: BatchTensor, y: TargetTensor
    ) -> Float[torch.Tensor, ""]: ...


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
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return log_dir / f"{timestamp}-{uuid.uuid4().hex[:6]}"


def _concat(batches: list[Any]) -> Any:
    """Concatenate per-batch outputs along the batch axis, also inside tuples."""
    first = batches[0]
    if isinstance(first, tuple):
        return tuple(_concat([b[i] for b in batches]) for i in range(len(first)))
    return torch.cat(batches, dim=0)


@dataclass
class TrainResult:
    """What one `Trainer.loop()` produced. Lets a caller -- a ray
    `trainable(config)`, a plain script, a notebook cell -- read the result
    directly rather than reaching into trainer internals.

    `epoch`, `train_loss`, `test_loss` and `metric_dict` describe the epoch
    whose model `trainer.model` holds after the loop: with early stopping on,
    the epoch with the lowest validation loss; without it, the last epoch.
    `stopped_epoch` is the last epoch that ran: `patience` epochs after
    `epoch` when early stopping stopped the loop, and equal to `epoch`
    whenever the last epoch was also the best one.
    """

    epoch: int
    train_loss: float
    test_loss: float
    metric_dict: dict[str, float] = field(default_factory=dict)
    stopped_epoch: int | None = None

    def __post_init__(self) -> None:
        if self.stopped_epoch is None:
            self.stopped_epoch = self.epoch


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
        step: Step | None = None,
        progress: bool = True,
    ) -> None:
        """
        Args:
            reporters: called once per epoch with
                ``(epoch, train_loss, test_loss, metric_dict)``. Anything
                matching that signature works -- see `Reporter`. Empty by
                default: a bare `Trainer` logs to loguru only and touches no
                logging framework.
            step: how to turn a batch into a scalar loss, for both training
                and validation. Defaults to ``loss_fn(model(x), y)`` -- see
                `Step`.
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
        # None means "use the default single-forward-pass path" -- kept as its
        # own flag (rather than always going through self.step) so evalbatches
        # can still share one model(x) call between the loss and the metrics
        # in the common case, and only pays for a second forward pass when a
        # custom step actually needs one.
        self._custom_step = step
        self.step: Step = step or (lambda model, x, y: loss_fn(model(x), y))
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
        # The epoch whose weights EarlyStopping.best_state holds; kept across
        # loop() calls, as EarlyStopping's own best loss is.
        self.best_result: TrainResult | None = None

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
        self, x: BatchTensor, y: TargetTensor
    ) -> tuple[BatchTensor, TargetTensor]:
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
    ) -> tuple[BatchTensor, TargetTensor]:
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
        """Train for `settings.epochs` epochs, or until early stopping stops.

        With early stopping on, the model's weights are copied in memory at
        every epoch that improves the validation loss, and after the loop
        `self.model` holds the weights of the best epoch, whether the loop
        stopped early or ran the full budget. The returned `TrainResult`
        describes that epoch, and its `stopped_epoch` the last one that ran.
        `earlystop_kwargs={"save": True}` also writes each improvement to
        `log_dir / "checkpoint.pt"`.

        Without early stopping, the model and the result are those of the
        last epoch.
        """
        epoch = 0
        result: TrainResult | None = None
        for epoch in self._bar(range(self.settings.epochs), colour="#1e4706"):
            train_loss = self.trainbatches()
            metric_dict, test_loss = self.evalbatches()
            resolved_epoch = self.report(epoch, train_loss, test_loss, metric_dict)
            result = TrainResult(resolved_epoch, train_loss, test_loss, metric_dict)

            if self.early_stopping is not None:
                self.early_stopping(test_loss, self.model)
                if self.early_stopping.improved:
                    self.best_result = result
                if self.early_stopping.early_stop:
                    logger.info("Interrupting loop due to early stopping patience.")
                    break
        self.last_epoch += epoch
        assert result is not None  # settings.epochs >= 1 is the caller's contract

        if self.early_stopping is None or self.best_result is None:
            return result
        assert self.early_stopping.best_state is not None
        self.model.load_state_dict(self.early_stopping.best_state)
        logger.info(
            f"Restored the model of epoch {self.best_result.epoch} "
            f"(test loss {self.best_result.test_loss:.4f}); "
            f"training stopped at epoch {result.epoch}."
        )
        return replace(self.best_result, stopped_epoch=result.epoch)

    def trainbatches(self) -> float:
        """Train for `settings.train_steps` batches.

        Returns the mean loss per example: each batch's loss (a mean over its
        examples) counts as many times as the batch has examples, so a short last
        batch weighs what its examples weigh.
        """
        self.model.train()
        loss_sum: float = 0.0
        examples = 0
        for _ in self._bar(range(self.settings.train_steps), colour="#1e4706"):
            x, y = self._next_batch(self.traindataloader, "_train_iter")
            x, y = self._to_device(x, y)
            self.optimizer.zero_grad()
            loss = self.step(self.model, x, y)
            loss.backward()
            self.optimizer.step()
            loss_sum += float(loss.cpu().detach()) * len(y)
            examples += len(y)
        return loss_sum / examples

    def evalbatches(self) -> tuple[dict[str, float], float]:
        """Evaluate the model on `settings.valid_steps` validation batches.

        The loss is the mean per example, weighted as in `trainbatches`. Every
        metric is computed once, on the targets and predictions of all validation
        batches concatenated, so it is exact also for metrics that are not a mean
        over examples (F1) and when the last batch is shorter.
        """
        self.model.eval()
        loss_sum: float = 0.0
        examples = 0
        targets: list[TargetTensor] = []
        predictions: list[Any] = []

        with torch.no_grad():
            for _ in range(self.settings.valid_steps):
                x, y = self._next_batch(self.validdataloader, "_valid_iter")
                x, y = self._to_device(x, y)

                if self._custom_step is None:
                    # one forward pass serves both the loss and the metrics
                    yhat = self.model(x)
                    loss = self.loss_fn(yhat, y)
                else:
                    loss = self.step(self.model, x, y)
                    # a custom step doesn't expose its own yhat, so metrics here
                    # cost a second forward pass, paid only when metrics are asked
                    yhat = self.model(x) if self.settings.metrics else None
                loss_sum += float(loss.cpu()) * len(y)
                examples += len(y)
                if self.settings.metrics:
                    targets.append(y)
                    predictions.append(yhat)

        test_loss = loss_sum / examples
        metric_dict: dict[str, float] = {}
        if self.settings.metrics:
            y_all, yhat_all = _concat(targets), _concat(predictions)
            for m in self.settings.metrics:
                metric_dict[str(m)] = m(y_all, yhat_all)

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
        """Track the validation loss and keep the weights of the best epoch.

        Every call that improves the best loss copies the model's
        `state_dict` into `best_state` (in memory, on the model's device) and
        sets `improved`; `patience` calls in a row without improvement set
        `early_stop`.

        Args:
            log_dir (Path): location to save checkpoint to.
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss
            improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as
            an improvement. Default: 0.0
            save (bool): If True, also write the whole model to
            `log_dir / "checkpoint.pt"` on every improvement, for `get_best`.
            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss: float | None = None
        self.best_state: dict[str, torch.Tensor] | None = None
        self.improved = False
        self.early_stop = False
        self.delta = delta
        self.path = Path(log_dir) / "checkpoint.pt"
        self.save = save

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        # we minimize loss: a loss that does not beat the best by more than
        # delta is not an improvement. The first call always is.
        self.improved = self.best_loss is None or val_loss + self.delta < self.best_loss
        if self.improved:
            self.best_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            if self.save:
                self.save_checkpoint(val_loss, model)
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                f"best loss: {self.best_loss:.4f}, current loss {val_loss:.4f}."
                f"Counter {self.counter}/{self.patience}."
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module) -> None:
        """Saves model when validation loss decrease."""
        if self.verbose:
            best = "inf" if self.best_loss is None else f"{self.best_loss:.4f}"
            logger.info(
                f"Validation loss ({best} --> {val_loss:.4f}). Saving {self.path} ..."
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
        """Load the checkpoint written with `save=True` from disk."""
        if self.verbose:
            logger.info(f"retrieving best model from {self.path}")
        return torch.load(self.path, weights_only=False)
