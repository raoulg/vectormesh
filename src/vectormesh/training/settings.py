"""Training-run configuration.

Deliberately plain data: `TrainerSettings` is a pydantic model, not a place to
put callables that need a specific logging framework installed. Where to log
to (a `Reporter`) is a `Trainer` constructor argument, not a settings field --
see `trainer.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, field_validator


class FormattedBase(BaseModel):
    def __str__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

    def __repr__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())


class TrainerSettings(FormattedBase):
    epochs: int
    metrics: list[Callable]
    logdir: Path
    train_steps: int
    valid_steps: int
    # Empty by default: the optimizer class you pass keeps its own defaults
    # (torch.optim.Adam: lr=1e-3, weight_decay=0) unless overridden here.
    optimizer_kwargs: dict[str, Any] = {}
    # ReduceLROnPlateau's own PyTorch default, restated as the fallback for when
    # a scheduler is requested but no kwargs were given.
    scheduler_kwargs: dict[str, Any] | None = {"factor": 0.1, "patience": 10}
    # No default: whether training stops early changes the number you report,
    # non-obviously, so every Trainer states its choice explicitly. Pass
    # `None` to train the full epoch count, or a dict of EarlyStopping kwargs.
    earlystop_kwargs: dict[str, Any] | None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("logdir")
    @classmethod
    def check_path(cls, logdir: Path) -> Path:  # noqa: N805
        # Normalise only -- no filesystem access here. Directory creation is
        # lazy, deferred to whatever actually writes into it: see
        # trainer.py's _timestamped_dir / EarlyStopping.save_checkpoint.
        return Path(logdir)
