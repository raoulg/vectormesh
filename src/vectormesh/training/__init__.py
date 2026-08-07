"""VectorMesh's training loop.

`Trainer` + `TrainerSettings` + `EarlyStopping` are the whole surface; logging
to mlflow, ray, tensorboard, or anything else is wired in via `Reporter` --
a plain callable, not a base class to subclass and not a dependency vectormesh
carries. See `docs/10-reporters.md` for adapters.
"""

from .settings import TrainerSettings
from .trainer import EarlyStopping, Reporter, Trainer, TrainResult

__all__ = [
    "Trainer",
    "TrainerSettings",
    "EarlyStopping",
    "Reporter",
    "TrainResult",
]
