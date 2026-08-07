"""Tests for the training loop: Trainer, TrainerSettings, EarlyStopping, Reporter.

Covers:
  - constructing a Trainer touches no filesystem (log_dir is lazy)
  - a checkpoint save is the only thing that ever creates a directory
  - reporters are plain callables, called once per epoch, in order
  - Trainer.loop() returns a TrainResult
"""

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vectormesh.training import EarlyStopping, Trainer, TrainerSettings, TrainResult


def _dataloaders(
    n: int = 32, dim: int = 4, batch_size: int = 8
) -> tuple[DataLoader, DataLoader]:
    x = torch.randn(n, dim)
    y = torch.randint(0, 2, (n,))
    ds = TensorDataset(x, y)
    return (
        DataLoader(ds, batch_size=batch_size, shuffle=True),
        DataLoader(ds, batch_size=batch_size, shuffle=False),
    )


def _settings(tmp_path, **overrides: Any) -> TrainerSettings:
    train_dl, valid_dl = _dataloaders()
    kwargs: dict[str, Any] = dict(
        epochs=2,
        metrics=[],
        logdir=tmp_path,
        train_steps=len(train_dl),
        valid_steps=len(valid_dl),
        earlystop_kwargs=None,
        scheduler_kwargs=None,
    )
    kwargs.update(overrides)
    return TrainerSettings(**kwargs)


def _trainer(tmp_path, **trainer_kwargs: Any) -> Trainer:
    train_dl, valid_dl = _dataloaders()
    model = nn.Linear(4, 2)
    settings = _settings(tmp_path)
    trainer_kwargs.setdefault("progress", False)
    return Trainer(
        model=model,
        settings=settings,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        traindataloader=train_dl,
        validdataloader=valid_dl,
        **trainer_kwargs,
    )


def test_loop_returns_trainresult(tmp_path):
    trainer = _trainer(tmp_path)
    result = trainer.loop()
    assert isinstance(result, TrainResult)
    assert result.epoch == 1  # 2 epochs, 0-indexed -> last is 1
    assert isinstance(result.train_loss, float)
    assert isinstance(result.test_loss, float)


def test_settings_construction_touches_no_filesystem(tmp_path):
    logdir = tmp_path / "does-not-exist-yet"
    _settings(logdir.parent, logdir=logdir)
    assert not logdir.exists()


def test_trainer_construction_creates_no_directory(tmp_path):
    """Constructing a Trainer -- even inside a loop that builds one per ray
    trial -- must not touch disk on its own."""
    trainer = _trainer(tmp_path)
    assert not trainer.log_dir.exists()
    before = list(tmp_path.iterdir())
    trainer.loop()
    after = list(tmp_path.iterdir())
    # no early stopping, no reporters that write files -> nothing appeared
    assert before == after == []


def test_early_stopping_creates_directory_only_on_save(tmp_path):
    train_dl, valid_dl = _dataloaders()
    model = nn.Linear(4, 2)
    settings = _settings(
        tmp_path,
        earlystop_kwargs={"save": True, "verbose": False, "patience": 10},
    )
    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        traindataloader=train_dl,
        validdataloader=valid_dl,
        progress=False,
    )
    assert not trainer.log_dir.exists()  # save=True alone doesn't create it
    trainer.loop()
    assert trainer.log_dir.exists()  # first improving epoch did
    assert (trainer.log_dir / "checkpoint.pt").exists()


def test_reporters_called_once_per_epoch_with_resolved_epoch(tmp_path):
    calls = []

    def collect(epoch, train_loss, test_loss, metric_dict):
        calls.append((epoch, train_loss, test_loss, metric_dict))

    trainer = _trainer(tmp_path, reporters=[collect])
    trainer.loop()

    assert len(calls) == 2  # settings.epochs
    assert [c[0] for c in calls] == [0, 1]
    for _, train_loss, test_loss, metric_dict in calls:
        assert isinstance(train_loss, float)
        assert isinstance(test_loss, float)
        assert metric_dict == {}


def test_multiple_reporters_all_called_in_order(tmp_path):
    order = []
    r1 = lambda epoch, tl, vl, m: order.append("first")  # noqa: E731
    r2 = lambda epoch, tl, vl, m: order.append("second")  # noqa: E731

    trainer = _trainer(tmp_path, reporters=[r1, r2])
    trainer.loop()

    assert order == ["first", "second"] * 2  # 2 epochs


def test_reporter_protocol_accepts_a_bare_function(tmp_path):
    """No import from vectormesh, no subclassing -- structural typing only."""
    from vectormesh.training.trainer import Reporter

    def to_ray_shaped(
        epoch: int, train_loss: float, test_loss: float, metric_dict: dict[str, float]
    ) -> None:
        pass

    assert isinstance(to_ray_shaped, Reporter)


def test_no_reporters_is_the_default(tmp_path):
    trainer = _trainer(tmp_path)
    assert trainer.reporters == []


def test_progress_false_disables_tqdm_bars(tmp_path):
    """progress=False must hand back the raw iterable, not a tqdm wrapper --
    the whole point is zero overhead per trial, not a silenced tqdm."""
    trainer = _trainer(tmp_path)
    bar = trainer._bar(range(3), colour="#1e4706")
    assert type(bar) is range


def test_progress_true_wraps_in_tqdm(tmp_path):
    from tqdm import tqdm

    trainer = _trainer(tmp_path, progress=True)
    bar = trainer._bar(range(3))
    assert isinstance(bar, tqdm)


def test_early_stopping_lazy_directory(tmp_path):
    """EarlyStopping itself never mkdirs at construction -- only save_checkpoint does."""
    log_dir = tmp_path / "run"
    es = EarlyStopping(log_dir, patience=1, save=True)
    assert not log_dir.exists()
    es.save_checkpoint(0.5, nn.Linear(2, 2))
    assert log_dir.exists()
    assert es.path.exists()


def test_default_optimizer_kwargs_adds_no_regularisation(tmp_path):
    """optimizer_kwargs defaults to {} -- Adam gets its own PyTorch defaults
    (weight_decay=0), not an opinion this library used to bake in silently."""
    settings = _settings(tmp_path)
    assert settings.optimizer_kwargs == {}
    trainer = _trainer(tmp_path)
    assert trainer.optimizer.defaults["lr"] == 1e-3  # torch.optim.Adam's own default
    assert (
        trainer.optimizer.defaults["weight_decay"] == 0
    )  # torch.optim.Adam's own default
