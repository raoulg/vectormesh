from pathlib import Path

from vectormesh.data.cache import VectorCache
from vectormesh.data import OneHot
from torch.utils.data import DataLoader

from vectormesh.components import FixedPadding
from vectormesh.data import Collate
from vectormesh.components import (
    MeanAggregator,
    Projection,
    Serial,
    MoE,
    TransformerBlock,
)
import torch
import torch.optim as optim

from vectormesh.components.metrics import F1Score
from vectormesh.training import Trainer, TrainerSettings
from vectormesh.data.vectorizers import detect_device
from loguru import logger


def get_dataloaders():
    artefacts = Path("artefacts")
    trainpath = next(artefacts.glob("*bert*train/"))
    validpath = next(artefacts.glob("*bert*valid/"))
    logger.info(
        f"Loading train cache from {trainpath} and valid cache from {validpath}"
    )
    traincache = VectorCache.load(path=trainpath)
    validcache = VectorCache.load(path=validpath)
    assert traincache.dataset is not None and validcache.dataset is not None

    onehot = OneHot(num_classes=32, label_col="labels", target_col="onehot")
    train_oh = traincache.dataset.map(onehot)
    valid_oh = validcache.dataset.map(onehot)
    collate_fn = Collate(
        embedding_col="legal_dutch",
        target_col="onehot",
        padder=FixedPadding(max_chunks=45),
    )

    trainloader = DataLoader(
        train_oh, batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    validloader = DataLoader(
        valid_oh, batch_size=32, shuffle=False, collate_fn=collate_fn
    )
    logger.success(
        f"Prepared dataloaders with {len(trainloader)} train / {len(validloader)} valid batches"
    )

    return trainloader, validloader


if __name__ == "__main__":
    trainloader, validloader = get_dataloaders()

    hidden_size = 32
    moe = MoE(
        experts=[TransformerBlock(hidden_size, num_heads=2) for _ in range(4)],
        hidden_size=hidden_size,
        out_size=hidden_size,
    )

    pipeline = Serial(
        [
            Projection(
                hidden_size=768, out_size=hidden_size
            ),  # Project to smaller hidden size
            moe,  # (batch, 30, d) -> (batch, 30, d)
            MeanAggregator(),  # -> (batch, d)
            Projection(hidden_size, 32),  # -> (batch, 32) logits
        ]
    )

    device = detect_device()
    log_dir = Path("logs/MoE").absolute()

    settings = TrainerSettings(
        epochs=100,
        metrics=[F1Score()],
        logdir=log_dir,
        train_steps=len(trainloader),
        valid_steps=len(validloader),
        earlystop_kwargs={"save": True, "verbose": True, "patience": 20},
        scheduler_kwargs={"factor": 0.5, "patience": 10},
    )

    # SoftMoE needs no auxiliary balancing loss, so a plain task loss suffices.
    loss_fn = torch.nn.BCEWithLogitsLoss()

    trainer = Trainer(
        model=pipeline,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optim.Adam,
        traindataloader=trainloader,
        validdataloader=validloader,
        scheduler=optim.lr_scheduler.ReduceLROnPlateau,
        device=device,
    )

    # TensorBoard is not a vectormesh dependency -- wiring it in is this one
    # closure, added after construction so it can point at the exact run
    # directory Trainer already computed (trainer.log_dir). Same pattern for
    # mlflow/ray/wandb: see docs/10-reporters.md.
    from torch.utils.tensorboard.writer import SummaryWriter

    writer = SummaryWriter(log_dir=trainer.log_dir)

    def to_tensorboard(epoch, train_loss, test_loss, metric_dict):
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        for name, value in metric_dict.items():
            writer.add_scalar(f"metric/{name}", value, epoch)

    trainer.reporters.append(to_tensorboard)

    trainer.loop()
    writer.close()
