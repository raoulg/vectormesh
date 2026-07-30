"""Train a MoE where each chunk fuses its embedding with its chunk-aligned regex.

Mirrors scripts/train_moe.py, but instead of a single embedding tensor per chunk
the model concatenates the chunk-aligned regex vector onto each chunk's embedding
*before* projection + the MoE transformer, so every expert sees the regex signal
for that chunk.

Prerequisite: run scripts/add_chunked_regex.py first to create the extended cache
containing both the embedding column and the chunked_regex column.

Run:  uv run python scripts/train_moe_parallel.py
"""

from pathlib import Path

import torch
import torch.optim as optim
from loguru import logger
from mltrainer import ReportTypes, Trainer, TrainerSettings
from torch.utils.data import DataLoader

from vectormesh.components import (
    Concatenate3D,
    FixedPadding,
    MeanAggregator,
    MoE,
    Projection,
    Serial,
    TransformerBlock,
)
from vectormesh.components.metrics import F1Score
from vectormesh.data import CollateParallel, OneHot
from vectormesh.data.cache import VectorCache
from vectormesh.data.vectorizers import detect_device

EMB_COL = "legal_dutch"
REGEX_COL = "chunked_regex"
MAX_CHUNKS = 45
NUM_CLASSES = 32
HIDDEN_SIZE = 32
BATCH_SIZE = 64


def find_chunked_cache(artefacts: Path, split: str) -> Path:
    candidates = sorted(artefacts.glob(f"*bert*{split}*{REGEX_COL}"))
    if not candidates:
        raise FileNotFoundError(
            f"No *{REGEX_COL} cache for '{split}' in {artefacts}. "
            "Run scripts/add_chunked_regex.py first."
        )
    return candidates[-1]  # most recent


def get_dataloaders():
    artefacts = Path("artefacts")
    trainpath = find_chunked_cache(artefacts, "train")
    validpath = find_chunked_cache(artefacts, "valid")
    logger.info(f"Loading train cache {trainpath} and valid cache {validpath}")

    traincache = VectorCache.load(path=trainpath)
    validcache = VectorCache.load(path=validpath)
    assert traincache.dataset is not None and validcache.dataset is not None
    assert traincache.metadata is not None

    emb_dim = traincache.metadata[EMB_COL]["hidden_size"]
    regex_dim = traincache.metadata[REGEX_COL]["hidden_size"]
    logger.info(
        f"emb_dim={emb_dim}, regex_dim={regex_dim}, fused={emb_dim + regex_dim}"
    )

    onehot = OneHot(num_classes=NUM_CLASSES, label_col="labels", target_col="onehot")
    train_oh = traincache.dataset.map(onehot)
    valid_oh = validcache.dataset.map(onehot)

    # Both branches are chunked, so both get padded to MAX_CHUNKS and stay aligned.
    collate_fn = CollateParallel(
        vec1_col=EMB_COL,
        vec2_col=REGEX_COL,
        target_col="onehot",
        padder=FixedPadding(max_chunks=MAX_CHUNKS),
        padder2=FixedPadding(max_chunks=MAX_CHUNKS),
    )

    trainloader = DataLoader(
        train_oh, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    validloader = DataLoader(
        valid_oh, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    logger.success(
        f"Prepared {len(trainloader)} train / {len(validloader)} valid batches"
    )
    return trainloader, validloader, emb_dim, regex_dim


if __name__ == "__main__":
    trainloader, validloader, emb_dim, regex_dim = get_dataloaders()
    fused_dim = emb_dim + regex_dim

    moe = MoE(
        experts=[TransformerBlock(HIDDEN_SIZE, num_heads=2) for _ in range(2)],
        hidden_size=HIDDEN_SIZE,
        out_size=HIDDEN_SIZE,
    )

    pipeline = Serial(
        [
            Concatenate3D(),  # (X1, X2) -> (batch, chunks, emb + regex)
            Projection(fused_dim, HIDDEN_SIZE),  # -> (batch, chunks, hidden)
            moe,  # (batch, chunks, hidden) -> (batch, chunks, hidden)
            # Plain mean, not MaskedMean: the TransformerBlock attends with a padding
            # mask but still writes non-zero values at padded positions, so the
            # all-zero signature MaskedMeanAggregator looks for is already gone here.
            MeanAggregator(),  # (batch, chunks, hidden) -> (batch, hidden)
            torch.nn.Dropout(0.2),
            Projection(HIDDEN_SIZE, NUM_CLASSES),  # -> (batch, 32) logits
        ]
    )

    device = detect_device()
    log_dir = Path("logs/MoE_parallel").absolute()

    settings = TrainerSettings(
        epochs=150,
        metrics=[F1Score()],
        logdir=log_dir,
        train_steps=len(trainloader),
        valid_steps=len(validloader),
        reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.TOML],
        earlystop_kwargs={"save": True, "verbose": True, "patience": 40},
        scheduler_kwargs={"factor": 0.5, "patience": 20},
    )

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
    trainer.loop()
