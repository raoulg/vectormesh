"""Tests for `VectorCache.join` and the hub loader.

Two caches built by different encoders over the same rows are the raw material for a
fusion pipeline, but they arrive as two independent downloads. The row orders *happen*
to match when both were built from the same shuffle -- which is exactly the kind of
coincidence that trains a model on mismatched pairs the day it stops holding. `join`
aligns on an explicit key and refuses when the alignment cannot be trusted.
"""

from pathlib import Path

import pytest
import torch
from datasets import Dataset

from vectormesh.data.cache import VectorCache
from vectormesh.types import VectorMeshError

DIM_A, DIM_B = 4, 3


def _meta(column: str, model_tag: str, hidden_size: int, columns: list[str]) -> dict:
    return {
        column: {
            "vectormesh_version": "test",
            "model_tag": model_tag,
            "vectorizer_type": "StubVectorizer",
            "tensordtype": 1,
            "hidden_size": hidden_size,
        },
        "features": columns,
        "created_at": "2026-01-01T00:00:00",
        "num_observations": len(columns),
    }


def make_cache(
    source_idx: list[int],
    labels: list[int],
    dim: int,
    model_tag: str = "org/stub-a",
    column: str = "embed",
    tmp_path: Path = Path("."),
) -> VectorCache:
    """A minimal cache: one vector column, a label, and a source_idx key."""
    rows = {
        "label": labels,
        "source_idx": source_idx,
        column: [[float(i)] * dim for i in source_idx],
    }
    dataset = Dataset.from_dict(rows)
    dataset.set_format(type="torch")
    return VectorCache(
        name=f"stub_{model_tag.split('/')[-1]}",
        cache_dir=tmp_path,
        dataset=dataset,
        metadata=_meta(column, model_tag, dim, list(rows)),
    )


@pytest.fixture
def left(tmp_path):
    return make_cache([7, 3, 9, 1], [0, 1, 0, 1], DIM_A, "org/dino", tmp_path=tmp_path)


def test_join_aligns_rows_by_key_not_position(tmp_path, left):
    """The whole point: `other` is in a different row order and must be reordered."""
    right = make_cache(
        [1, 9, 3, 7], [1, 0, 1, 0], DIM_B, "org/resnet-18", tmp_path=tmp_path
    )
    joined = left.join(right)

    assert joined.dataset.column_names == [
        "label",
        "source_idx",
        "embed",
        "embed_resnet-18",
    ]
    assert list(joined.dataset.with_format(None)["source_idx"]) == [7, 3, 9, 1]
    # the stub encodes source_idx into every element, so alignment is checkable
    for row in joined.dataset.with_format(None):
        assert row["embed_resnet-18"] == [float(row["source_idx"])] * DIM_B


def test_join_keeps_differing_widths(tmp_path, left):
    """Fusing a 4-dim and a 3-dim encoder is the ordinary case, not an error."""
    right = make_cache(
        [7, 3, 9, 1], [0, 1, 0, 1], DIM_B, "org/resnet-18", tmp_path=tmp_path
    )
    joined = left.join(right)
    item = joined.dataset[0]
    assert item["embed"].shape == (DIM_A,)
    assert item["embed_resnet-18"].shape == (DIM_B,)


def test_join_merges_metadata(tmp_path, left):
    right = make_cache(
        [7, 3, 9, 1], [0, 1, 0, 1], DIM_B, "org/resnet-18", tmp_path=tmp_path
    )
    joined = left.join(right)
    assert joined.metadata["embed"]["model_tag"] == "org/dino"
    assert joined.metadata["embed_resnet-18"]["model_tag"] == "org/resnet-18"
    assert joined.metadata["embed_resnet-18"]["hidden_size"] == DIM_B
    assert set(joined.vector_columns) == {"embed", "embed_resnet-18"}


def test_join_raises_on_mismatched_labels(tmp_path, left):
    """The guard that catches a genuinely wrong join rather than a merely odd one."""
    right = make_cache(
        [7, 3, 9, 1], [1, 1, 1, 1], DIM_B, "org/resnet-18", tmp_path=tmp_path
    )
    with pytest.raises(VectorMeshError, match="disagrees between the two caches"):
        left.join(right)


def test_join_raises_on_different_row_sets(tmp_path, left):
    right = make_cache(
        [7, 3, 9, 42], [0, 1, 0, 1], DIM_B, "org/resnet-18", tmp_path=tmp_path
    )
    with pytest.raises(VectorMeshError, match="do not cover the same rows"):
        left.join(right)


def test_join_raises_on_duplicate_keys(tmp_path, left):
    right = make_cache(
        [7, 7, 9, 1], [0, 0, 0, 1], DIM_B, "org/resnet-18", tmp_path=tmp_path
    )
    with pytest.raises(VectorMeshError, match="duplicate"):
        left.join(right)


def test_join_raises_when_key_column_absent(tmp_path, left):
    """Caches built before source_idx existed cannot be aligned, and must say so."""
    right = make_cache(
        [7, 3, 9, 1], [0, 1, 0, 1], DIM_B, "org/resnet-18", tmp_path=tmp_path
    )
    right = VectorCache(
        name=right.name,
        cache_dir=right.cache_dir,
        dataset=right.dataset.remove_columns("source_idx"),
        metadata=right.metadata,
    )
    with pytest.raises(VectorMeshError, match="no column 'source_idx'"):
        left.join(right)


def test_join_explicit_into_name(tmp_path, left):
    right = make_cache(
        [7, 3, 9, 1], [0, 1, 0, 1], DIM_B, "org/resnet-18", tmp_path=tmp_path
    )
    joined = left.join(right, into="second")
    assert "second" in joined.dataset.column_names
    assert joined.metadata["second"]["model_tag"] == "org/resnet-18"


def test_join_result_feeds_collate_parallel(tmp_path, left):
    """The reason join exists: the output is what CollateParallel consumes."""
    from torch.utils.data import DataLoader

    from vectormesh.components import Concatenate2D, NeuralNet, Parallel, Serial
    from vectormesh.data import CollateParallel, OneHot

    right = make_cache(
        [7, 3, 9, 1], [0, 1, 0, 1], DIM_B, "org/resnet-18", tmp_path=tmp_path
    )
    joined = left.join(right)
    data = joined.dataset.map(
        OneHot(num_classes=2, label_col="label", target_col="onehot")
    )
    data.set_format(type="torch")

    loader = DataLoader(
        data,
        batch_size=2,
        collate_fn=CollateParallel(
            vec1_col="embed",
            vec2_col="embed_resnet-18",
            target_col="onehot",
            padder=torch.stack,
        ),
    )
    pipeline = Serial(
        [
            Parallel([NeuralNet(DIM_A, 5), NeuralNet(DIM_B, 5)]),
            Concatenate2D(),
            NeuralNet(10, 2),
        ]
    )
    (x1, x2), y = next(iter(loader))
    assert x1.shape == (2, DIM_A) and x2.shape == (2, DIM_B)
    assert pipeline((x1, x2)).shape == (2, 2)


def test_concatenate2d_accepts_unequal_widths():
    """Regression: a *named* axis binds across a variadic tuple, so `"batch dim"`
    silently required every branch to be the same width -- rejecting the case the
    component exists for."""
    from vectormesh.components import Concatenate2D

    out = Concatenate2D()((torch.rand(3, 384), torch.rand(3, 512)))
    assert out.shape == (3, 896)


def test_concatenate3d_accepts_unequal_widths():
    from vectormesh.components import Concatenate3D

    out = Concatenate3D()((torch.rand(3, 5, 384), torch.rand(3, 5, 12)))
    assert out.shape == (3, 5, 396)


@pytest.mark.integration
def test_from_hub_round_trip():
    """Downloads a real published cache. Marked integration: needs the network."""
    cache = VectorCache.from_hub("pttrn-io/dtd-dinov2-small", split="test")
    assert len(cache) == 1880
    assert cache.metadata["embed"]["model_tag"] == "facebook/dinov2-small"
    assert cache.vector_columns == ["embed"]


@pytest.mark.integration
def test_from_hub_unknown_split_lists_available():
    with pytest.raises(VectorMeshError, match="has no split"):
        VectorCache.from_hub("pttrn-io/dtd-dinov2-small", split="validation")
