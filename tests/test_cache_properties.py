"""Tests for `VectorCache.num_classes` and `VectorCache.dim`.

Both are shortcuts for boilerplate every lesson-1 notebook currently re-derives by hand:
`cache.features["label"].num_classes` and `cache.metadata[<col>]["hidden_size"]`.
"""

from pathlib import Path

import pytest
from datasets import ClassLabel, Dataset, Features, Sequence, Value

from vectormesh.data.cache import VectorCache
from vectormesh.types import VectorMeshError


def _meta(column: str, hidden_size: int, columns: list[str]) -> dict:
    return {
        column: {
            "vectormesh_version": "test",
            "model_tag": "org/stub",
            "vectorizer_type": "StubVectorizer",
            "tensordtype": 1,
            "hidden_size": hidden_size,
        },
        "features": columns,
        "created_at": "2026-01-01T00:00:00",
        "num_observations": len(columns),
    }


def _labelled_cache(tmp_path: Path, dim: int = 4, num_classes: int = 5) -> VectorCache:
    features = Features(
        {
            "label": ClassLabel(num_classes=num_classes),
            "embed": Sequence(Value("float32")),
        }
    )
    dataset = Dataset.from_dict(
        {"label": [0, 1, 2], "embed": [[0.0] * dim] * 3}, features=features
    )
    dataset.set_format(type="torch")
    return VectorCache(
        name="stub",
        cache_dir=tmp_path,
        dataset=dataset,
        metadata=_meta("embed", dim, ["label", "embed"]),
    )


def test_num_classes_reads_the_classlabel(tmp_path):
    cache = _labelled_cache(tmp_path, num_classes=7)
    assert cache.num_classes == 7


def test_num_classes_raises_without_a_label_column(tmp_path):
    dataset = Dataset.from_dict({"embed": [[0.0, 1.0]]})
    dataset.set_format(type="torch")
    cache = VectorCache(
        name="stub",
        cache_dir=tmp_path,
        dataset=dataset,
        metadata=_meta("embed", 2, ["embed"]),
    )
    with pytest.raises(VectorMeshError, match="no 'label' column"):
        cache.num_classes


def test_num_classes_raises_when_label_is_not_a_classlabel(tmp_path):
    features = Features({"label": Value("int64"), "embed": Sequence(Value("float32"))})
    dataset = Dataset.from_dict(
        {"label": [0, 1], "embed": [[0.0], [1.0]]}, features=features
    )
    dataset.set_format(type="torch")
    cache = VectorCache(
        name="stub",
        cache_dir=tmp_path,
        dataset=dataset,
        metadata=_meta("embed", 1, ["label", "embed"]),
    )
    with pytest.raises(VectorMeshError, match="not a ClassLabel"):
        cache.num_classes


def test_dim_reads_hidden_size(tmp_path):
    cache = _labelled_cache(tmp_path, dim=384)
    assert cache.dim == 384


def test_dim_raises_when_ambiguous_after_join(tmp_path):
    cache = _labelled_cache(tmp_path, dim=4)
    # simulate the post-join() state directly: two vector columns in metadata
    two_col_metadata = dict(cache.metadata)
    two_col_metadata["embed_other"] = dict(two_col_metadata["embed"])
    cache = VectorCache(
        name=cache.name,
        cache_dir=cache.cache_dir,
        dataset=cache.dataset,
        metadata=two_col_metadata,
    )
    with pytest.raises(VectorMeshError, match="cannot guess which column"):
        cache.dim
