"""Tests for a cache whose vector column is a fixed-shape `Array2D`.

A rank-2 vectorizer whose rows all share one shape, a `(steps, 1)` signal say, can have
its column written as one block of float32 per row instead of a list of lists. The caller
asks for it through `create(features=...)`; without that, or with `Sequence` features,
the column is the nested-list default. These tests hold both halves, and every path that
reads a cache's vector column: `load`, `get_or_create`, `join`, the hub round trip and a
DataLoader with `Collate`.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.ipc
import pytest
import torch
from beartype import beartype
from datasets import Array2D, Dataset, Features, Sequence, Value
from datasets.features.features import Array2DExtensionType
from jaxtyping import Float, jaxtyped
from pydantic import model_validator
from torch.utils.data import DataLoader

from vectormesh.components.padding import DynamicPadding
from vectormesh.data import hub
from vectormesh.data.cache import VectorCache
from vectormesh.data.dataset import Collate
from vectormesh.data.vectorizers import ChunkedVectorizer
from vectormesh.types import VectorMeshError

ROWS = 12
STEPS = 7


class RawSequenceVectorizer(ChunkedVectorizer):
    """Hand each row's steps over unchanged, as a float32 `(steps, 1)` tensor."""

    model_name: str = "raw"
    col_name: str = "beat"
    input_col: str = "steps"

    @model_validator(mode="after")
    def initialize_model(self):
        self._metadata = SimpleNamespace(hidden_size=1)
        self._effective_max_length = None
        return self

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, inputs: list, batchsize: int
    ) -> dict[str, list[Float[torch.Tensor, "_ dim"]]]:
        return {self.col_name: [torch.tensor(row).unsqueeze(-1) for row in inputs]}


def _steps(rows: int = ROWS, steps: int = STEPS) -> np.ndarray:
    return np.random.default_rng(0).random((rows, steps), dtype=np.float32)


def _source(steps: np.ndarray) -> Dataset:
    return Dataset.from_dict(
        {
            "steps": steps.tolist(),
            "label": [i % 2 for i in range(len(steps))],
            "source_idx": list(range(100, 100 + len(steps))),
        }
    )


def _fixed_features(steps: int = STEPS) -> Features:
    return Features(
        {
            "label": Value("int64"),
            "source_idx": Value("int64"),
            "beat": Array2D(shape=(steps, 1), dtype="float32"),
        }
    )


def _create(tmp_path: Path, steps: np.ndarray, features: Features | None, tag="fx"):
    return VectorCache.create(
        cache_dir=tmp_path,
        vectorizer=RawSequenceVectorizer(),
        dataset=_source(steps),
        dataset_tag=tag,
        features=features,
        remove_columns=["steps"],
        map_batch=5,
    )


def _arrow_type(folder: Path, column: str) -> pa.DataType:
    """The column's type as written in the cache's arrow file, read without `datasets`."""
    (arrow,) = folder.glob("data-*.arrow")
    with pa.memory_map(str(arrow)) as source:
        schema = pyarrow.ipc.open_stream(source).schema
    return schema.field(column).type


@pytest.fixture
def steps() -> np.ndarray:
    return _steps()


@pytest.fixture
def fixed(tmp_path: Path, steps: np.ndarray) -> VectorCache:
    return _create(tmp_path, steps, _fixed_features())


# --- the write ------------------------------------------------------------------


def test_fixed_shape_cache_loads_as_one_block(fixed: VectorCache, steps, tmp_path):
    loaded = VectorCache.load(tmp_path / fixed.name)

    assert loaded.features["beat"] == Array2D(shape=(STEPS, 1), dtype="float32")
    arrow = _arrow_type(tmp_path / fixed.name, "beat")
    assert isinstance(arrow, Array2DExtensionType)
    assert arrow.shape == (STEPS, 1)
    assert arrow.value_type == pa.float32()

    block = loaded[0:5]["beat"]
    assert block.shape == (5, STEPS, 1)
    assert block.dtype == torch.float32
    row = loaded[3]["beat"]
    assert row.shape == (STEPS, 1)
    assert row.dtype == torch.float32

    everything = loaded[:]["beat"]
    assert torch.equal(everything[..., 0], torch.from_numpy(steps))
    assert loaded[:]["source_idx"].tolist() == list(range(100, 100 + ROWS))


def test_asked_for_array2d_is_not_written_as_nested_lists(fixed: VectorCache, tmp_path):
    """The arrow file marks the column as one `(steps, 1)` block per row.

    A plain list type here is the nested-list layout, decoded row by row and step by
    step; the `Array2D` extension type is what lets a slice decode as one array.
    """
    arrow = _arrow_type(tmp_path / fixed.name, "beat")

    assert not pa.types.is_list(arrow), f"written as nested lists: {arrow}"
    assert isinstance(arrow, Array2DExtensionType), f"not an Array2D column: {arrow}"


def test_fixed_shape_metadata(fixed: VectorCache, tmp_path):
    meta = VectorCache.load(tmp_path / fixed.name).metadata["beat"]

    assert meta["tensordtype"] == 2
    assert meta["hidden_size"] == 1
    assert meta["chunk_sizes"] == {str(STEPS): ROWS}
    assert fixed.metadata["beat"]["chunk_sizes"] == {STEPS: ROWS}


def test_a_row_of_another_shape_is_refused(tmp_path, steps):
    with pytest.raises(VectorMeshError, match="Failed to create cache") as info:
        _create(tmp_path, steps, _fixed_features(steps=STEPS + 1))

    assert isinstance(info.value.__cause__, VectorMeshError)
    assert "shape (8, 1)" in str(info.value.__cause__)
    assert "(7, 1)" in str(info.value.__cause__)


# --- the default stays the default ----------------------------------------------


def test_without_features_the_column_is_nested_lists(tmp_path, steps):
    cache = _create(tmp_path, steps, None)

    assert cache.features["beat"] == Sequence(Sequence(Value("float32")))
    assert _arrow_type(tmp_path / cache.name, "beat") == pa.list_(
        pa.list_(pa.float32())
    )
    assert torch.equal(cache[0:ROWS]["beat"][..., 0], torch.from_numpy(steps))


def test_sequence_features_write_what_the_default_writes(tmp_path, steps):
    default = _create(tmp_path / "default", steps, None)
    explicit = _create(
        tmp_path / "explicit",
        steps,
        VectorCache.get_features(_source(steps), 2, embedding_column="beat"),
    )

    assert explicit.features == default.features
    assert explicit.dataset._fingerprint == default.dataset._fingerprint
    assert torch.equal(explicit[:]["beat"], default[:]["beat"])


def test_ragged_rows_keep_the_nested_layout(tmp_path):
    lengths = [3, 5, 5, 2]
    rows = [np.arange(n, dtype=np.float32).tolist() for n in lengths]
    dataset = Dataset.from_dict({"steps": rows, "label": [0, 1, 0, 1]})

    cache = VectorCache.create(
        cache_dir=tmp_path,
        vectorizer=RawSequenceVectorizer(),
        dataset=dataset,
        remove_columns=["steps"],
    )

    assert cache.features["beat"] == Sequence(Sequence(Value("float32")))
    assert [cache[i]["beat"].shape[0] for i in range(4)] == lengths
    assert cache.metadata["beat"]["chunk_sizes"] == {3: 1, 5: 2, 2: 1}


def test_the_fingerprint_names_the_layout(tmp_path, steps):
    """Two layouts of the same rows are two different map results."""
    default = _create(tmp_path / "default", steps, None)
    fixed = _create(tmp_path / "fixed", steps, _fixed_features())

    assert default.dataset._fingerprint != fixed.dataset._fingerprint


# --- the paths that read a cache --------------------------------------------------


def test_get_or_create_serves_a_fixed_shape_cache(fixed: VectorCache, tmp_path):
    vectorizer = RawSequenceVectorizer()
    built = VectorCache.create(
        cache_dir=tmp_path / "goc",
        vectorizer=vectorizer,
        dataset=_source(_steps()),
        dataset_tag=f"heartbeat_{VectorCache._model_slug(vectorizer)}",
        features=_fixed_features(),
        remove_columns=["steps"],
    )

    def never() -> Dataset:
        raise AssertionError("a cache hit must not build the dataset")

    hit = VectorCache.get_or_create(tmp_path / "goc", "heartbeat", vectorizer, never)

    assert hit.name == built.name
    assert hit.features["beat"] == Array2D(shape=(STEPS, 1), dtype="float32")
    assert hit[0:3]["beat"].shape == (3, STEPS, 1)


def test_hub_round_trip_keeps_the_fixed_shape(
    fixed: VectorCache, tmp_path, monkeypatch
):
    upload = fixed.push_to_hub("acme/heartbeat-raw", dry_run=True, write_card=False)
    assert "data-00000-of-00001.arrow" in upload.files

    local = tmp_path / "snapshot"
    hub.stage_payload(fixed.dataset.with_format(None), fixed.metadata, local / "train")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: local)
    pulled = VectorCache.from_hub("acme/heartbeat-raw")

    assert pulled.features["beat"] == Array2D(shape=(STEPS, 1), dtype="float32")
    assert torch.equal(pulled[:]["beat"], fixed[:]["beat"])
    assert pulled.metadata["beat"]["chunk_sizes"] == {str(STEPS): ROWS}


def test_collate_batches_a_fixed_shape_cache(fixed: VectorCache, steps):
    collate = Collate(embedding_col="beat", target_col="label", padder=DynamicPadding())
    loader = DataLoader(fixed, batch_size=5, collate_fn=collate)

    X, y = next(iter(loader))

    assert X.shape == (5, STEPS, 1)
    assert X.dtype == torch.float32
    assert torch.equal(X[..., 0], torch.from_numpy(steps[:5]))
    assert y.tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]
