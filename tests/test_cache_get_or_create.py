"""Tests for `VectorCache.get_or_create`, promoted from the course's
`notebooks/5_attention/cache_utils.py` (main_principles.md P2).
"""

import pytest
import torch
from datasets import Dataset
from jaxtyping import Float
from torch import Tensor

from vectormesh.data.cache import VectorCache
from vectormesh.types import VectorMeshError


class _StubVectorizer:
    """Minimal vectorizer: doubles a scalar 'value' column into a 1-wide embedding."""

    col_name = "embed"
    input_col = "value"
    get_hidden_size = 1
    get_context_size = None

    def __init__(self, model_name: str = "org/stub-a"):
        self.model_name = model_name

    def __call__(
        self, batch, batchsize: int = 32
    ) -> "dict[str, list[Float[Tensor, 'dim']]]":
        # "dim" (one name, not "_ dim") -> get_dtensor reads this as rank 1: an item is
        # a plain (D,) vector, matching an image encoder's output rather than a chunked
        # text vectorizer's (C, D).
        return {"embed": [torch.tensor([float(v) * 2]) for v in batch]}

    def fingerprint_fields(self) -> dict:
        return {"model_name": self.model_name}


def _dataset() -> Dataset:
    return Dataset.from_dict({"value": [1, 2, 3]})


def test_builds_when_no_cache_exists(tmp_path):
    vec = _StubVectorizer()
    cache = VectorCache.get_or_create(tmp_path, "mytag", vec, _dataset())
    assert len(cache) == 3
    assert cache.metadata["embed"]["model_tag"] == "org/stub-a"


def test_second_call_loads_instead_of_rebuilding(tmp_path, monkeypatch):
    vec = _StubVectorizer()
    VectorCache.get_or_create(tmp_path, "mytag", vec, _dataset())

    calls = []
    original_create = VectorCache.create.__func__

    def spy_create(cls, *args, **kwargs):
        calls.append(1)
        return original_create(cls, *args, **kwargs)

    monkeypatch.setattr(VectorCache, "create", classmethod(spy_create))
    cache = VectorCache.get_or_create(tmp_path, "mytag", vec, _dataset())
    assert len(cache) == 3
    assert calls == []  # loaded from disk, never rebuilt


def test_lazy_dataset_callable_not_invoked_on_a_cache_hit(tmp_path):
    vec = _StubVectorizer()
    VectorCache.get_or_create(tmp_path, "mytag", vec, _dataset())

    def boom():
        raise AssertionError("dataset builder should not run on a cache hit")

    cache = VectorCache.get_or_create(tmp_path, "mytag", vec, boom)
    assert len(cache) == 3


def test_raises_on_model_tag_mismatch(tmp_path):
    # Same slug ("stub" = model_name.split("/")[-1]), different full model_name --
    # the case the encoder-in-the-tag naming alone cannot catch, per the docstring.
    VectorCache.get_or_create(
        tmp_path, "mytag", _StubVectorizer("org-one/stub"), _dataset()
    )
    with pytest.raises(VectorMeshError, match="was built with model_tag"):
        VectorCache.get_or_create(
            tmp_path, "mytag", _StubVectorizer("org-two/stub"), _dataset()
        )
