"""Tests for PatchImageVectorizer -- the (C, D) image path that keeps the patch axis.

The sibling of ImageVectorizer. Where that one pools every image to a single
vector, this one keeps the grid, so the cache lands on the same rung of the
tensor ladder as chunked text and the aggregators apply unchanged.
"""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from pydantic import model_validator
from transformers import BatchFeature

from vectormesh import PatchImageVectorizer

STUB_DIM = 4


class StubProcessor:
    """Stand-in for AutoImageProcessor: encodes a pixel value into pixel_values."""

    def __call__(self, images: list, return_tensors: str = "pt") -> BatchFeature:
        pixel_values = torch.stack(
            [torch.full((3, 8, 8), float(img.getpixel((0, 0))[0])) for img in images]
        )
        return BatchFeature({"pixel_values": pixel_values}, tensor_type=return_tensors)


class StubViT:
    """Stand-in for a ViT: -> last_hidden_state (b, prefix + grid**2, dim)."""

    def __init__(self, grid: int = 4, prefix: int = 1):
        self.grid = grid
        self.prefix = prefix

    def __call__(self, pixel_values: torch.Tensor):
        # Derive the output from the input's *content*, never from the batch size,
        # so that batching cannot change the result -- otherwise a batch-invariance
        # test would be testing the stub rather than the vectorizer.
        b = pixel_values.shape[0]
        seq = self.prefix + self.grid**2
        dev = pixel_values.device
        base = pixel_values.mean(dim=(1, 2, 3)).view(b, 1, 1) * 1000.0
        offsets = (
            torch.arange(seq, dtype=torch.float32, device=dev).view(1, seq, 1) * 10.0
        )
        dims = torch.arange(STUB_DIM, dtype=torch.float32, device=dev).view(
            1, 1, STUB_DIM
        )
        return SimpleNamespace(
            last_hidden_state=base + offsets + dims, pooler_output=None
        )


class StubCNNGrid:
    """Stand-in for a CNN: -> last_hidden_state (b, dim, h, w)."""

    def __init__(self, hw: int = 3):
        self.hw = hw

    def __call__(self, pixel_values: torch.Tensor):
        b = pixel_values.shape[0]
        hidden = torch.arange(b * STUB_DIM * self.hw**2, dtype=torch.float32).reshape(
            b, STUB_DIM, self.hw, self.hw
        )
        return SimpleNamespace(last_hidden_state=hidden, pooler_output=None)


def make_vectorizer(model, **kwargs) -> PatchImageVectorizer:
    """A PatchImageVectorizer with the network swapped for a stub."""

    class _Stubbed(PatchImageVectorizer):
        @model_validator(mode="after")
        def initialize_model(self):
            self._metadata = SimpleNamespace(hidden_size=STUB_DIM)
            self._processor = StubProcessor()
            self._model = model
            self._effective_max_length = None
            self._hidden_size = self._probe_dim()
            return self

    return _Stubbed(model_name="stub", **kwargs)


def test_vit_patch_axis_survives():
    v = make_vectorizer(StubViT(grid=4, prefix=1))
    images = [Image.new("RGB", (8, 8), (i, 0, 0)) for i in range(3)]

    out = v(images, batchsize=2)["patches"]

    assert len(out) == 3
    assert out[0].shape == (16, STUB_DIM), "a 4x4 grid is 16 patches, CLS dropped"
    assert v.get_grid == (4, 4)


def test_cnn_spatial_grid_is_flattened():
    v = make_vectorizer(StubCNNGrid(hw=3))

    out = v([Image.new("RGB", (8, 8), (1, 0, 0))], batchsize=1)["patches"]

    assert out[0].shape == (9, STUB_DIM), "(dim, 3, 3) -> (9, dim)"
    assert v.get_grid == (3, 3)


@pytest.mark.parametrize(
    "seq_len,expected",
    [
        (257, 1),  # DINOv2: 1 CLS + 16x16
        (256, 0),  # no CLS at all -- assuming 1 would eat a real patch
        (261, 5),  # 1 CLS + 4 register tokens + 16x16
        (197, 1),  # ViT-B/16 at 224: 1 CLS + 14x14
    ],
)
def test_prefix_tokens_are_inferred_not_assumed(seq_len, expected):
    """The count of non-patch tokens is found by looking for a square grid.

    Hardcoding 1 is wrong for a model with no CLS token (it silently drops a
    real patch, shifting every subsequent patch's position by one) and wrong
    for a model with register tokens.
    """
    v = make_vectorizer(StubViT(grid=4))
    assert v._prefix(seq_len) == expected


def test_explicit_prefix_overrides_inference():
    v = make_vectorizer(StubViT(grid=4), drop_prefix_tokens=0)
    assert v._prefix(257) == 0


def test_batching_does_not_change_the_result():
    images = [Image.new("RGB", (8, 8), (i, 0, 0)) for i in range(4)]
    one = make_vectorizer(StubViT(grid=4))(images, batchsize=1)["patches"]
    two = make_vectorizer(StubViT(grid=4))(images, batchsize=4)["patches"]

    assert len(one) == len(two) == 4
    for a, b in zip(one, two):
        assert torch.equal(a, b)


def test_cache_reads_this_as_rank_two():
    """VectorCache infers rank from the __call__ annotation, so it must say (C, D).

    This is why PatchImageVectorizer is a separate class rather than a
    `pool="none"` flag: a runtime flag cannot change the annotation, so the
    cache would record tensordtype=1 for rank-2 data.
    """
    from vectormesh.data.cache import VectorCache

    assert VectorCache.get_dtensor(make_vectorizer(StubViT(grid=4))) == 2


def test_metadata_records_the_grid():
    """The grid is needed to fold (C, D) back into a spatial map."""
    meta = make_vectorizer(StubViT(grid=4)).get_metadata

    assert meta["patch_grid"] == [4, 4]
    assert meta["hidden_size"] == STUB_DIM
    assert meta["col_name"] == "patches"


def test_fingerprint_separates_patch_from_pooled():
    """A patch cache and a pooled cache of the same model are different artefacts."""
    from vectormesh import ImageVectorizer

    class _StubbedPooled(ImageVectorizer):
        @model_validator(mode="after")
        def initialize_model(self):
            self._metadata = SimpleNamespace(hidden_size=STUB_DIM)
            self._processor = StubProcessor()
            self._model = StubViT(grid=4)
            self._effective_max_length = None
            self._hidden_size = STUB_DIM
            return self

    patched = make_vectorizer(StubViT(grid=4)).fingerprint_fields()
    pooled = _StubbedPooled(model_name="stub").fingerprint_fields()

    assert patched["class"] != pooled["class"]
