"""Tests for the image embedding path (ImageVectorizer)."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from pydantic import model_validator
from transformers import BatchFeature

from vectormesh import ImageVectorizer

STUB_DIM = 4


class StubProcessor:
    """Stand-in for AutoImageProcessor.

    Encodes each image's pixel value into `pixel_values` so a test can check
    *which* image ended up in *which* output slot, and records the size of every
    batch it was handed.
    """

    def __init__(self):
        self.batch_sizes: list[int] = []

    def __call__(self, images: list, return_tensors: str = "pt") -> BatchFeature:
        self.batch_sizes.append(len(images))
        # .convert("RGB") happened upstream, so getpixel returns an (r, g, b) tuple.
        pixel_values = torch.stack(
            [torch.full((3, 8, 8), float(img.getpixel((0, 0))[0])) for img in images]
        )
        return BatchFeature({"pixel_values": pixel_values}, tensor_type=return_tensors)


class StubModel:
    """Stand-in for a CNN vision model: (b, 3, 8, 8) -> pooler_output (b, dim, 1, 1)."""

    def __call__(self, pixel_values: torch.Tensor):
        pooled = pixel_values.mean(dim=(1, 2, 3), keepdim=False)
        pooled = pooled[:, None].repeat(1, STUB_DIM)
        return SimpleNamespace(pooler_output=pooled[..., None, None])


class StubImageVectorizer(ImageVectorizer):
    """ImageVectorizer with the HuggingFace download swapped out for stubs."""

    @model_validator(mode="after")
    def initialize_model(self):
        self._metadata = None
        self._processor = StubProcessor()
        self._model = StubModel()
        self._hidden_size = STUB_DIM
        self._effective_max_length = None
        return self


def test_image_vectorizer_multiple_batches():
    """More images than batchsize: every image gets its own vector, in order.

    Regression test -- __call__ used to rebind its `inputs` argument to the
    processor output, so from the second batch onwards it sliced the processed
    tensors instead of the original images.
    """
    vec = StubImageVectorizer(model_name="stub", device="cpu")

    colors = [0, 50, 100, 150, 200]
    images = [Image.new("L", (28, 28), color=c) for c in colors]
    out = vec(images, batchsize=2)

    assert len(out["embed"]) == len(images)
    # 5 images at batchsize 2 -> the loop must actually run three times.
    assert vec.get_processor.batch_sizes == [2, 2, 1]
    for color, tensor in zip(colors, out["embed"]):
        assert tensor.shape == (STUB_DIM,)
        # Each vector carries its own image's pixel value: order is preserved and
        # no image was dropped or embedded twice.
        assert torch.allclose(tensor, torch.full((STUB_DIM,), float(color)))


@pytest.mark.integration
def test_image_vectorizer_resnet18():
    """ResNet-18 should embed grayscale images into 512-dim vectors.

    Marked integration because it downloads `microsoft/resnet-18`.
    """
    vec = ImageVectorizer(model_name="microsoft/resnet-18", device="cpu")

    assert vec.input_col == "image"
    assert vec.col_name == "embed"
    assert vec.get_hidden_size == 512

    # Two synthetic grayscale (mode "L") images, like Fashion-MNIST.
    images = [Image.new("L", (28, 28), color=c) for c in (0, 255)]
    out = vec(images, batchsize=2)

    assert set(out.keys()) == {"embed"}
    assert len(out["embed"]) == 2
    for tensor in out["embed"]:
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (512,)
        assert tensor.dtype == torch.float32
