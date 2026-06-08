"""Tests for the image embedding path (ImageVectorizer)."""

import pytest
import torch
from PIL import Image

from vectormesh import ImageVectorizer


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
