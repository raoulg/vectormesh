"""`Norm`: the four kinds normalise what they claim to, at rank 2 and rank 3."""

import pytest
import torch

from vectormesh.components import Norm
from vectormesh.types import VectorMeshError


@pytest.mark.parametrize("kind", ["layer", "batch", "rms", "none"])
@pytest.mark.parametrize("shape", [(8, 16), (4, 5, 16)])
def test_keeps_the_shape(kind, shape):
    assert Norm(16, kind)(torch.randn(*shape)).shape == shape


def test_layer_centres_and_scales_each_vector():
    out = Norm(16, "layer")(torch.randn(8, 16) * 5 + 3)
    assert torch.allclose(out.mean(-1), torch.zeros(8), atol=1e-5)
    assert torch.allclose(out.std(-1, unbiased=False), torch.ones(8), atol=1e-3)


def test_batch_centres_each_feature_over_every_position():
    x = torch.randn(4, 5, 16) * 5 + 3
    out = Norm(16, "batch").train()(x)
    flat = out.reshape(-1, 16)
    assert torch.allclose(flat.mean(0), torch.zeros(16), atol=1e-5)
    # a per-vector mean is not zero: it is not a layer norm
    assert not torch.allclose(out.mean(-1), torch.zeros(4, 5), atol=1e-2)


def test_batch_uses_running_statistics_at_eval():
    norm = Norm(16, "batch")
    norm.train()
    for _ in range(100):  # momentum 0.1: the running mean needs steps to reach the data
        norm(torch.randn(32, 16) + 10)
    norm.eval()
    one = torch.full((1, 16), 10.0)
    assert norm(one).abs().max() < 1.0  # a single row is normalised by the running mean


def test_rms_scales_without_centring():
    x = torch.randn(8, 16) + 3
    out = Norm(16, "rms")(x)
    assert torch.allclose(out.pow(2).mean(-1).sqrt(), torch.ones(8), atol=1e-3)
    assert (out.mean(-1) > 0.5).all()


def test_none_is_the_identity():
    x = torch.randn(8, 16)
    assert torch.equal(Norm(16, "none")(x), x)
    assert sum(p.numel() for p in Norm(16, "none").parameters()) == 0


def test_unknown_kind_is_refused():
    with pytest.raises(VectorMeshError, match="group"):
        Norm(16, "group")  # ty: ignore[invalid-argument-type]


def test_wrong_width_is_refused():
    with pytest.raises(TypeError):
        Norm(16, "layer")(torch.randn(8, 12))
