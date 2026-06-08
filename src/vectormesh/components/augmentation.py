"""Feature-space augmentation components.

These perturb *embeddings* rather than raw inputs. In the embed-and-cache workflow
the expensive encoder is run once and frozen, so classic pixel-space augmentation
(random crops/flips) would mean re-running the encoder every epoch. Instead we
augment the cached vectors directly: cheap, fresh on every step, and a good
regulariser for the small head -- especially when training on few examples.
"""

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from vectormesh.types import BaseComponent


class GaussianNoise(BaseComponent):
    """Add Gaussian noise to embeddings during training (a no-op at eval).

    Acts on the last dimension, so it accepts any leading shape -- ``(batch, dim)``
    or ``(batch, seq, dim)`` -- and leaves the shape unchanged, which means it drops
    straight into a ``Serial`` pipeline in front of the classifier head.

    Args:
        sigma: noise standard deviation. With ``relative=True`` it is a *fraction*
            of each sample's feature spread; with ``relative=False`` it is the
            absolute std of the added noise.
        relative: if True (default), the noise is scaled by each sample's standard
            deviation over the feature dimension. This makes a single ``sigma``
            behave consistently across encoders whose embeddings live on different
            scales (e.g. a ResNet pooler vs. a DINOv2 CLS vector).
    """

    def __init__(self, sigma: float = 0.1, relative: bool = True):
        super().__init__()
        self.sigma = sigma
        self.relative = relative

    @jaxtyped(typechecker=beartype)
    def forward(self, tensors: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        if not self.training or self.sigma == 0:
            return tensors
        noise = torch.randn_like(tensors)
        if self.relative:
            scale = tensors.detach().std(dim=-1, keepdim=True)
            noise = noise * scale
        return tensors + noise * self.sigma
