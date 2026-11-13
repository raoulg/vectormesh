"""An image classifier: kernels, a residual block, and a gate that weighs whole channels.

Each stage is a method whose signature states the shape it takes and the shape it gives.
Three steps are plain PyTorch, because the library has no component for them: the
activation, the residual sum on an image, and the multiply that puts one gate value on
a whole channel. Run from the repository root:

    uv run python scripts/design_image.py
"""

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from vectormesh.components import Conv2D, GlobalPool2D, NeuralNet, Projection


class ImageClassifier(nn.Module):
    """``(batch, colours, height, width) -> (batch, classes)``."""

    def __init__(
        self, n_colours: int, channels: int, reduced: int, n_classes: int
    ) -> None:
        super().__init__()
        self.n_colours = n_colours
        self.channels = channels
        self.n_classes = n_classes
        # padding 1 with kernel 3 keeps height and width
        self.stem = Conv2D(n_colours, channels, kernel_size=3, padding=1)
        self.block_in = Conv2D(channels, channels, kernel_size=3, padding=1)
        self.block_out = Conv2D(channels, channels, kernel_size=3, padding=1)
        self.channel_means = GlobalPool2D("mean")
        self.excite = NeuralNet([channels, reduced], channels)
        self.to_vector = GlobalPool2D("mean")
        self.head = Projection(channels, n_classes)

    @jaxtyped(typechecker=beartype)
    def local_patterns(
        self, image: Float[Tensor, "batch {self.n_colours} height width"]
    ) -> Float[Tensor, "batch {self.channels} height width"]:
        """A kernel slides over height and width: one feature map per channel."""
        return torch.relu(self.stem(image))

    @jaxtyped(typechecker=beartype)
    def transform(
        self, maps: Float[Tensor, "batch {self.channels} height width"]
    ) -> Float[Tensor, "batch {self.channels} height width"]:
        """Two more kernels that keep the shape."""
        return self.block_out(torch.relu(self.block_in(maps)))

    @jaxtyped(typechecker=beartype)
    def residual(
        self, maps: Float[Tensor, "batch {self.channels} height width"]
    ) -> Float[Tensor, "batch {self.channels} height width"]:
        """The block adds to its input: the input passes unchanged beside it."""
        return maps + self.transform(maps)

    @jaxtyped(typechecker=beartype)
    def channel_gate(
        self, maps: Float[Tensor, "batch {self.channels} height width"]
    ) -> Float[Tensor, "batch {self.channels}"]:
        """One value between 0 and 1 per channel, computed from the channel means."""
        return torch.sigmoid(self.excite(self.channel_means(maps)))

    @jaxtyped(typechecker=beartype)
    def weigh_channels(
        self,
        maps: Float[Tensor, "batch {self.channels} height width"],
        gate: Float[Tensor, "batch {self.channels}"],
    ) -> Float[Tensor, "batch {self.channels} height width"]:
        """The gate of a channel multiplies every pixel of that channel."""
        return maps * gate[:, :, None, None]

    @jaxtyped(typechecker=beartype)
    def summarise(
        self, maps: Float[Tensor, "batch {self.channels} height width"]
    ) -> Float[Tensor, "batch {self.channels}"]:
        """From an image to a vector: the mean of every channel."""
        return self.to_vector(maps)

    @jaxtyped(typechecker=beartype)
    def classify(
        self, vector: Float[Tensor, "batch {self.channels}"]
    ) -> Float[Tensor, "batch {self.n_classes}"]:
        return self.head(vector)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, image: Float[Tensor, "batch {self.n_colours} height width"]
    ) -> Float[Tensor, "batch {self.n_classes}"]:
        maps = self.local_patterns(image)
        maps = self.residual(maps)
        gate = self.channel_gate(maps)
        weighed = self.weigh_channels(maps, gate)
        vector = self.summarise(weighed)
        return self.classify(vector)


@jaxtyped(typechecker=beartype)
def random_images(
    batch_size: int, n_colours: int, size: int
) -> Float[Tensor, "{batch_size} {n_colours} {size} {size}"]:
    return torch.randn(batch_size, n_colours, size, size)


def main() -> None:
    model = ImageClassifier(n_colours=3, channels=8, reduced=4, n_classes=3)
    images = random_images(batch_size=4, n_colours=model.n_colours, size=32)
    logits = model.forward(images)
    print(model)
    print(f"{tuple(images.shape)} -> {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
