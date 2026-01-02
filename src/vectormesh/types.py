from torch import Tensor
from jaxtyping import Float

# Define strict tensor types for shape validation
# Usage: def forward(x: TwoDTensor) -> OneDTensor: ...

OneDTensor = Float[Tensor, "dim"]
OneDTensor.__doc__ = "1D Tensor representing a single vector. Shape: (dim,)"

TwoDTensor = Float[Tensor, "batch dim"]
TwoDTensor.__doc__ = "2D Tensor representing a batch of vectors. Shape: (batch, dim)"

ThreeDTensor = Float[Tensor, "batch seq dim"]
ThreeDTensor.__doc__ = "3D Tensor representing a batch of sequences of vectors. Shape: (batch, seq, dim)"
