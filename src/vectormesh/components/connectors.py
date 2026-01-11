import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor


class Concatenate2D:
    """Concatenate tuples from parallel branches at last dimension.

    input: ((batch dim), (batch dim), ...)
    output: (batch nstack*dim)

    where nstack is the number of tensors in the tuple

    """

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, tensors: tuple[Float[Tensor, "batch dim"], ...]
    ) -> Float[Tensor, "batch comb_dim"]:
        return torch.cat(tensors, dim=-1)


class Stack2D:
    """Stack tuples from parallel branches, default 1st dimension.

    input : ((batch dim), (batch dim), ...)
    output: (batch nstack dim)

    where nstack is the number of tensors in the tuple
    """

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, tensors: tuple[Float[Tensor, "batch dim"], ...]
    ) -> Float[Tensor, "batch stack dim"]:
        return torch.stack(tensors, dim=1)
