import torch.nn as nn
from beartype import beartype
from jaxtyping import jaxtyped

from vectormesh.types import TensorInput


class Serial(nn.Module):
    """Sequential composition - just runs components in order.
    Tensor checking happens via jaxtyping decorators on each component.
    """

    components: nn.ModuleList

    def __init__(self, components: list[nn.Module]):
        super().__init__()
        self.components = nn.ModuleList(components)

    @jaxtyped(typechecker=beartype)
    def forward(self, tensors: TensorInput) -> TensorInput:
        """Execute pipeline. Type checking via component decorators."""
        result = tensors
        for component in self.components:
            result = component(result)
        return result


class Parallel(nn.Module):
    """Parallel composition - runs branches independently and returns tuple.
    All branches receive the same input.
    """

    branches: nn.ModuleList

    def __init__(self, branches: list[nn.Module]):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    @jaxtyped(typechecker=beartype)
    def forward(self, tensors: TensorInput) -> TensorInput:
        return tuple(branch(t) for branch, t in zip(self.branches, tensors))
