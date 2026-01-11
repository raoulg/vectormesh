import torch.nn as nn
from beartype import beartype
from jaxtyping import jaxtyped


class Serial(nn.Module):
    """Sequential composition - just runs components in order.
    Tensor checking happens via jaxtyping decorators on each component.
    """

    def __init__(self, components: list):
        super().__init__()
        self.components = nn.ModuleList(
            [c for c in components if isinstance(c, nn.Module)]
        )
        self._all_components = components

    @jaxtyped(typechecker=beartype)
    def forward(self, tensors):
        """Execute pipeline. Type checking via component decorators."""
        result = tensors
        for component in self._all_components:
            result = component(result)
        return result


class Parallel(nn.Module):
    """Parallel composition - runs branches independently and returns tuple.
    All branches receive the same input.
    """

    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList([b for b in branches if isinstance(b, nn.Module)])
        self._all_branches = branches

    @jaxtyped(typechecker=beartype)
    def forward(self, tensors):
        return tuple(branch(tensors) for branch in self._all_branches)
