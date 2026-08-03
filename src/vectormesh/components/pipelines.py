import copy

import torch
import torch.nn as nn
from torch import Tensor
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
    All branches receive the same input of Tuple[Tensor, ...] and return Tuple[Tensor, ...].

    The number of branches must match the number of tensors in the input tuple.
    """

    branches: nn.ModuleList

    def __init__(self, branches: list[nn.Module]):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    @jaxtyped(typechecker=beartype)
    def forward(self, tensors: TensorInput) -> TensorInput:
        return tuple(branch(t) for branch, t in zip(self.branches, tensors))


class Siamese(nn.Module):
    """`Parallel`, with the branches' weights coupled and a gradient gate on one side.

    This is the shape shared by every joint-embedding self-supervised method --
    SimCLR, MoCo, BYOL, SimSiam, DINO, I-JEPA. They differ only in how the two
    views are made, whether the target branch trails the online one, whether a
    predictor sits on the online side, and which loss compares the outputs. So
    rather than five implementations, this is one module with three switches:

    ==================  ===========================================  ==============
    ``momentum``        target branch                                 as in
    ==================  ===========================================  ==============
    ``0.0``             *is* the online encoder (shared weights)      SimSiam, SimCLR
    ``0 < m < 1``       an EMA copy trailing the online encoder       MoCo, BYOL, DINO
    ==================  ===========================================  ==============

    Why the gradient gate matters: making two views agree has a trivial solution
    -- output the same vector for every input. Both branches drifting toward each
    other is how a model finds it. ``stopgrad=True`` freezes the target side for
    the step, so only one branch can move, and the collapse is much harder to
    reach. Setting it to ``False`` is the ablation, and it collapses reliably.

    The EMA update deliberately lives in ``forward`` rather than in the training
    loop, so this drops into any trainer -- including ``mltrainer.Trainer``, which
    has no per-step hook -- without that trainer knowing self-supervision exists.

    Args:
        online: the encoder being trained.
        momentum: EMA coefficient for the target branch. ``0.0`` shares weights.
        predictor: optional head on the online branch only (BYOL, SimSiam, I-JEPA).
        stopgrad: whether the target branch is detached. Keep ``True`` unless you
            are demonstrating collapse.

    Returns a pair of ``(prediction, target)`` tuples -- symmetrised, so each view
    takes a turn as the target. Losses consume that pair; see the SSL losses.
    """

    def __init__(
        self,
        online: nn.Module,
        momentum: float = 0.0,
        predictor: nn.Module | None = None,
        stopgrad: bool = True,
    ):
        super().__init__()
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        self.online = online
        self.momentum = momentum
        self.stopgrad = stopgrad
        self.predictor = predictor
        if momentum > 0.0:
            self.target: nn.Module | None = copy.deepcopy(online)
            for p in self.target.parameters():
                p.requires_grad = False
        else:
            self.target = None

    @torch.no_grad()
    def update_target(self) -> None:
        """One EMA step: ``target <- m * target + (1 - m) * online``.

        Buffers (BatchNorm running statistics) are copied rather than averaged --
        they are estimates of the data, not learned parameters, so blending them
        would just lag the same estimate.
        """
        if self.target is None:
            return
        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.mul_(self.momentum).add_(op.detach(), alpha=1 - self.momentum)
        for tb, ob in zip(self.target.buffers(), self.online.buffers()):
            tb.copy_(ob)

    def _one_way(self, view_online: Tensor, view_target: Tensor):
        z_online = self.online(view_online)
        target = self.target if self.target is not None else self.online
        if self.stopgrad:
            with torch.no_grad():
                z_target = target(view_target)
        else:
            z_target = target(view_target)
        prediction = self.predictor(z_online) if self.predictor else z_online
        return prediction, z_target

    def forward(self, views: tuple[Tensor, Tensor]):
        view_a, view_b = views
        if self.training:
            self.update_target()
        return self._one_way(view_a, view_b), self._one_way(view_b, view_a)
