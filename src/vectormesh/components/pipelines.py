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


def _ema_update(target: nn.Module, online: nn.Module, momentum: float) -> None:
    """One EMA step, shared by every self-updating target-encoder component:
    ``target <- momentum * target + (1 - momentum) * online``.

    Buffers (BatchNorm running statistics) are copied rather than averaged --
    they are estimates of the data, not learned parameters, so blending them
    would just lag the same estimate.
    """
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.mul_(momentum).add_(op.detach(), alpha=1 - momentum)
    for tb, ob in zip(target.buffers(), online.buffers()):
        tb.copy_(ob)


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
        """One EMA step -- see ``_ema_update``. A no-op when ``momentum == 0.0``,
        since there is no separate target module to update."""
        if self.target is None:
            return
        _ema_update(self.target, self.online, self.momentum)

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


class JEPA(nn.Module):
    """Predict a masked-out representation from the visible ones -- I-JEPA's shape.

    ``Siamese`` covers the augmentation family: two independently-made views of the
    same architecture, agreement is the whole task. JEPA is a different shape --
    ONE input, masked two ways. The online encoder sees only the context positions;
    a slowly-trailing EMA copy of it sees the whole input and supplies the
    prediction target at the positions the online side never saw. A predictor turns
    "the visible tokens, plus which positions to guess" into predictions there.

    Same EMA-target-in-``forward`` trick as ``Siamese``, so this also drops into
    any trainer without the trainer knowing self-supervision exists -- but the
    predictor's signature is different (it needs *where* to predict, not just
    *what*), which is why this is a sibling component rather than a ``Siamese``
    configuration.

    Args:
        online: encoder called as ``online(x, keep=ctx_idx)``, returning
            ``(batch, len(ctx_idx), dim)`` -- must accept a ``keep`` kwarg
            selecting which positions to encode.
        predictor: called as ``predictor(context, ctx_idx, tgt_idx)``, returning
            ``(batch, len(tgt_idx), dim)``.
        momentum: EMA coefficient for the target branch. ``0.0`` means the target
            *is* the online encoder (no trailing copy) -- the ablation notebook
            exercises reach for. Ramp it yourself between calls
            (``jepa.momentum = ...``) if your schedule needs that; this component
            holds today's value, not a schedule.
        stopgrad: whether the target branch is detached. Keep ``True`` unless you
            are demonstrating collapse (see ``Siamese``).

    Called as ``jepa((x, ctx_idx, tgt_idx))`` -- a 3-tuple, so a plain
    ``vectormesh.training.Trainer`` (whose ``_to_device`` already handles tuple
    inputs) can train this with no custom ``step``. Returns ``(prediction,
    target)``, a flat pair -- not ``Siamese``'s nested one -- so
    ``RepresentationStd(index=(0,))`` reaches the prediction.
    """

    def __init__(
        self,
        online: nn.Module,
        predictor: nn.Module,
        momentum: float = 0.996,
        stopgrad: bool = True,
    ):
        super().__init__()
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        self.online = online
        self.predictor = predictor
        self.momentum = momentum
        self.stopgrad = stopgrad
        if momentum > 0.0:
            self.target: nn.Module | None = copy.deepcopy(online)
            for p in self.target.parameters():
                p.requires_grad = False
        else:
            self.target = None

    @torch.no_grad()
    def update_target(self) -> None:
        """One EMA step -- see ``_ema_update``. A no-op when ``momentum == 0.0``."""
        if self.target is None:
            return
        _ema_update(self.target, self.online, self.momentum)

    def forward(self, inputs: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        x, ctx_idx, tgt_idx = inputs
        if self.training:
            self.update_target()

        z_ctx = self.online(x, keep=ctx_idx)

        target_enc = self.target if self.target is not None else self.online
        if self.stopgrad:
            with torch.no_grad():
                z_all = target_enc(x)
        else:
            z_all = target_enc(x)
        target = z_all[:, tgt_idx, :]

        prediction = self.predictor(z_ctx, ctx_idx, tgt_idx)
        return prediction, target
