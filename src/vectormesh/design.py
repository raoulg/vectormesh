"""A design: an architecture written as typed stages, with the options a search may switch.

A design is a supergraph: every block a search may build, written once as a method of an
`nn.Module` (a stage). A stage's signature states the shape it takes and the shape it
gives, and inside a stage the components compose as everywhere else (`Serial`,
`Parallel`, a connector). A block the search may leave out is a choice in `forward`
(`if self.config.skip:`), and `@option("skip")` on the stage names the axis of the search
space that decides it. The config a trial draws picks the subgraph that is built.

    from vectormesh.design import Design, option

    class Head(nn.Module):
        @option("skip")
        @jaxtyped(typechecker=beartype)
        def skip(self, x: Float[Tensor, "batch {self.dim}"]) -> ...: ...

    DESIGN = Design(Head, Config)

`Design.build(config, **widths)` makes the module a config chooses; `Design.show(example,
**widths)` builds every combination of the options and prints the shape after every
stage.

    DESIGN.show(torch.randn(2, 384), dim=384, n_classes=47)
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Mapping
from dataclasses import fields, is_dataclass

import torch
from pydantic import BaseModel
from torch import Tensor, nn

from vectormesh.types import TensorInput

# a value a config holds for one choice of a design
Choice = str | int | float | bool


def option[F: Callable[..., object]](axis: str) -> Callable[[F], F]:
    """Mark a stage as one the search may leave out, decided by the axis `axis` of the
    search space. The stage stays as it is; the axis name is kept on it as `__option__`."""

    def mark(stage: F) -> F:
        vars(stage)["__option__"] = axis
        return stage

    return mark


def axis_of(stage: object) -> str | None:
    """The axis `option` named on `stage`, or None for a stage that is always built."""
    try:
        return vars(stage).get("__option__")
    except TypeError:  # an object without attributes of its own
        return None


class StageShape(BaseModel):
    """One stage a forward pass ran, and the shape it gave."""

    stage: str
    shape: tuple[int, ...]


class Design[M: nn.Module, C]:
    """The design class `module` and the config dataclass `config` that holds its
    choices."""

    def __init__(self, module: type[M], config: type[C]) -> None:
        if not is_dataclass(config):
            raise TypeError(f"Design: {config.__name__} is not a dataclass")
        self.module = module
        self.config = config

    def options(self) -> list[str]:
        """The axis names of the options, in the order the stages are written."""
        return [
            axis for stage in vars(self.module).values() if (axis := axis_of(stage))
        ]

    def stages(self) -> list[str]:
        """The names of the stages: the module's methods other than `forward`."""
        return [
            name
            for name, stage in vars(self.module).items()
            if callable(stage) and not name.startswith("_") and name != "forward"
        ]

    def choices(self) -> list[str]:
        """The field names of the config: the axes the design reads."""
        return [f.name for f in fields(self.config)]

    def drawn(self, config: Mapping[str, Choice]) -> C:
        """The choices of the design in a config a trial drew; the rest of the config
        (the dataset, the learning rate, ...) is the trainer's."""
        names = set(self.choices())
        return self.config(**{k: v for k, v in config.items() if k in names})

    def combinations(self) -> Iterator[dict[str, bool]]:
        """Every combination of the options, each option on or off."""
        axes = self.options()
        for values in itertools.product([False, True], repeat=len(axes)):
            yield dict(zip(axes, values, strict=True))

    def trace(self, model: M, example: TensorInput) -> list[StageShape]:
        """The stages `model(example)` runs, in order, each with the shape it gives.
        `example` is one batch input: a tensor, or a tuple of tensors for a model with
        more than one stream. For a tuple, the shape is that of its first tensor. The
        model is left as it was found."""
        first = example[0] if isinstance(example, tuple | list) else example
        seen = [StageShape(stage="input", shape=tuple(first.shape))]

        def watch(name: str, stage: Callable[..., Tensor]) -> Callable[..., Tensor]:
            def watched(*args: object, **kwargs: object) -> Tensor:
                out = stage(*args, **kwargs)
                shape = out[0].shape if isinstance(out, tuple | list) else out.shape
                seen.append(StageShape(stage=name, shape=tuple(shape)))
                return out

            return watched

        names = self.stages()
        for name in names:
            # an attribute on the instance comes before the method of the class
            object.__setattr__(model, name, watch(name, getattr(model, name)))
        training = model.training
        try:
            model.eval()
            with torch.no_grad():
                model(example)
        finally:
            model.train(training)
            for name in names:
                object.__delattr__(model, name)
        return seen

    def build(self, config: C | Mapping[str, Choice], **widths: int) -> M:
        """The module a config chooses. `config` is an instance of the config dataclass
        or a config a trial drew; `widths` are the sizes the data fixes (`dim`,
        `n_classes`), passed to the module as keywords after its config."""
        if isinstance(config, Mapping):
            config = self.drawn(config)
        return self.module(config, **widths)

    def show(self, example: TensorInput, **widths: int) -> None:
        """Print, for every combination of the options, its number of params and the
        shape after every stage. Each combination is built by `build(combination,
        **widths)`; the choices that are no options keep their defaults."""
        for chosen in self.combinations():
            model = self.build(chosen, **widths)
            n_params = sum(p.numel() for p in model.parameters())
            label = (
                "  ".join(f"{axis}={on}" for axis, on in chosen.items())
                or "(no options)"
            )
            print(f"{label}   ({n_params:,} params)")
            for step in self.trace(model, example):
                print(f"  {step.stage:<12} {step.shape}")
            print()
