"""A design as typed stages: the options it names, the subgraphs they build, and the shape
after every stage. The design under test is a fusion of two streams whose branches may be
projected before the join."""

import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from vectormesh.components import Concatenate2D, NeuralNet, Parallel, Projection
from vectormesh.design import Design, axis_of, option

A_DIM, B_DIM, N_CLASSES = 12, 20, 5


@dataclass(frozen=True)
class Config:
    shared_dim: int = 8
    project: bool = False


class Fusion(nn.Module):
    projections: Parallel

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.shared_dim = config.shared_dim
        self.n_classes = N_CLASSES
        self.joined_dim = 2 * config.shared_dim if config.project else A_DIM + B_DIM
        if config.project:
            self.projections = Parallel(
                [
                    Projection(A_DIM, config.shared_dim),
                    Projection(B_DIM, config.shared_dim),
                ]
            )
        self.concatenate = Concatenate2D()
        self.head = NeuralNet(hidden_size=self.joined_dim, out_size=N_CLASSES)

    @option("project")
    @jaxtyped(typechecker=beartype)
    def project(
        self, streams: tuple[Float[Tensor, "batch _"], Float[Tensor, "batch _"]]
    ) -> tuple[
        Float[Tensor, "batch {self.shared_dim}"],
        Float[Tensor, "batch {self.shared_dim}"],
    ]:
        return self.projections(streams)

    @jaxtyped(typechecker=beartype)
    def join(
        self, streams: tuple[Float[Tensor, "batch _"], Float[Tensor, "batch _"]]
    ) -> Float[Tensor, "batch {self.joined_dim}"]:
        return self.concatenate(streams)

    @jaxtyped(typechecker=beartype)
    def classify(
        self, joined: Float[Tensor, "batch {self.joined_dim}"]
    ) -> Float[Tensor, "batch {self.n_classes}"]:
        return self.head(joined)

    def forward(
        self, streams: tuple[Float[Tensor, "batch _"], Float[Tensor, "batch _"]]
    ) -> Float[Tensor, "batch _"]:
        if self.config.project:
            streams = self.project(streams)
        return self.classify(self.join(streams))


DESIGN = Design(Fusion, Config)


def pair() -> tuple[Tensor, Tensor]:
    return torch.randn(4, A_DIM), torch.randn(4, B_DIM)


def test_option_leaves_the_stage_as_it_is():
    def stage(x):
        return x + 1

    assert option("width")(stage) is stage
    assert axis_of(stage) == "width"
    assert axis_of(len) is None


def test_design_refuses_a_config_that_is_no_dataclass():
    with pytest.raises(TypeError, match="not a dataclass"):
        Design(Fusion, dict)


def test_options_stages_and_choices_are_read_off_the_classes():
    assert DESIGN.options() == ["project"]
    assert DESIGN.stages() == ["project", "join", "classify"]
    assert DESIGN.choices() == ["shared_dim", "project"]
    assert list(DESIGN.combinations()) == [{"project": False}, {"project": True}]


def test_drawn_keeps_the_choices_of_the_design():
    config = DESIGN.drawn({"project": True, "shared_dim": 4, "lr": 1e-3})
    assert config == Config(shared_dim=4, project=True)


@pytest.mark.parametrize("chosen", list(DESIGN.combinations()), ids=str)
def test_trace_gives_the_stages_a_subgraph_runs(chosen):
    model = Fusion(DESIGN.drawn(chosen))
    steps = DESIGN.trace(model, pair())
    ran = [step.stage for step in steps]
    assert ran[0] == "input" and ran[-2:] == ["join", "classify"]
    assert ("project" in ran) == chosen["project"]
    assert steps[-1].shape == (4, N_CLASSES)
    joined = next(step.shape for step in steps if step.stage == "join")
    assert joined == (4, 16 if chosen["project"] else A_DIM + B_DIM)


def test_trace_leaves_the_model_as_it_found_it():
    model = Fusion(Config(project=True)).train()
    DESIGN.trace(model, pair())
    assert model.training and "project" not in vars(model)


def test_show_prints_every_combination(capsys):
    DESIGN.show(lambda chosen: Fusion(DESIGN.drawn(chosen)), pair())
    out = capsys.readouterr().out
    assert "project=False" in out and "project=True" in out
    assert out.count("classify") == 2


def test_the_design_example_in_the_docs_prints_what_the_docs_show(capsys):
    doc = (Path(__file__).parent.parent / "docs" / "05-architectures.md").read_text()
    section = doc[doc.index("## 5.8") :]
    code = re.search(r"```python\n(.*?)```", section, re.DOTALL)
    shown = re.search(r"```\n(project=False.*?)```", section, re.DOTALL)
    assert code and shown
    exec(code.group(1), {"__name__": "docs"})  # noqa: S102
    assert capsys.readouterr().out.strip() == shown.group(1).strip()
