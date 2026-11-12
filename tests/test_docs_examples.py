"""The §9.5 trainable template in docs/09-architecture-search.md runs against the
current API.

The template is extracted from the markdown and executed on a tiny synthetic cache
pair. Ray is not a vectormesh dependency, so ``ray`` is replaced by a stub whose
``tune.report`` records the reported dict: the trainable is called directly with a
config, which is what a ray worker does inside its own process. Everything that
touches vectormesh (cache loading, collation, components, ``Trainer``) is real.
"""

import json
import re
import sys
import types
from pathlib import Path

import pytest
from datasets import ClassLabel, Dataset, Features, Sequence, Value

DOC = Path(__file__).parent.parent / "docs" / "09-architecture-search.md"
DIM = 8
N_CLASSES = 3


def _section_code(heading: str) -> str:
    """The first python block under the heading that starts with ``heading``."""
    text = DOC.read_text()
    start = text.index(f"\n## {heading}")
    match = re.search(r"```python\n(.*?)```", text[start:], re.DOTALL)
    assert match is not None, f"no python block under {heading}"
    return match.group(1)


def _write_cache(path: Path, n: int, seed: int) -> None:
    """A chunked cache on disk: each item is a ``(chunks, DIM)`` block of vectors."""
    import random

    rng = random.Random(seed)
    labels = [i % N_CLASSES for i in range(n)]
    embeds = [
        [[rng.random() + label for _ in range(DIM)] for _ in range(rng.randint(1, 4))]
        for label in labels
    ]
    features = Features(
        {
            "label": ClassLabel(num_classes=N_CLASSES),
            "embed": Sequence(Sequence(Value("float32"))),
        }
    )
    Dataset.from_dict(
        {"label": labels, "embed": embeds}, features=features
    ).save_to_disk(path)
    metadata = {
        "embed": {
            "vectormesh_version": "test",
            "model_tag": "org/stub",
            "vectorizer_type": "StubVectorizer",
            "tensordtype": 2,
            "hidden_size": DIM,
        },
        "features": ["label", "embed"],
        "created_at": "2026-01-01T00:00:00",
        "num_observations": n,
    }
    (path / "metadata.json").write_text(json.dumps(metadata))


@pytest.fixture
def reported(monkeypatch) -> list[dict]:
    """Stand in for ray, returning the list that ``tune.report`` appends to."""
    calls: list[dict] = []
    tune = types.SimpleNamespace(report=calls.append)
    modules = {
        "ray": types.SimpleNamespace(tune=tune),
        "ray.tune": tune,
        "ray.tune.search": types.SimpleNamespace(),
        "ray.tune.search.hyperopt": types.SimpleNamespace(HyperOptSearch=object),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return calls


@pytest.fixture
def template(tmp_path, reported, monkeypatch) -> dict:
    """The §9.5 block executed up to (not including) the ``Tuner`` call, with the cache
    paths pointed at synthetic caches."""
    code = _section_code("9.5")
    head, sep, _ = code.partition("\ntuner = tune.Tuner(")
    assert sep, "the §9.5 block no longer ends in a Tuner call"
    monkeypatch.chdir(tmp_path)  # the trainable's logdir is relative to the cwd
    namespace: dict = {"__name__": "docs_example"}
    exec(compile(head, str(DOC), "exec"), namespace)  # noqa: S102
    _write_cache(tmp_path / "train", n=24, seed=0)
    _write_cache(tmp_path / "test", n=12, seed=1)
    namespace["TRAIN_CACHE"] = tmp_path / "train"
    namespace["TEST_CACHE"] = tmp_path / "test"
    return namespace


@pytest.mark.parametrize("mechanism", ["plain", "gate", "highway", "moe"])
def test_trainable_runs_one_trial(template, reported, mechanism):
    config = {
        "mechanism": mechanism,
        "hidden": 16,
        "lr": 1e-3,
        "n_experts": 2,
        "expert_hidden": 16,
        "epochs": 2,
    }
    template["trainable"](config)

    assert len(reported) == 1
    assert 0.0 <= reported[0]["accuracy"] <= 1.0


def test_every_searched_mechanism_has_a_builder(template):
    doc_choices = re.search(r'"mechanism": tune.choice\((\[.*?\])\)', DOC.read_text())
    assert doc_choices is not None
    assert set(json.loads(doc_choices.group(1))) == set(template["MECHANISMS"])
