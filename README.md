# VectorMesh

A PyTorch framework for **embed once, reuse many times**.

A large pretrained encoder — a BERT-family text model, a vision CNN/ViT, or a hand-written regex
feature extractor — is run over a dataset exactly once, by whoever has the hardware and the
judgement to pick it. The resulting vectors go to disk as a `VectorCache`: a versioned,
documented artefact. Everything after that trains a *small* head on those frozen vectors, cheaply
enough to run on a laptop CPU.

That split is the point. One embedding job, many models trained against its output — and the
interesting design space (fusing representations, gating, mixtures of experts) lives entirely on
the cheap side of it.

## 📚 Documentation

Full documentation is in **[`docs/`](docs/README.md)**.

| | |
|---|---|
| [Core concepts](docs/01-core-concepts.md) | The embed-once economics, thinking at the vector level, the 1D/2D/3D tensor-flow ladder |
| [Tensor contracts](docs/02-tensor-contracts.md) | Shapes as a type system: jaxtyping + beartype, and how to read a shape error |
| [The data layer](docs/03-data-layer.md) | Vectorizers, `VectorCache`, metadata, `DatasetSchema`, collation |
| [Components](docs/04-components.md) | Every building block, with shapes and when to reach for it |
| [Architectures](docs/05-architectures.md) | Composition patterns from a two-line baseline to chunk-level MoE fusion |
| [Training](docs/06-training.md) | Metrics, loss choice, `mltrainer` wiring, the batch scripts |
| [API reference](docs/07-api-reference.md) | Signature tables for everything exported |
| [Teaching path](docs/08-teaching-path.md) | Notebook-by-notebook map and the questions each one raises |

## Installation

```bash
uv sync
```

Requires Python ≥ 3.12. See `pyproject.toml` for the full dependency list.

## Quick start

```python
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vectormesh import VectorCache
from vectormesh.components import FixedPadding, MaskedMeanAggregator, NeuralNet, Serial
from vectormesh.data import Collate, OneHot

cache = VectorCache.load(path=Path("artefacts/my_dataset_train"))
# ...or a cache someone else already paid to compute:
# cache = VectorCache.from_hub("pttrn-io/eurosat-dinov2-small", split="train")
hidden_size = cache.metadata["legal_dutch"]["hidden_size"]

data = cache.dataset.map(OneHot(num_classes=32, label_col="labels", target_col="onehot"))
loader = DataLoader(
    data,
    batch_size=32,
    shuffle=True,
    collate_fn=Collate(
        embedding_col="legal_dutch",
        target_col="onehot",
        padder=FixedPadding(max_chunks=30),
    ),
)

pipeline = Serial([
    MaskedMeanAggregator(),                  # (batch, chunks, dim) -> (batch, dim)
    NeuralNet(hidden_size, out_size=32),     # (batch, dim)         -> (batch, 32)
])
```

Then hand `pipeline` and `loader` to `mltrainer.Trainer` — see
[Training](docs/06-training.md) for the full wiring.

Building a cache instead of loading one, extending a cache with extra feature columns, and the
image path are all covered in [The data layer](docs/03-data-layer.md).

## What's in the box

**Data** — `VectorCache`, `Vectorizer` (chunked text), `ImageVectorizer`, `RegexVectorizer`,
`ChunkedRegexVectorizer`, `DatasetSchema`, `OneHot`, `Collate`, `CollateParallel`, `LabelEncoder`.

**Components** — pipelines (`Serial`, `Parallel`), padding (`FixedPadding`, `DynamicPadding`),
aggregation (`Mean`, `MaskedMean`, `Attention`, `RNN`), neural blocks (`NeuralNet`, `Projection`,
`Attention`, `TransformerBlock`), connectors (`Concatenate2D`, `Concatenate3D`, `Stack2D`), gating
(`Skip`, `Gate`, `Highway`, `MoE`), augmentation (`GaussianNoise`), metrics (`Accuracy`, `F1Score`,
`MAE`, `MASE`).

Details and signatures: [Components](docs/04-components.md),
[API reference](docs/07-api-reference.md).

## Runtime type checking

VectorMesh annotates every `forward` with jaxtyping shapes, checked at runtime by beartype:

```python
@jaxtyped(typechecker=beartype)
def forward(self, tensors: Float[Tensor, "batch chunks dim"]) -> Float[Tensor, "batch dim"]:
    return tensors.mean(dim=1)
```

This catches the failure mode that matters most here: `nn.Linear` accepts any leading shape, so
feeding it `(batch, chunks, dim)` when you meant `(batch, dim)` raises no error — it just trains
a model that quietly answers the wrong question. A `BeartypeCallHintParamViolation` naming a 3D
tensor where a 2D one was expected almost always means *a missing aggregator*.

How to read those errors: [Tensor contracts](docs/02-tensor-contracts.md).

## Notebooks

| Notebook | Topic |
|---|---|
| `0_vectorizer.ipynb` | Creating vector caches; extending one with regex features |
| `1_training.ipynb` | Cache → padding → aggregation → pipeline → trained model |
| `2_design.ipynb` | Parallel branches, fusing two representations, skip connections |
| `3_moe.ipynb` | Mixture of experts |
| `4_image_vectorizer.ipynb` | The same pipeline on images; feature-space augmentation |

Walkthrough and exercises: [Teaching path](docs/08-teaching-path.md).

## Scripts

Batch counterparts to the notebooks, for real dataset sizes:

```bash
uv run python scripts/create_cache_aktes.py    # embeddings
uv run python scripts/add_chunked_regex.py     # + chunk-aligned regex column
uv run python scripts/train_moe_parallel.py    # train
```

Full list: [Training §6.5](docs/06-training.md#65-the-scripts).

## Development

```bash
uv run pytest -m "not integration"   # unit tests, no model downloads
uv run pytest                        # everything, downloads HF models
uv run ruff check src tests
uv run ty check
```

## Project structure

```
src/vectormesh/
├── types.py               # VectorMeshError, Cachable, BaseComponent, TensorInput
├── data/                  # vectorizers, cache, schema, collation
└── components/            # pipelines, padding, aggregation, neural,
                           # connectors, gating, augmentation, metrics
docs/                      # documentation (start at docs/README.md)
notebooks/                 # tutorials, in order
scripts/                   # batch pipelines
references/                # source papers
tests/
```
