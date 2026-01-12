# VectorMesh

A PyTorch-based framework for efficient vector embedding management and multi-modal text classification. VectorMesh provides a flexible pipeline architecture for combining different types of text embeddings and building sophisticated neural architectures.

## Features

- **Efficient Vector Caching**: Pre-compute and store embeddings to avoid redundant processing
- **Multiple Vectorizers**: Support for Hugging Face models and regex-based feature extraction
- **Flexible Pipeline Architecture**: Compose models using Serial and Parallel pipelines
- **Chunked Document Processing**: Handle long documents with automatic chunking and padding
- **Advanced Components**: Aggregation, gating mechanisms, skip connections, and Mixture of Experts (MoE)
- **Easy Extension**: Add new vector types to existing caches without recomputing

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Using pip
pip install -e .
```

## Understanding Type Checking with Jaxtyping and Beartype

VectorMesh uses **jaxtyping** and **beartype** for runtime tensor shape validation. While this may produce new errors you haven't seen before, it's extremely helpful for two reasons:

### 1. Understanding Tensor Dimensionality

Type annotations make it explicit what tensor shapes each function expects and returns:

```python
@jaxtyped(typechecker=beartype)
def forward(
    self, embeddings: Float[Tensor, "batch chunks dim"]
) -> Float[Tensor, "batch dim"]:
    return embeddings.mean(dim=1)
```

This tells you immediately:
- **Input**: 3D tensor with shape (batch_size, num_chunks, embedding_dim)
- **Output**: 2D tensor with shape (batch_size, embedding_dim)

### 2. Catching Shape Mismatches Early

Without type checking, PyTorch often silently processes tensors with wrong shapes, leading to subtle bugs:

```python
# WITHOUT type checking - this runs but gives wrong results!
linear = nn.Linear(768, 32)
x = torch.randn(16, 30, 768)  # 3D tensor: (batch, chunks, dim)
output = linear(x)  # Returns (16, 30, 32) - probably not what you want!
print(output.shape)  # torch.Size([16, 30, 32])

# WITH type checking - this catches the error immediately!
class SafeProjection(nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "batch dim"]  # Expects 2D!
    ) -> Float[Tensor, "batch dim"]:
        return self.linear(x)

projection = SafeProjection(768, 32)
x = torch.randn(16, 30, 768)  # 3D tensor
output = projection(x)  # ❌ Raises TypeError immediately!
# beartype.roar.BeartypeCallHintParamViolation:
# Expected 2D tensor "batch dim", got 3D tensor with shape (16, 30, 768)
```

### Common Error Messages

When you see errors like:
```
beartype.roar.BeartypeCallHintParamViolation: Forward parameter 'embeddings'
violates type hint Float[Tensor, "batch dim"], as 3D tensor != 2D tensor
```

This means:
- You're passing the wrong tensor shape to a function
- Check the function signature to see what shape it expects
- You probably need to add an aggregator (e.g., `MeanAggregator`) or change your pipeline

**Pro tip**: Read the type hints in error messages carefully - they tell you exactly what went wrong!

## Quick Start

**Note**: You will receive pre-built datasets from your instructor. These datasets were created using the `build` function (see [Dataset Creation](#dataset-creation) section below), which splits raw data into train/test/validation sets and filters labels based on frequency thresholds.

### 1. Creating Vector Caches

```python
from pathlib import Path
from datasets import load_from_disk
from vectormesh import Vectorizer, VectorCache

# Load your dataset
dataset = load_from_disk("assets/dataset/train")

# Create a vectorizer with a Hugging Face model
vectorizer = Vectorizer(
    model_name="Gerwin/legal-bert-dutch-english",
    col_name="legal_dutch"
)

# Create and save vector cache
cache = VectorCache.create(
    cache_dir=Path("artefacts"),
    vectorizer=vectorizer,
    dataset=dataset,
    dataset_tag="my_dataset"
)
```

### 2. Extending Caches with Additional Features

```python
from vectormesh import RegexVectorizer, VectorCache
from vectormesh.data.vectorizers import (
    build_legal_reference_pattern,
    harmonize_legal_reference
)

# Create a regex-based vectorizer
regex_vectorizer = RegexVectorizer(
    pattern_builder=build_legal_reference_pattern,
    harmonizer=harmonize_legal_reference,
    min_doc_frequency=15,
    max_features=200,
    training_texts=dataset["text"]
)

# Extend existing cache with new features
extended_cache = VectorCache.create(
    cache_dir=Path("artefacts"),
    vectorizer=regex_vectorizer,
    dataset=cache.dataset,
    dataset_tag="my_dataset"
)
```

### 3. Training Models

```python
import torch
from torch.utils.data import DataLoader
from mltrainer import Trainer, TrainerSettings
from vectormesh.components import (
    Serial, MeanAggregator, NeuralNet, FixedPadding
)
from vectormesh.data import Collate, OneHot

# Load cache
cache = VectorCache.load(path=Path("artefacts/my_dataset"))

# Prepare data
onehot = OneHot(num_classes=32, label_col="labels", target_col="onehot")
train_data = cache.select(range(1000)).map(onehot)

# Create collate function with padding
collate_fn = Collate(
    embedding_col="legal_dutch",
    target_col="onehot",
    padder=FixedPadding(max_chunks=30)
)

# Create dataloader
trainloader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

# Build pipeline
pipeline = Serial([
    MeanAggregator(),  # (batch, chunks, dim) -> (batch, dim)
    NeuralNet(hidden_size=768, out_size=32)  # (batch, dim) -> (batch, 32)
])

# Train
trainer = Trainer(
    model=pipeline,
    settings=TrainerSettings(epochs=10, logdir=Path("logs")),
    loss_fn=torch.nn.BCEWithLogitsLoss(),
    optimizer=torch.optim.Adam,
    traindataloader=trainloader,
    validdataloader=validloader
)
trainer.loop()
```

## Advanced Usage

### Parallel Processing with Multiple Vector Types

Combine embeddings from different sources using parallel pipelines:

```python
from vectormesh.components import (
    Parallel, Serial, MeanAggregator, NeuralNet,
    Concatenate2D, FixedPadding
)
from vectormesh.data import CollateParallel

# Create parallel pipeline
parallel = Parallel([
    # Branch 1: Process 3D embeddings
    Serial([
        MeanAggregator(),
        NeuralNet(hidden_size=768, out_size=32)
    ]),
    # Branch 2: Process regex features
    Serial([
        NeuralNet(hidden_size=123, out_size=32)
    ])
])

# Combine outputs
pipeline = Serial([
    parallel,           # (X1, X2) -> (Y1, Y2)
    Concatenate2D(),    # (Y1, Y2) -> (batch, 64)
    NeuralNet(hidden_size=64, out_size=32)
])

# Use CollateParallel for multiple inputs
collate_fn = CollateParallel(
    vec1_col="legal_dutch",
    vec2_col="regex",
    target_col="onehot",
    padder=FixedPadding(max_chunks=30)
)
```

### Mixture of Experts (MoE)

```python
from vectormesh.components import MeanAggregator, NeuralNet, Serial
from vectormesh.components.gating import MoE

# Create MoE with 4 experts, select top 2
moe = MoE(
    experts=[
        NeuralNet(hidden_size=768, out_size=32),
        NeuralNet(hidden_size=768, out_size=32),
        NeuralNet(hidden_size=768, out_size=32),
        NeuralNet(hidden_size=768, out_size=32),
    ],
    hidden_size=768,
    out_size=32,
    top_k=2
)

pipeline = Serial([MeanAggregator(), moe])
```

### Advanced Aggregation

```python
from vectormesh.components import AttentionAggregator, RNNAggregator

# Use attention-based aggregation (learnable)
pipeline = Serial([
    AttentionAggregator(hidden_size=768),
    NeuralNet(hidden_size=768, out_size=32)
])

# Or use RNN-based aggregation
pipeline = Serial([
    RNNAggregator(hidden_size=768),
    NeuralNet(hidden_size=768, out_size=32)
])
```

### Skip Connections and Gating

```python
from vectormesh.components import Skip, Gate, Highway, Projection

# Skip connection with residual learning
pipeline = Serial([
    Projection(in_size=64, out_size=32),
    Skip(
        transform=NeuralNet(hidden_size=32, out_size=32),
        in_size=32
    )
])

# Simple gating mechanism
pipeline = Serial([
    MeanAggregator(),
    Gate(hidden_size=768),
    NeuralNet(hidden_size=768, out_size=32)
])

# Highway network
pipeline = Serial([
    MeanAggregator(),
    Highway(
        transform=NeuralNet(hidden_size=768, out_size=768),
        hidden_size=768
    ),
    NeuralNet(hidden_size=768, out_size=32)
])
```

## Components

### Data Processing
- `VectorCache`: Efficient storage and retrieval of pre-computed embeddings
- `Vectorizer`: Hugging Face model-based text vectorization
- `RegexVectorizer`: Pattern-based feature extraction
- `LabelEncoder`: Encode categorical labels
- `OneHot`: One-hot encoding for multi-label classification

### Pipeline Components
- `Serial`: Sequential processing of components
- `Parallel`: Parallel processing of multiple input streams

### Aggregation Components
Reduce 3D tensors (batch, chunks, dim) to 2D tensors (batch, dim):
- `MeanAggregator`: Average pooling over chunks (no learnable parameters)
- `AttentionAggregator`: Learnable attention weights over chunks
- `RNNAggregator`: GRU-based sequential aggregation

### Padding Components
- `FixedPadding`: Pad sequences to fixed length
- `DynamicPadding`: Dynamic padding per batch

### Neural Components
- `NeuralNet`: Multi-layer perceptron with dropout
- `Projection`: Linear projection layer
- `Concatenate2D`: Concatenate 2D tensors

### Gating Mechanisms
Control information flow with learnable gates:
- `Skip`: Residual skip connection with layer normalization and optional projection
- `Gate`: Simple multiplicative gating with sigmoid activation
- `Highway`: Highway network combining transformed and original input
- `MoE`: Mixture of Experts with top-k routing and optional noisy gating

## Dataset Creation

The datasets you receive were created using the `build` function, which processes raw data and creates train/test/validation splits. Understanding this process helps you work with the data structure:

```python
from pathlib import Path
from vectormesh import build

# This is what your instructor used to create the datasets
build(
    input_file=Path("assets/data.jsonl"),  # Raw data file
    threshold=50,                           # Minimum label frequency
    trainsplit=0.8,                        # 80% for training
    testvalsplit=0.5,                      # Split remaining 20% equally
    output_dir=Path("assets/")             # Output directory
)
```

The `build` function:
- Filters out labels that appear less than `threshold` times
- Splits data into train/test/validation sets according to the specified ratios
- Saves the splits as Hugging Face datasets in the output directory
- Ensures balanced representation of labels across splits

## Scripts

The `scripts/` directory contains utilities for data preparation and embedding generation:

- `build_dataset.py`: Creates train/test/validation splits from raw data (instructor use)
- `create_cache.py`: Creates vector caches for datasets
- `embed_legal_dutch.py`: Creates embeddings with Dutch legal models
- `embed_multilegal.py`: Creates embeddings with multilingual legal models
- `embed_debertav3.py`: Creates embeddings with DeBERTa models

Example usage for creating caches:
```bash
python scripts/create_cache.py
python scripts/embed_legal_dutch.py
```

## Notebooks

The `notebooks/` directory contains detailed tutorials:

1. **0_vectorizer.ipynb**: Introduction to vectorizers and vector caches
   - Creating embeddings with Hugging Face models
   - Extending caches with regex features
   - Managing vector metadata

2. **1_training.ipynb**: Training models with VectorMesh
   - Loading vector caches
   - Creating dataloaders with padding
   - Building and training pipelines

3. **2_design.ipynb**: Advanced pipeline architectures
   - Parallel processing of multiple vector types
   - Combining embeddings with concatenation
   - Skip connections and gating mechanisms

4. **3_moe.ipynb**: Mixture of Experts implementation
   - MoE architecture and training
   - Expert selection and gating

## Development

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=src/vectormesh --cov-report=html

# Format code
ruff check --fix .
```

## Project Structure

```
vectormesh/
├── src/vectormesh/
│   ├── components/          # Pipeline components
│   │   ├── aggregation.py   # Pooling operations
│   │   ├── connectors.py    # Tensor operations
│   │   ├── gating.py        # Gating mechanisms
│   │   ├── metrics.py       # Evaluation metrics
│   │   ├── neural.py        # Neural network layers
│   │   ├── padding.py       # Sequence padding
│   │   └── pipelines.py     # Pipeline composition
│   ├── data/                # Data processing
│   │   ├── cache.py         # Vector cache management
│   │   ├── dataset.py       # Dataset utilities
│   │   └── vectorizers.py   # Vectorization implementations
│   └── types.py             # Type definitions
├── scripts/                 # Utility scripts
├── notebooks/               # Tutorial notebooks
└── tests/                   # Unit tests
```

## Requirements

- Python >= 3.12
- PyTorch >= 2.9.1
- transformers >= 4.57.3
- sentence-transformers >= 2.0.0
- datasets >= 4.4.2
- mltrainer >= 0.2.7

See `pyproject.toml` for complete dependencies.

## License

See LICENSE file for details.

## Author

Raoul Grouls (Raoul.Grouls@han.nl)
