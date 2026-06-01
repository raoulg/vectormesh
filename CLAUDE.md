# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

VectorMesh is a PyTorch-based framework for efficient vector embedding management and multi-modal text classification. It provides a flexible pipeline architecture for combining different types of text embeddings and building sophisticated neural architectures.

## Development Commands

### Setup
```bash
# Install dependencies using uv (recommended)
uv sync
```

### Testing
```bash
# Run all tests with coverage
uv run pytest

# Run specific test file
uv run pytest path/to/test_file.py

# Run tests excluding slow integration tests
uv run pytest -m "not integration"

# Run only integration tests (slow, requires real models)
uv run pytest -m integration
```

Test configuration is in `pyproject.toml` under `[tool.pytest.ini_options]`. Coverage reports are generated in `htmlcov/`.

### Code Quality
```bash
# Format code with ruff
uv run ruff format .

# Lint and auto-fix issues
uv run ruff check . --fix

# Type checking (using ty)
uv run ty check src/vectormesh/**/*.py
```

### Pre-commit Hooks
The project uses Lefthook (`.lefthook.yml`) with the following checks:
1. **notebooktester**: Validates notebooks execute without errors (240s timeout)
2. **clean-jupyter**: Clears notebook outputs before commit
3. **format**: Auto-formats Python and notebook files with ruff
4. **ruff**: Lints Python files
5. **typecheck**: Type checks with ty

### Notebooks
```bash
# Test all notebooks
uv run notebooktester notebooks -v -t 240

# Clear notebook outputs
uv run jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

## Architecture

### Type System and Runtime Validation

VectorMesh uses **jaxtyping** with **beartype** for runtime tensor shape validation. This is critical for understanding the framework:

- Type annotations explicitly declare tensor shapes: `Float[Tensor, "batch chunks dim"]`
- Runtime validation catches shape mismatches immediately instead of silent PyTorch broadcasting
- All component `forward()` methods use `@jaxtyped(typechecker=beartype)` decorator
- Ruff is configured to ignore F722/F821 errors from jaxtyping syntax (see `pyproject.toml`)

**Important**: When you see beartype errors about tensor dimensions, they indicate actual shape mismatches. Check the function signature to understand expected shapes.

### Core Pipeline Architecture

**Serial and Parallel Composition** (`src/vectormesh/components/pipelines.py`):
- `Serial`: Sequential composition that passes output of one component to the next
- `Parallel`: Parallel branches that process multiple input streams independently
- Both components distinguish between `nn.Module` instances (stored in `nn.ModuleList`) and callable non-modules (stored in `_all_components/_all_branches`)

**Tensor Flow Pattern**:
```
3D tensors (batch, chunks, dim) → Aggregator → 2D tensors (batch, dim) → Neural layers
```

This pattern is fundamental: documents are chunked, embedded into 3D tensors, aggregated to 2D, then processed through neural networks.

### VectorCache System

**VectorCache** (`src/vectormesh/data/cache.py`) is the core data abstraction:

1. **Cache Creation**:
   - Takes a `Vectorizer` and a Hugging Face `Dataset`
   - Processes text through vectorizer in batches
   - Stores embeddings as Hugging Face Dataset with torch format
   - Metadata tracks vectorizer config, hidden sizes, chunk sizes

2. **Cache Extension**:
   - Caches can be extended with additional vector types (e.g., adding regex features to existing embeddings)
   - Uses `update_metadata()` to merge metadata from existing caches
   - Each vector type gets its own column in the dataset

3. **Key Design**: VectorCache inherits from `VectorMeshComponent` (Pydantic BaseModel with `frozen=True` and `arbitrary_types_allowed=True`)

4. **Metadata Structure**:
   ```python
   {
     "<column_name>": {
       "vectormesh_version": str,
       "model_tag": str,
       "vectorizer_type": str,
       "tensordtype": int,  # 1 for 1D, 2 for 2D
       "hidden_size": int,
       "context_size": int,
       "chunk_sizes": Optional[...]
     },
     "features": List[str],
     "created_at": str,
     "num_observations": int
   }
   ```

### Component Categories

**Aggregators** (`src/vectormesh/components/aggregation.py`):
- Reduce 3D → 2D tensors
- `MeanAggregator`: Simple average pooling (no parameters)
- `AttentionAggregator`: Learnable attention weights
- `RNNAggregator`: GRU-based sequential processing

**Gating Mechanisms** (`src/vectormesh/components/gating.py`):
- `Skip`: Residual connection with optional projection and layer normalization
- `Gate`: Simple multiplicative gating with sigmoid
- `Highway`: Highway network combining transformed and original input
- `MoE`: Mixture of Experts with top-k routing and optional noisy gating

**Neural Components** (`src/vectormesh/components/neural.py`):
- `NeuralNet`: Multi-layer perceptron with dropout
- `Projection`: Single linear layer

**Connectors** (`src/vectormesh/components/connectors.py`):
- `Concatenate2D`: Concatenates 2D tensors along feature dimension

**Padding** (`src/vectormesh/components/padding.py`):
- `FixedPadding`: Pad to fixed `max_chunks`
- `DynamicPadding`: Pad to batch maximum

### Data Processing

**Vectorizers** (`src/vectormesh/data/vectorizers.py`):
- `BaseVectorizer`: Abstract base with `col_name`, `model_name`, `get_hidden_size`, `get_context_size`
- `Vectorizer`: Hugging Face model-based vectorization (uses sentence-transformers)
- `RegexVectorizer`: Pattern-based feature extraction with TF-IDF

**Dataset Utilities** (`src/vectormesh/data/dataset.py`):
- `LabelEncoder`: Maps sparse integer codes to dense indices with one-hot encoding
- `OneHot`: Converts sparse labels to one-hot vectors
- `Collate`: Batch processor for single input (embeddings + targets)
- `CollateParallel`: Batch processor for dual inputs (two vector types)
- `build()`: Creates train/test/validation splits with label filtering

### Error Handling

**VectorMeshError** (`src/vectormesh/types.py`):
- Custom exception with optional `hint` and `fix` fields
- Provides educational context for tensor flow issues
- Used throughout codebase for informative error messages

## Project Structure

```
src/vectormesh/
├── components/
│   ├── aggregation.py    # 3D→2D pooling operations
│   ├── connectors.py     # Tensor concatenation
│   ├── gating.py         # Skip, Gate, Highway, MoE
│   ├── neural.py         # NeuralNet, Projection
│   ├── padding.py        # FixedPadding, DynamicPadding
│   ├── pipelines.py      # Serial, Parallel composition
│   └── metrics.py        # Evaluation metrics
├── data/
│   ├── cache.py          # VectorCache management
│   ├── dataset.py        # Dataset utilities and splits
│   └── vectorizers.py    # Vectorization implementations
└── types.py              # VectorMeshComponent, VectorMeshError

scripts/                  # Dataset preparation and embedding generation
notebooks/                # Tutorial notebooks (0-3)
dev/                      # Development notebooks
references/               # Academic papers (Highway Networks, MoE)
```

## Common Patterns

### Creating and Training a Model

1. Load a VectorCache: `cache = VectorCache.load(path)`
2. Apply label encoding: `dataset.map(OneHot(...))`
3. Create collate function with padding: `Collate(embedding_col=..., padder=FixedPadding(...))`
4. Build pipeline: `Serial([MeanAggregator(), NeuralNet(...)])`
5. Train with mltrainer: `Trainer(model=pipeline, ...)`

### Multi-Input Pipelines

```python
# Pattern for parallel processing:
parallel = Parallel([
    Serial([...]),  # Branch 1
    Serial([...]),  # Branch 2
])
pipeline = Serial([
    parallel,           # Process branches
    Concatenate2D(),    # Merge outputs
    NeuralNet(...)      # Final processing
])
```

Use `CollateParallel` for dataloaders with multiple input types.

### Visualizing Pipelines

Use `summarize()` to create interactive visualizations of your pipeline architecture:

```python
from vectormesh import summarize

# For any Serial or Parallel pipeline
pipeline = Serial([
    MeanAggregator(),
    NeuralNet(hidden_size=768, out_size=32)
])

# Creates an interactive HTML graph showing tensor shape flow
summarize(pipeline, output_file="my_pipeline.html")
```

The visualization:
- Shows **left-to-right flow** for Serial pipelines
- Shows **parallel branches** (multiple rows) for Parallel pipelines
- Displays **tensor shapes** extracted from jaxtyping type hints
- Creates interactive HTML using pyvis/networkx
- Shows shape transformations: `(batch chunks dim) → (batch dim) → (batch 32)`

### Adding New Components

1. Inherit from `nn.Module` (not `VectorMeshComponent`)
2. Add `@jaxtyped(typechecker=beartype)` to `forward()`
3. Specify exact tensor shapes in type annotations
4. Use jaxtyping `Float[Tensor, "batch dim"]` syntax for shapes

## Important Notes

- **Never modify `Makefile`**: It's for remote server file transfers (SCP commands)
- **Dataset loading**: Most datasets are pre-created by instructors using `build()` function
- **MoE implementation**: See `references/` folder for "Outrageously Large Neural Networks" paper
- **Highway Networks**: See `references/` folder for paper details
- **Python version**: Requires >= 3.12
- **Immutable configs**: All `VectorMeshComponent` subclasses use Pydantic's `frozen=True`
