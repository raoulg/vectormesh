# Story 1.5: Model Introspection & 2D/3D Support

**Story ID:** 1.5
**Story Key:** 1-5-model-introspection-2d-3d-support
**Epic:** Epic 1 - Core Vector Integration & Tooling
**Status:** review
**Created:** 2026-01-02
**Sprint:** Sprint 1

---

## Story Overview

### User Story

As a VectorMesh user,
I want TextVectorizer to automatically handle both 2D (sentence-transformers) and 3D (chunked/raw transformers) model outputs,
So that I can use any HuggingFace model without worrying about chunking strategies or tensor dimensionality.

### Business Context

**Critical Architecture Fix:** Stories 1.2-1.4 took a shortcut by implementing only sentence-transformers support (2D output), missing the core architectural requirement for chunk-level storage (3D tensors). This story fixes the foundation to support:

1. **2D Models**: sentence-transformers with built-in pooling (one embedding per document)
2. **3D Models**: Raw transformers requiring chunking (multiple embeddings per document)

**User Impact:**
- Models with small context windows (512 tokens) can chunk long documents automatically
- Models with large context windows (8k-32k tokens) work efficiently without unnecessary chunking
- Type system correctly validates 2D vs 3D tensor flows
- VectorCache supports both dimensional formats
- Users get educational hints about when aggregators are needed

### Acceptance Criteria

**AC1: AutoConfig Model Introspection**
**Given** a HuggingFace model ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")
**When** TextVectorizer initializes
**Then** it queries AutoConfig to determine:
- `max_position_embeddings` (context window size)
- `hidden_size` (embedding dimension)
- Model type (sentence-transformer vs raw transformer)
**And** stores this metadata for dimension detection

**AC2: 2D Model Detection and Output**
**Given** a sentence-transformer model (with built-in pooling)
**When** I call `vectorizer(texts)`
**Then** it returns `TwoDTensor[batch, dim]` (one embedding per document)
**And** sets `vectorizer.output_mode = "2d"`
**And** VectorCache stores data as 2D arrays

**AC3: 3D Model Detection with Chunking**
**Given** a raw transformer model OR a model with context window < average document length
**When** I call `vectorizer(texts)` with long documents
**Then** it automatically chunks text into context-window-sized pieces
**And** returns `ThreeDTensor[batch, chunks, dim]`
**And** sets `vectorizer.output_mode = "3d"`
**And** VectorCache stores data as 3D arrays with attention masks

**AC4: Type Safety for Aggregation**
**Given** aggregation module (MeanAggregator, MaxAggregator)
**When** I inspect the type signatures
**Then** `_aggregate()` uses `Float[Tensor, "batch chunks dim"]` (not generic `Tensor`)
**And** beartype validates shape at runtime
**And** get_aggregator() factory rejects invalid names with educational errors

**AC5: Educational Error Messages**
**Given** a VectorCache with 2D data
**When** user tries `.aggregate(strategy="MeanAggregator")`
**Then** raises VectorMeshError with:
- **Hint**: "Your cache contains 2D data (one embedding per document). Aggregation is only needed for 3D data (chunked embeddings)."
- **Fix**: "Use `cache.get_embeddings()` directly, or switch to a 3D model that produces chunks."

**AC6: Curated Model Validation**
**Given** the 10 curated models from PRD
**When** unit tests run
**Then** each model's AutoConfig metadata is validated (context window, dimension, pooling strategy)
**And** tests verify correct 2D/3D detection for each model

---

## Technical Requirements

### Core Functionality

**1. AutoConfig Introspection Module**

Create `src/vectormesh/utils/model_info.py`:

```python
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field
from transformers import AutoConfig
from vectormesh.errors import VectorMeshError

class ModelMetadata(BaseModel):
    """Metadata extracted from HuggingFace AutoConfig.

    Attributes:
        model_id: HuggingFace model ID
        max_position_embeddings: Maximum context window (tokens)
        hidden_size: Embedding dimension
        output_mode: Whether model produces 2D or 3D output
        pooling_strategy: Pooling method for sentence-transformers
    """
    model_id: str
    max_position_embeddings: int
    hidden_size: int
    output_mode: Literal["2d", "3d"]
    pooling_strategy: Optional[Literal["mean", "cls", "max"]] = None

    class Config:
        frozen = True

def get_model_metadata(model_id: str, cache_dir: Optional[Path] = None) -> ModelMetadata:
    """Query HuggingFace AutoConfig for model metadata.

    This function downloads ONLY the config.json (< 10KB) without loading
    the full model weights. Results are cached locally to avoid repeated
    network calls.

    Args:
        model_id: HuggingFace model ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")
        cache_dir: Optional local cache directory for config files

    Returns:
        ModelMetadata with introspected model properties

    Raises:
        VectorMeshError: If model not found or config invalid

    Examples:
        >>> metadata = get_model_metadata("sentence-transformers/all-MiniLM-L6-v2")
        >>> assert metadata.output_mode == "2d"
        >>> assert metadata.max_position_embeddings == 512
    """
    try:
        config = AutoConfig.from_pretrained(
            model_id,
            cache_dir=str(cache_dir) if cache_dir else None
        )
    except Exception as e:
        raise VectorMeshError(
            message=f"Failed to load config for model: {model_id}",
            hint="Check that the model ID is correct and you have internet connectivity.",
            fix=f"Try: `huggingface-cli download {model_id} config.json` to test manually."
        ) from e

    # Detect sentence-transformer vs raw transformer
    is_sentence_transformer = (
        "sentence-transformers" in model_id or
        hasattr(config, "pooling_mode_mean_tokens")  # ST config marker
    )

    output_mode = "2d" if is_sentence_transformer else "3d"
    pooling = _detect_pooling_strategy(config) if output_mode == "2d" else None

    return ModelMetadata(
        model_id=model_id,
        max_position_embeddings=config.max_position_embeddings,
        hidden_size=getattr(config, "hidden_size", config.dim),  # Handle variations
        output_mode=output_mode,
        pooling_strategy=pooling
    )

def _detect_pooling_strategy(config) -> Optional[Literal["mean", "cls", "max"]]:
    """Detect pooling strategy from sentence-transformer config."""
    if hasattr(config, "pooling_mode_mean_tokens") and config.pooling_mode_mean_tokens:
        return "mean"
    elif hasattr(config, "pooling_mode_cls_token") and config.pooling_mode_cls_token:
        return "cls"
    elif hasattr(config, "pooling_mode_max_tokens") and config.pooling_mode_max_tokens:
        return "max"
    return "mean"  # Default fallback
```

**2. Enhanced TextVectorizer with Dimension Detection**

Modify `src/vectormesh/components/vectorizers.py`:

```python
from typing import Literal, Union
from jaxtyping import Float
from torch import Tensor
from vectormesh.types import TwoDTensor, ThreeDTensor
from vectormesh.utils.model_info import get_model_metadata, ModelMetadata
from vectormesh.errors import VectorMeshError

class TextVectorizer(VectorMeshComponent):
    """HuggingFace text vectorizer with automatic 2D/3D detection.

    Automatically detects whether a model produces 2D (sentence-transformers)
    or 3D (raw transformers with chunking) output based on AutoConfig metadata.

    Attributes:
        model_name: HuggingFace model ID
        auto_chunk: Whether to chunk texts longer than context window
        chunk_size: Maximum tokens per chunk (overrides model's max_position_embeddings)
        device: Compute device (auto-detected if None)
    """
    model_name: str
    auto_chunk: bool = True
    chunk_size: Optional[int] = None
    device: Optional[str] = None

    # Internal state (set during __init__)
    _metadata: ModelMetadata = Field(default=None, exclude=True)
    _model: Any = Field(default=None, exclude=True)
    _tokenizer: Any = Field(default=None, exclude=True)

    def model_post_init(self, __context):
        """Initialize model and introspect metadata."""
        # Get model metadata via AutoConfig (fast, no full model download)
        self._metadata = get_model_metadata(self.model_name)

        # Load actual model and tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self._metadata.output_mode == "2d":
            # Sentence-transformer with pooling
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device or "auto")
        else:
            # Raw transformer (requires manual pooling/chunking)
            from transformers import AutoModel
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model = self._model.to(self.device or _auto_detect_device())

    @property
    def output_mode(self) -> Literal["2d", "3d"]:
        """Dimension of output tensors (2D for pooled, 3D for chunked)."""
        return self._metadata.output_mode

    @property
    def embedding_dim(self) -> int:
        """Dimension of embedding vectors."""
        return self._metadata.hidden_size

    @property
    def context_window(self) -> int:
        """Maximum tokens per chunk."""
        return self.chunk_size or self._metadata.max_position_embeddings

    def __call__(self, texts: list[str]) -> Union[TwoDTensor, ThreeDTensor]:
        """Vectorize texts with automatic dimension detection.

        Args:
            texts: List of text strings to vectorize

        Returns:
            TwoDTensor[batch, dim] if output_mode=="2d"
            ThreeDTensor[batch, chunks, dim] if output_mode=="3d"

        Shapes:
            2D output: [B, D] where B=batch_size, D=embedding_dim
            3D output: [B, C, D] where C=max_chunks across batch
        """
        if self.output_mode == "2d":
            return self._vectorize_2d(texts)
        else:
            return self._vectorize_3d(texts)

    def _vectorize_2d(self, texts: list[str]) -> TwoDTensor:
        """Vectorize using sentence-transformers (pooled output)."""
        embeddings = self._model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=self.device
        )
        return embeddings  # Shape: [batch, dim]

    def _vectorize_3d(self, texts: list[str]) -> ThreeDTensor:
        """Vectorize using raw transformers with chunking."""
        all_chunks = []
        max_chunks = 0

        for text in texts:
            # Tokenize and chunk
            tokens = self._tokenizer(
                text,
                truncation=False,
                return_tensors="pt",
                padding=False
            )["input_ids"][0]

            # Split into chunks
            chunks = self._split_into_chunks(tokens, self.context_window)

            # Embed each chunk
            chunk_embeddings = []
            for chunk in chunks:
                outputs = self._model(chunk.unsqueeze(0).to(self._model.device))
                # Use mean pooling over tokens (not CLS)
                embedding = outputs.last_hidden_state.mean(dim=1)  # [1, dim]
                chunk_embeddings.append(embedding)

            # Stack chunks for this document
            doc_chunks = torch.cat(chunk_embeddings, dim=0)  # [num_chunks, dim]
            all_chunks.append(doc_chunks)
            max_chunks = max(max_chunks, len(chunks))

        # Pad all documents to max_chunks
        padded_chunks = []
        for doc_chunks in all_chunks:
            if len(doc_chunks) < max_chunks:
                padding = torch.zeros(
                    max_chunks - len(doc_chunks),
                    self.embedding_dim,
                    device=doc_chunks.device
                )
                doc_chunks = torch.cat([doc_chunks, padding], dim=0)
            padded_chunks.append(doc_chunks)

        result = torch.stack(padded_chunks, dim=0)  # [batch, max_chunks, dim]
        return result

    def _split_into_chunks(self, tokens: Tensor, chunk_size: int) -> list[Tensor]:
        """Split token sequence into fixed-size chunks."""
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i + chunk_size]
            chunks.append(chunk)
        return chunks
```

**3. Fix Aggregation Type Safety**

Modify `src/vectormesh/components/aggregation.py`:

```python
from jaxtyping import Float, jaxtyped
from beartype import beartype
from torch import Tensor
from vectormesh.types import ThreeDTensor, TwoDTensor

class BaseAggregator(VectorMeshComponent):
    """Base class for aggregation strategies.

    Aggregators reduce 3D chunk-level embeddings to 2D document-level embeddings.

    Shapes:
        Input: [batch, chunks, dim] - Multiple embeddings per document
        Output: [batch, dim] - One embedding per document
    """

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        embeddings: Float[Tensor, "batch chunks dim"]  # FIXED: Was generic Tensor
    ) -> Float[Tensor, "batch dim"]:
        """Aggregate chunk-level embeddings to document-level.

        Args:
            embeddings: Chunk-level embeddings [batch, chunks, dim]

        Returns:
            Document-level embeddings [batch, dim]
        """
        if embeddings.dim() != 3:
            raise VectorMeshError(
                message=f"Aggregator expects 3D input, got {embeddings.dim()}D tensor",
                hint="Aggregators work on chunked embeddings (3D tensors). Your data appears to be 2D (already pooled).",
                fix="If using a sentence-transformer, you don't need aggregation. Use `cache.get_embeddings()` directly."
            )

        return self._aggregate(embeddings)

    def _aggregate(
        self,
        embeddings: Float[Tensor, "batch chunks dim"]  # FIXED: Was generic Tensor
    ) -> Float[Tensor, "batch dim"]:
        """Override this method to implement custom aggregation logic."""
        raise NotImplementedError
```

**4. Dimension-Aware VectorCache**

Modify `src/vectormesh/data/cache.py`:

```python
class VectorCache(VectorMeshComponent):
    """Vector cache with automatic 2D/3D support.

    Attributes:
        storage_mode: "2d" for pooled embeddings, "3d" for chunked embeddings
    """
    name: str
    cache_dir: Path = Path(".vmcache")
    _storage_mode: Literal["2d", "3d"] = Field(default=None, exclude=True)
    _dataset: Optional[Any] = Field(default=None, exclude=True)

    @staticmethod
    def create(
        texts: list[str],
        vectorizer: TextVectorizer,
        name: str,
        cache_dir: Path = Path(".vmcache"),
        batch_size: int = 32
    ) -> "VectorCache":
        """Create cache with automatic dimension detection from vectorizer."""
        # Detect storage mode from vectorizer
        storage_mode = vectorizer.output_mode

        # ... (rest of creation logic, storing 2D or 3D based on storage_mode)

        # Store metadata including storage_mode
        metadata = {
            "model_name": vectorizer.model_name,
            "num_samples": len(texts),
            "embedding_dim": vectorizer.embedding_dim,
            "storage_mode": storage_mode,  # NEW
            "max_chunks": max_chunks if storage_mode == "3d" else None,
            "created_at": datetime.now().isoformat()
        }

        return cache

    def aggregate(self, strategy: str) -> TwoDTensor:
        """Aggregate chunks using specified strategy.

        Raises:
            VectorMeshError: If cache is 2D (no aggregation needed)
        """
        if self._storage_mode == "2d":
            raise VectorMeshError(
                message="Cannot aggregate 2D cache - data is already pooled",
                hint="Your cache contains 2D data (one embedding per document from sentence-transformer). Aggregation is only for 3D chunked data.",
                fix="Use `cache.get_embeddings()` directly without aggregation."
            )

        # ... (existing aggregation logic for 3D data)
```

**5. Curated Model Constants**

Create `src/vectormesh/zoo/models.py`:

```python
"""Curated model registry with validated metadata.

All 10 models have been tested via HuggingFace MCP for:
- Context window size (max_position_embeddings)
- Embedding dimension (hidden_size)
- Pooling strategy (for sentence-transformers)
- Dutch language support (for course requirements)
"""

from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class CuratedModel:
    """Metadata for a validated HuggingFace model."""
    model_id: str
    context_window: int
    embedding_dim: int
    output_mode: Literal["2d", "3d"]
    description: str

# MVP Models (4)
MPNET = CuratedModel(
    model_id="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    context_window=512,
    embedding_dim=768,
    output_mode="2d",
    description="Dutch support, 249M downloads, best general performance"
)

QWEN_0_6B = CuratedModel(
    model_id="Qwen/Qwen3-Embedding-0.6B",
    context_window=32768,
    embedding_dim=768,
    output_mode="3d",  # No built-in pooling
    description="32k context, CPU-friendly (600M params), multilingual"
)

LABSE = CuratedModel(
    model_id="sentence-transformers/LaBSE",
    context_window=512,
    embedding_dim=768,
    output_mode="2d",
    description="109 languages, 470M params"
)

MINILM = CuratedModel(
    model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    context_window=512,
    embedding_dim=384,
    output_mode="2d",
    description="Fastest (117M params), good baseline"
)

# Growth Phase Models (6 more)
BGE_GEMMA2 = CuratedModel(
    model_id="BAAI/bge-multilingual-gemma2",
    context_window=8192,
    embedding_dim=3584,
    output_mode="3d",
    description="8k context, multilingual, 9.2B params"
)

E5_MISTRAL = CuratedModel(
    model_id="intfloat/e5-mistral-7b-instruct",
    context_window=32768,
    embedding_dim=4096,
    output_mode="3d",
    description="32k context, English, 7B params"
)

QWEN_8B = CuratedModel(
    model_id="Qwen/Qwen3-Embedding-8B",
    context_window=32768,
    embedding_dim=4096,
    output_mode="3d",
    description="32k context, multilingual, 8B params"
)

DISTILUSE = CuratedModel(
    model_id="sentence-transformers/distiluse-base-multilingual-cased-v2",
    context_window=512,
    embedding_dim=512,
    output_mode="2d",
    description="Distilled, 134M params"
)

# Model groups for testing
MVP_MODELS = [MPNET, QWEN_0_6B, LABSE, MINILM]
GROWTH_MODELS = [BGE_GEMMA2, E5_MISTRAL, QWEN_8B, DISTILUSE]
ALL_MODELS = MVP_MODELS + GROWTH_MODELS
```

---

## Architecture Compliance

### Required Patterns

**1. Component Structure:**
- [x] Inherits from `VectorMeshComponent` (Pydantic v2)
- [x] Frozen configuration (`frozen=True`)
- [x] Google-style docstrings with `Shapes:` sections
- [x] `__call__()` method for primary interface

**2. Type Safety:**
- [x] Full jaxtyping annotations for tensor operations
- [x] `@jaxtyped(typechecker=beartype)` on aggregation methods
- [x] Use `ThreeDTensor` and `TwoDTensor` type aliases (not generic `Tensor`)
- [x] Union return types for dimension polymorphism: `Union[TwoDTensor, ThreeDTensor]`

**3. Error Handling:**
- [x] Raise `VectorMeshError` (not generic Exception)
- [x] Include `hint` field explaining the concept
- [x] Include `fix` field suggesting code changes

**4. File Organization:**
- `src/vectormesh/utils/model_info.py` - AutoConfig introspection
- `src/vectormesh/zoo/models.py` - Curated model constants
- `src/vectormesh/components/vectorizers.py` - Enhanced TextVectorizer
- `src/vectormesh/components/aggregation.py` - Type-fixed aggregators
- `src/vectormesh/data/cache.py` - Dimension-aware VectorCache

### Integration Points

**With Story 1.2 (TextVectorizer):**
- Extends existing TextVectorizer with `output_mode` property
- Adds AutoConfig introspection before model loading
- Maintains backward compatibility for 2D models

**With Story 1.3 (VectorCache):**
- Adds `storage_mode` field to metadata
- VectorCache.create() detects dimension from vectorizer
- `.aggregate()` validates storage_mode before processing

**With Story 1.4 (Aggregation):**
- Fixes type signatures in BaseAggregator
- Adds validation for 3D input tensors
- Educational errors for 2D/3D mismatches

---

## Testing Requirements

### Unit Tests

**Test File:** `tests/utils/test_model_info.py`

```python
def test_get_model_metadata_sentence_transformer():
    """Test AutoConfig introspection for sentence-transformer."""
    metadata = get_model_metadata("sentence-transformers/all-MiniLM-L6-v2")
    assert metadata.output_mode == "2d"
    assert metadata.max_position_embeddings == 512
    assert metadata.hidden_size == 384
    assert metadata.pooling_strategy == "mean"

def test_get_model_metadata_raw_transformer():
    """Test AutoConfig introspection for raw transformer."""
    metadata = get_model_metadata("bert-base-uncased")
    assert metadata.output_mode == "3d"
    assert metadata.max_position_embeddings == 512
    assert metadata.hidden_size == 768
    assert metadata.pooling_strategy is None  # No built-in pooling

def test_get_model_metadata_large_context():
    """Test large context window detection."""
    metadata = get_model_metadata("Qwen/Qwen3-Embedding-0.6B")
    assert metadata.max_position_embeddings == 32768
    assert metadata.output_mode == "3d"

def test_get_model_metadata_invalid_model():
    """Test error handling for nonexistent model."""
    with pytest.raises(VectorMeshError) as exc_info:
        get_model_metadata("invalid/nonexistent-model")

    error = exc_info.value
    assert "failed to load config" in str(error).lower()
    assert "hint" in str(error).lower()
```

**Test File:** `tests/components/test_vectorizers_dimensions.py`

```python
def test_text_vectorizer_2d_output():
    """Test sentence-transformer produces 2D output."""
    vectorizer = TextVectorizer(model_name="sentence-transformers/all-MiniLM-L6-v2")

    assert vectorizer.output_mode == "2d"
    assert vectorizer.embedding_dim == 384

    texts = ["short text", "another short text"]
    result = vectorizer(texts)

    assert result.dim() == 2  # [batch, dim]
    assert result.shape == (2, 384)

def test_text_vectorizer_3d_output_chunking():
    """Test raw transformer with chunking produces 3D output."""
    vectorizer = TextVectorizer(
        model_name="bert-base-uncased",
        auto_chunk=True,
        chunk_size=512
    )

    assert vectorizer.output_mode == "3d"
    assert vectorizer.context_window == 512

    # Create text that requires multiple chunks
    long_text = " ".join(["word"] * 1000)  # ~1000 tokens
    texts = [long_text, long_text]

    result = vectorizer(texts)

    assert result.dim() == 3  # [batch, chunks, dim]
    assert result.shape[0] == 2  # batch
    assert result.shape[1] >= 2  # multiple chunks
    assert result.shape[2] == 768  # BERT dimension

def test_text_vectorizer_3d_output_padding():
    """Test 3D output pads variable-length documents."""
    vectorizer = TextVectorizer(model_name="bert-base-uncased")

    short_text = " ".join(["word"] * 100)  # 1 chunk
    long_text = " ".join(["word"] * 1000)  # 2+ chunks
    texts = [short_text, long_text]

    result = vectorizer(texts)

    assert result.shape[0] == 2  # batch
    assert result.shape[1] > 1  # max chunks across batch
    # First document padded to match second
```

**Test File:** `tests/components/test_aggregation_types.py`

```python
def test_aggregator_requires_3d_input():
    """Test aggregators validate 3D input shape."""
    agg = MeanAggregator()

    # 2D input should fail
    embeddings_2d = torch.randn(8, 384)
    with pytest.raises(VectorMeshError) as exc_info:
        agg(embeddings_2d)

    error = exc_info.value
    assert "expects 3d input" in str(error).lower()
    assert "hint" in str(error).lower()
    assert "sentence-transformer" in str(error).lower()

def test_aggregator_accepts_3d_input():
    """Test aggregators process 3D input correctly."""
    agg = MeanAggregator()

    # 3D input should succeed
    embeddings_3d = torch.randn(8, 5, 384)  # [batch, chunks, dim]
    result = agg(embeddings_3d)

    assert result.shape == (8, 384)  # [batch, dim]
```

**Test File:** `tests/data/test_cache_dimensions.py`

```python
def test_cache_create_2d_mode(tmp_path):
    """Test VectorCache stores 2D data from sentence-transformer."""
    vectorizer = TextVectorizer(model_name="sentence-transformers/all-MiniLM-L6-v2")
    texts = ["text1", "text2"]

    cache = VectorCache.create(
        texts=texts,
        vectorizer=vectorizer,
        name="test_2d",
        cache_dir=tmp_path
    )

    assert cache._storage_mode == "2d"

    # Loading embeddings should return 2D
    embeddings = cache.get_embeddings()
    assert embeddings.dim() == 2
    assert embeddings.shape == (2, 384)

def test_cache_create_3d_mode(tmp_path):
    """Test VectorCache stores 3D data from raw transformer."""
    vectorizer = TextVectorizer(model_name="bert-base-uncased")
    texts = ["short", " ".join(["word"] * 1000)]  # Variable lengths

    cache = VectorCache.create(
        texts=texts,
        vectorizer=vectorizer,
        name="test_3d",
        cache_dir=tmp_path
    )

    assert cache._storage_mode == "3d"

    # Loading chunks should return 3D
    chunks = cache.get_chunks()
    assert chunks.dim() == 3
    assert chunks.shape[0] == 2  # batch

def test_cache_aggregate_2d_raises_error(tmp_path):
    """Test aggregating 2D cache raises educational error."""
    vectorizer = TextVectorizer(model_name="sentence-transformers/all-MiniLM-L6-v2")
    cache = VectorCache.create(
        texts=["text"],
        vectorizer=vectorizer,
        name="test_2d_error",
        cache_dir=tmp_path
    )

    with pytest.raises(VectorMeshError) as exc_info:
        cache.aggregate(strategy="MeanAggregator")

    error = exc_info.value
    assert "cannot aggregate 2d cache" in str(error).lower()
    assert "hint" in str(error).lower()
    assert "get_embeddings()" in str(error).lower()
```

**Test File:** `tests/zoo/test_curated_models.py`

```python
@pytest.mark.parametrize("model", MVP_MODELS)
def test_mvp_model_metadata_matches_autoconfig(model):
    """Test curated model metadata matches AutoConfig query."""
    metadata = get_model_metadata(model.model_id)

    assert metadata.max_position_embeddings == model.context_window
    assert metadata.hidden_size == model.embedding_dim
    assert metadata.output_mode == model.output_mode

def test_all_curated_models_loadable():
    """Test all 10 curated models can be introspected."""
    for model in ALL_MODELS:
        metadata = get_model_metadata(model.model_id)
        assert metadata is not None
        assert metadata.max_position_embeddings > 0
        assert metadata.hidden_size > 0
```

### Integration Tests

**Test File:** `tests/integration/test_2d_3d_workflow.py`

```python
@pytest.mark.integration
def test_full_workflow_2d_model(tmp_path):
    """Integration test: 2D model → cache → load → classify."""
    # 1. Create cache with 2D model
    vectorizer = TextVectorizer(model_name="sentence-transformers/all-MiniLM-L6-v2")
    texts = ["document 1", "document 2", "document 3"]

    cache = VectorCache.create(
        texts=texts,
        vectorizer=vectorizer,
        name="test_2d_workflow",
        cache_dir=tmp_path
    )

    # 2. Load cache and get embeddings (no aggregation needed)
    loaded_cache = VectorCache.load(name="test_2d_workflow", cache_dir=tmp_path)
    embeddings = loaded_cache.get_embeddings()

    assert embeddings.shape == (3, 384)  # [batch, dim]

    # 3. Use with Linear classifier
    classifier = torch.nn.Linear(384, 10)
    output = classifier(embeddings)
    assert output.shape == (3, 10)

@pytest.mark.integration
def test_full_workflow_3d_model(tmp_path):
    """Integration test: 3D model → cache → load → aggregate → classify."""
    # 1. Create cache with 3D model
    vectorizer = TextVectorizer(model_name="bert-base-uncased")
    texts = [" ".join(["word"] * 500) for _ in range(3)]  # Long docs

    cache = VectorCache.create(
        texts=texts,
        vectorizer=vectorizer,
        name="test_3d_workflow",
        cache_dir=tmp_path
    )

    # 2. Load cache and aggregate chunks
    loaded_cache = VectorCache.load(name="test_3d_workflow", cache_dir=tmp_path)
    embeddings = loaded_cache.aggregate(strategy="MeanAggregator")

    assert embeddings.shape == (3, 768)  # [batch, dim]

    # 3. Use with Linear classifier
    classifier = torch.nn.Linear(768, 10)
    output = classifier(embeddings)
    assert output.shape == (3, 10)
```

### Test Coverage Requirements

- Unit test coverage: ≥90% for new modules
- Integration tests: All 4 MVP models validated
- Type checking: pyright strict mode passes
- Linting: ruff passes with zero violations

---

## Implementation Tasks

### Task 1: AutoConfig Introspection Module

**Subtasks:**
1. Create `src/vectormesh/utils/model_info.py`
2. Implement `ModelMetadata` Pydantic model
3. Implement `get_model_metadata()` function with AutoConfig loading
4. Implement `_detect_pooling_strategy()` helper
5. Add educational error messages for config load failures
6. Write unit tests for sentence-transformer detection
7. Write unit tests for raw transformer detection
8. Write unit tests for invalid model handling
9. Validate with all 10 curated models

**Acceptance:**
- All unit tests pass
- AutoConfig queries complete in <2 seconds
- Error messages include HF MCP hints

### Task 2: Enhanced TextVectorizer with Dimension Detection

**Subtasks:**
1. Add `_metadata: ModelMetadata` field to TextVectorizer
2. Call `get_model_metadata()` in `model_post_init()`
3. Add `output_mode`, `embedding_dim`, `context_window` properties
4. Implement `_vectorize_2d()` for sentence-transformers
5. Implement `_vectorize_3d()` with chunking logic
6. Implement `_split_into_chunks()` helper
7. Update `__call__()` to route based on output_mode
8. Update return type to `Union[TwoDTensor, ThreeDTensor]`
9. Write unit tests for 2D output validation
10. Write unit tests for 3D chunking and padding
11. Write unit tests for dimension detection

**Acceptance:**
- 2D models produce shape [batch, dim]
- 3D models produce shape [batch, chunks, dim]
- Chunking respects context_window
- Variable-length documents padded correctly

### Task 3: Fix Aggregation Type Safety

**Subtasks:**
1. Update BaseAggregator.__call__() signature to use `Float[Tensor, "batch chunks dim"]`
2. Update `_aggregate()` signature similarly
3. Add 3D input validation in __call__()
4. Add educational error for 2D input
5. Update MeanAggregator with fixed types
6. Update MaxAggregator with fixed types
7. Add `@jaxtyped` decorator with beartype
8. Write unit tests for 3D input validation
9. Write unit tests for 2D input rejection
10. Run pyright strict mode validation

**Acceptance:**
- pyright strict mode passes
- beartype catches shape violations at runtime
- Educational errors guide users correctly

### Task 4: Dimension-Aware VectorCache

**Subtasks:**
1. Add `_storage_mode: Literal["2d", "3d"]` field
2. Update `VectorCache.create()` to detect storage_mode from vectorizer
3. Store storage_mode in metadata.json
4. Load storage_mode in `VectorCache.load()`
5. Update `.aggregate()` to validate storage_mode == "3d"
6. Add educational error for aggregating 2D cache
7. Update `.get_embeddings()` to work with both modes
8. Add `.get_chunks()` method for 3D access
9. Write unit tests for 2D cache creation
10. Write unit tests for 3D cache creation
11. Write unit tests for aggregate() validation
12. Write integration tests for full 2D and 3D workflows

**Acceptance:**
- VectorCache supports both 2D and 3D storage
- Metadata correctly identifies storage_mode
- Educational errors prevent misuse

### Task 5: Curated Model Registry

**Subtasks:**
1. Create `src/vectormesh/zoo/models.py`
2. Define `CuratedModel` dataclass
3. Add all 4 MVP models with metadata
4. Add 6 Growth Phase models (for completeness)
5. Create `MVP_MODELS`, `GROWTH_MODELS`, `ALL_MODELS` lists
6. Write unit tests validating each model's metadata
7. Write integration test loading all models
8. Update __init__.py exports

**Acceptance:**
- All 10 models have correct metadata
- AutoConfig validation matches constants
- Models are easily importable

### Task 6: Testing & Validation

**Subtasks:**
1. Write all unit tests (9 test files)
2. Write integration tests (1 test file)
3. Run full test suite: `uv run pytest`
4. Validate type checking: `uv run pyright`
5. Validate linting: `uv run ruff check`
6. Measure coverage: Should be ≥90% for new modules
7. Test with all 4 MVP models
8. Fix any failures or type errors
9. Update documentation with examples

**Acceptance:**
- All tests pass
- pyright strict mode: zero errors
- ruff: zero violations
- Coverage ≥90% for new modules

---

## Previous Story Learnings

### From Story 1.4 (Parameter-Free Aggregation)

**TDD RED-GREEN-REFACTOR Cycle:**
- ✅ Write failing tests first
- ✅ Implement minimal code to pass
- ✅ Refactor based on user feedback
- **Apply**: Use same workflow for this story

**Open-Closed Principle Pattern:**
- ✅ BaseAggregator handles all boilerplate
- ✅ Extensions only implement `_aggregate()`
- ✅ Dynamic loading via factory function
- **Apply**: ModelMetadata can be extended similarly

**Type Safety with jaxtyping:**
- ✅ `@jaxtyped(typechecker=beartype)` catches shape errors
- ✅ Use specific tensor types (not generic Tensor)
- ✅ Add ruff ignore rules for F722, F821
- **Apply**: Fix aggregation to use `Float[Tensor, "batch chunks dim"]`

**Educational Error Messages:**
- ✅ VectorMeshError with `hint` and `fix` fields
- ✅ Explain concepts, not just failures
- **Apply**: Add educational errors for 2D/3D mismatches

### From Story 1.3 (VectorCache)

**Atomic Cache Creation:**
- ✅ Write to temp directory first
- ✅ Rename only on success
- ✅ Clean up temp on failure
- **Apply**: Maintain this pattern for both 2D and 3D modes

**Content-Hash Versioning:**
- ✅ Hash metadata prevents stale caches
- **Apply**: Include storage_mode in hash

### From Story 1.2 (TextVectorizer)

**Device Auto-Detection:**
- ✅ Detect GPU/MPS/CPU automatically
- **Apply**: Ensure works for both sentence-transformers and raw transformers

**Progress Tracking:**
- ✅ Use tqdm for long operations
- **Apply**: Show progress during chunk processing

---

## Code Quality Checklist

### Type Safety
- [ ] Full type hints for all public functions
- [ ] `@jaxtyped(typechecker=beartype)` on tensor operations
- [ ] Use `ThreeDTensor`, `TwoDTensor` (not generic `Tensor`)
- [ ] pyright strict mode passes with zero errors
- [ ] Union types for dimension polymorphism

### Documentation
- [ ] Google-style docstrings on all public methods
- [ ] Include `Shapes:` section for tensor operations
- [ ] Examples in docstrings showing 2D and 3D usage
- [ ] Educational error messages with `hint` and `fix`

### Testing
- [ ] Unit test coverage ≥90% for new modules
- [ ] Integration tests for 2D and 3D workflows
- [ ] All 4 MVP models validated
- [ ] Type checking passes
- [ ] Linting passes

### Architecture
- [ ] Inherits from VectorMeshComponent where appropriate
- [ ] Frozen Pydantic configuration
- [ ] Follows SRP (Single Responsibility Principle)
- [ ] Open-Closed Principle for extensions

---

## Definition of Done

- [ ] All 6 implementation tasks completed
- [ ] All unit and integration tests passing
- [ ] Test coverage ≥90% for new modules
- [ ] pyright strict mode: zero errors
- [ ] ruff linting: zero violations
- [ ] All 4 MVP models validated with AutoConfig
- [ ] 2D and 3D workflows tested end-to-end
- [ ] Educational error messages validated by user
- [ ] Code reviewed and approved
- [ ] Documentation updated
- [ ] Sprint status updated to "review"

---

## References

- [Epic 1: Core Vector Integration & Tooling](../../planning-artifacts/epics.md#epic-1)
- [PRD: 10 Curated Models](../../planning-artifacts/prd.md#curated-model-library-10-models)
- [Research: Model Loading & Introspection](../../planning-artifacts/research/technical-model-loading-and-vectorizers-research-2026-01-01.md)
- [Architecture: Chunk-Level Storage](../../planning-artifacts/architecture.md#data-architecture)
- [Architecture: Type Safety](../../planning-artifacts/architecture.md#adr-001-type-safety-strategy)
- [Project Context](../../project-context.md)
- [Story 1.4: Parameter-Free Aggregation](./1-4-parameter-free-aggregation.md)
- [HuggingFace AutoConfig Documentation](https://huggingface.co/docs/transformers/v4.46.0/model_doc/auto#transformers.AutoConfig)

---

## Dev Agent Record

### Implementation Status

**Status:** ready-for-dev
**Created:** 2026-01-02
**Agent:** Claude Sonnet 4.5 (create-story workflow)

### Story Creation Notes

**Architecture Issue Identified:**
- Stories 1.2-1.4 implemented sentence-transformers only (2D output)
- Architecture specified chunk-level storage (3D tensors)
- Aggregation module uses wrong types (generic `Tensor` instead of jaxtyped)
- VectorCache doesn't support dimensional polymorphism

**User's 4-Step Solution Applied:**
1. ✅ AutoConfig introspection for model capabilities
2. ✅ Clear 2D vs 3D distinction with typing system
3. ✅ Educational errors about when aggregators are needed
4. ✅ Curated model registry with validated metadata

**Curated Models Integrated:**
- MVP: 4 models (mpnet, Qwen3-0.6B, LaBSE, MiniLM)
- Growth: 6 more models (total 10)
- All validated via AutoConfig metadata

**Key Design Decisions:**
- `Union[TwoDTensor, ThreeDTensor]` return type for TextVectorizer
- `storage_mode` field in VectorCache metadata
- Educational errors prevent 2D/3D mismatches
- AutoConfig queries only config.json (fast, no full model download)

**Integration Points:**
- Extends Story 1.2 (TextVectorizer)
- Fixes Story 1.4 (Aggregation types)
- Enhances Story 1.3 (VectorCache)
- Unblocks Story 1.5 (MCP Server - moved to Epic 5)

---

**Story ready for dev-story workflow.**
