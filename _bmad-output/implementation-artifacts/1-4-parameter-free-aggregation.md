# Story 1.4: Parameter-Free Aggregation

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a data scientist,
I want to aggregate cached 2D chunks into 1D vectors using simple strategies,
So that I can train simple classifiers (Linear, MLP) quickly.

## Acceptance Criteria

1. **Given** a loaded `VectorCache` containing 2D chunks (batch, chunks, dim)
   **When** I call `.aggregate(strategy="mean")`
   **Then** it returns a `OneDTensor` (batch, dim) representing the average vector per document

2. **Given** the same cache
   **When** I call `.aggregate(strategy="max")`
   **Then** it returns the max-pooled representation
   **And** shape validation ensures the output is compatible with standard Linear layers

## Tasks / Subtasks

- [x] Task 1: Create Aggregation Component Base (AC: 1, 2)
  - [x] Create `src/vectormesh/components/aggregation.py`
  - [x] Implement base aggregation logic inheriting from `VectorMeshComponent`
  - [x] Add Pydantic fields: `strategy: Literal["mean", "max"]`
  - [x] Implement `forward()` or `__call__()` method with strict type hints

- [x] Task 2: Implement Mean Aggregation (AC: 1)
  - [x] Implement mean pooling across chunks dimension
  - [x] Input shape: `(batch, chunks, dim)` → Output: `(batch, dim)`
  - [x] Use `torch.mean(dim=1)` to average across chunks
  - [x] Verify output shape matches `OneDTensor` type

- [x] Task 3: Implement Max Aggregation (AC: 2)
  - [x] Implement max pooling across chunks dimension
  - [x] Use `torch.max(dim=1).values` to get max across chunks
  - [x] Verify output is compatible with Linear layers
  - [x] Add shape validation to ensure correct dimensions

- [x] Task 4: Type Safety Integration (AC: 1, 2)
  - [x] Add `@jaxtyping.jaxtyped(typechecker=beartype)` decorator
  - [x] Define input as `TwoDTensor` (despite being 3D)
  - [x] Define output as `OneDTensor`
  - [x] Include shape comments in docstrings

- [x] Task 5: VectorCache Integration
  - [x] Add `aggregate()` method to VectorCache class
  - [x] Method should create aggregator and apply to embeddings
  - [x] Return aggregated tensor directly
  - [x] Maintain memory-mapped efficiency (no unnecessary copies)

- [x] Task 6: Error Handling
  - [x] Handle invalid strategy names (not "mean" or "max")
  - [x] Handle empty cache scenarios
  - [x] Wrap errors in `VectorMeshError` with hints/fixes
  - [x] Validate input shapes before aggregation

- [x] Task 7: Testing
  - [x] Unit test: Mean aggregation with known inputs
  - [x] Unit test: Max aggregation with known inputs
  - [x] Unit test: Shape validation
  - [x] Unit test: Invalid strategy error handling
  - [x] Integration test: VectorCache.aggregate() with real cache

## Dev Notes

### Technical Requirements

**Libraries Required:**
- `torch>=2.9.1` - Already installed, primary tensor library
- `jaxtyping>=0.3.4` - Already installed, for type annotations
- `beartype>=0.22.9` - Already installed, for runtime type checking
- `pydantic>=2.12.5` - Already installed, for component configuration

**Aggregation Strategies:**
- **Mean Pooling**: Average embeddings across chunks dimension
  - Formula: `output[b, d] = mean(input[b, :, d])` for all chunks
  - Reduces (B, N, D) → (B, D) by averaging N chunks
- **Max Pooling**: Take maximum value across chunks dimension
  - Formula: `output[b, d] = max(input[b, :, d])` for all chunks
  - Captures strongest signal from any chunk

### Architecture Compliance

**File Structure:**
- Create `src/vectormesh/components/aggregation.py`
- Modify `src/vectormesh/data/cache.py` to add `.aggregate()` method
- Export from `src/vectormesh/components/__init__.py`
- Export from `src/vectormesh/__init__.py`

**Component Design Pattern:**
- MUST inherit from `VectorMeshComponent` (Pydantic BaseModel)
- Configuration MUST be `frozen=True` (immutable)
- Use `model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)`
- Functional style: aggregation returns new tensor, doesn't modify input

**Type Safety:**
- Input: `TwoDTensor` = `Float[Tensor, "batch chunks dim"]` (3D tensor)
- Output: `OneDTensor` = `Float[Tensor, "batch dim"]` (2D tensor)
- Use `@jaxtyping.jaxtyped(typechecker=beartype)` on aggregation methods
- Include `Shapes:` section in docstrings

### Implementation Patterns

**Aggregator Component Pattern:**

```python
from vectormesh.base import VectorMeshComponent
from vectormesh.types import OneDTensor, TwoDTensor
from jaxtyping import jaxtyped
from beartype import beartype
from typing import Literal
import torch

class Aggregator(VectorMeshComponent):
    """Parameter-free aggregation strategies.

    Reduces 3D tensor (batch, chunks, dim) to 2D (batch, dim) using
    simple pooling strategies like mean or max.

    Args:
        strategy: Aggregation method ("mean" or "max")

    Shapes:
        Input: (batch, chunks, dim)
        Output: (batch, dim)
    """

    model_config = ConfigDict(frozen=True)

    strategy: Literal["mean", "max"]

    @jaxtyped(typechecker=beartype)
    def __call__(self, embeddings: TwoDTensor) -> OneDTensor:
        """Apply aggregation strategy.

        Args:
            embeddings: Tensor of shape (batch, chunks, dim)

        Returns:
            Aggregated tensor of shape (batch, dim)
        """
        if self.strategy == "mean":
            return torch.mean(embeddings, dim=1)
        elif self.strategy == "max":
            return torch.max(embeddings, dim=1).values
        else:
            raise VectorMeshError(
                message=f"Unknown strategy: {self.strategy}",
                hint="Valid strategies are 'mean' or 'max'",
                fix="Use strategy='mean' or strategy='max'"
            )
```

**VectorCache Integration:**

```python
# In src/vectormesh/data/cache.py

def aggregate(self, strategy: Literal["mean", "max"] = "mean") -> torch.Tensor:
    """Aggregate embeddings using simple pooling strategy.

    Args:
        strategy: Aggregation method ("mean" or "max")

    Returns:
        Aggregated tensor of shape (batch, dim)

    Shapes:
        Input: get_embeddings() returns (batch, chunks, dim)
        Output: (batch, dim)
    """
    from vectormesh.components.aggregation import Aggregator

    embeddings = self.get_embeddings()
    aggregator = Aggregator(strategy=strategy)
    return aggregator(embeddings)
```

### Previous Story Intelligence

**Learnings from Story 1.3 (VectorCache):**

1. **Lazy Loading with object.__setattr__**: Cache data in frozen Pydantic instances
   - Not needed for aggregation (stateless operation)
   - But follows same immutable config pattern

2. **Error Wrapping**: All exceptions wrapped in `VectorMeshError`
   - Apply to invalid strategy errors
   - Apply to shape mismatch errors

3. **Type Imports**: Import from `beartype.typing`
   - Use `Literal` from `beartype.typing` for strategy type

4. **Testing Pattern**:
   - Use `tmp_path` fixture for file operations
   - Mock external dependencies
   - Test with known inputs for deterministic results

5. **Export Pattern**: Add to `__init__.py` with `__all__`
   - Export `Aggregator` from components
   - Export from main `__init__.py`

**Learnings from Story 1.2 (TextVectorizer):**

1. **Device Management**: Already implemented in TextVectorizer
   - Aggregation is CPU-only operation (no model inference)
   - Works on tensors returned by VectorCache (already on correct device)

2. **Google-style Docstrings**: Include Args, Returns, Shapes
   - Critical for aggregation to document shape transformations

3. **Frozen Configuration**: Pydantic v2 with frozen=True
   - Aggregator should be immutable configuration object

### Architecture Decisions (from architecture.md)

**Aggregation Strategy (ADR):**
> "Decision: Parameter-free pooling (mean, max) as first-class operations"
> "Rationale: 80% of use cases need simple aggregation before Linear layers"
> "Implementation: Composable components compatible with Serial/Parallel"

**Type System (from ADR-001 Type Safety Strategy):**
> "All tensor-transforming components MUST declare input/output shapes using jaxtyping"
> "Shape validation at both definition time (static) and runtime (beartype)"
> "Include Shapes: section in docstrings for clear shape contracts"

**Composition Pattern (from ADR-003 Architecture Patterns):**
> "Components must support >> operator for chaining"
> "Serial([cache, aggregator, linear]) should validate shapes across boundaries"
> "Frozen config: state changes return new objects"

**Critical Rules from project-context.md:**
- NEVER use `os.path`, always use `pathlib.Path` (not applicable here)
- Strict type hints on all functions (CRITICAL for aggregation)
- Google-style docstrings mandatory
- PascalCase for components (`Aggregator`), snake_case for methods (`aggregate`)

### Implementation Strategy Recommendations

**Recommended Implementation Order:**

1. **Start Simple**: Create `Aggregator` component with mean strategy only
2. **Add Max Strategy**: Extend to support max pooling
3. **Add Type Safety**: Apply jaxtyping + beartype decorators
4. **VectorCache Integration**: Add `.aggregate()` convenience method
5. **Add Error Handling**: Wrap invalid strategies and shape errors
6. **Add Tests**: Unit tests for both strategies with known inputs
7. **Integration Test**: Test with real VectorCache from Story 1.3

**Key Design Decisions:**

**Decision: Standalone Component vs VectorCache Method**
- **RECOMMENDED**: Both patterns
- **Rationale**:
  - Standalone `Aggregator` component for composition flexibility
  - VectorCache `.aggregate()` method for convenience
- **Pattern**:
  ```python
  # Compositional approach
  from vectormesh.components.aggregation import Aggregator
  agg = Aggregator(strategy="mean")
  result = agg(cache.get_embeddings())

  # Convenience approach
  result = cache.aggregate(strategy="mean")
  ```

**Decision: Strategy as Configuration vs Separate Classes**
- **RECOMMENDED**: Single class with `strategy` parameter
- **Rationale**: Only 2 strategies, no complex logic differences
- **Alternative**: Could create `MeanAggregator`, `MaxAggregator` subclasses later if needed

**Potential Pitfalls to Avoid:**

1. **Don't**: Use `torch.mean(dim=2)` (wrong dimension)
   - **Why**: Would average across embedding dims instead of chunks
   - **Do**: Use `dim=1` to aggregate across chunks dimension

2. **Don't**: Return `torch.max(embeddings, dim=1)` directly
   - **Why**: Returns namedtuple with (values, indices)
   - **Do**: Use `.values` to get only the max values tensor

3. **Don't**: Modify input tensor in-place
   - **Why**: Violates functional/immutable principles
   - **Do**: Return new tensor, leave input unchanged

4. **Don't**: Skip shape validation
   - **Why**: Silent failures hard to debug
   - **Do**: Use jaxtyping + beartype for automatic validation

5. **Don't**: Assume 2D input
   - **Why**: VectorCache returns 3D tensors (batch, chunks, dim)
   - **Do**: Explicitly handle 3D → 2D reduction

### Testing Strategy

**Unit Tests:**

```python
def test_aggregator_mean_pooling():
    """Test mean aggregation with known values."""
    # Create test tensor (batch=2, chunks=3, dim=4)
    embeddings = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0],
         [2.0, 3.0, 4.0, 5.0],
         [3.0, 4.0, 5.0, 6.0]],
        [[0.0, 1.0, 2.0, 3.0],
         [1.0, 2.0, 3.0, 4.0],
         [2.0, 3.0, 4.0, 5.0]]
    ])

    agg = Aggregator(strategy="mean")
    result = agg(embeddings)

    # Verify shape
    assert result.shape == (2, 4)

    # Verify values (mean of each dim across chunks)
    expected = torch.tensor([
        [2.0, 3.0, 4.0, 5.0],  # Mean of [1,2,3], [2,3,4], [3,4,5], [4,5,6]
        [1.0, 2.0, 3.0, 4.0]   # Mean of [0,1,2], [1,2,3], [2,3,4], [3,4,5]
    ])
    assert torch.allclose(result, expected)
```

**Integration Tests:**

```python
@pytest.mark.integration
def test_vectorcache_aggregate_integration(tmp_path):
    """Test end-to-end: cache creation → aggregation."""
    # Create cache with real vectorizer
    vectorizer = TextVectorizer(model_name="sentence-transformers/all-MiniLM-L6-v2")
    texts = ["Document one", "Document two", "Document three"]

    cache = VectorCache.create(
        texts=texts,
        vectorizer=vectorizer,
        name="test_agg",
        cache_dir=tmp_path
    )

    # Test mean aggregation
    mean_result = cache.aggregate(strategy="mean")
    assert mean_result.shape == (3, 384)  # 3 docs, 384 dims

    # Test max aggregation
    max_result = cache.aggregate(strategy="max")
    assert max_result.shape == (3, 384)

    # Verify results are different (mean != max)
    assert not torch.allclose(mean_result, max_result)
```

### References

- [Epics: Story 1.4](../../planning-artifacts/epics.md#story-14-parameter-free-aggregation)
- [Architecture: Aggregation Patterns](../../planning-artifacts/architecture.md#component-architecture)
- [Architecture: Type Safety](../../planning-artifacts/architecture.md#adr-001-type-safety-strategy)
- [Project Context](../../_bmad-output/project-context.md)
- [PyTorch Reduction Operations](https://pytorch.org/docs/stable/torch.html#reduction-ops)

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)

### Debug Log References

No critical debugging required. Implementation followed TDD RED-GREEN-REFACTOR cycle successfully.

### Completion Notes List

**Implementation Approach:**
- Followed TDD RED-GREEN-REFACTOR cycle as per BMad workflow
- Created 12 unit tests for aggregation components + 3 integration tests for VectorCache
- **REFACTORED after user feedback**: Simplified Open-Closed Principle with minimal boilerplate for extensions
- Used jaxtyping + beartype for strict shape validation
- Full type safety with Pydantic frozen configuration

**Key Implementation Details:**
1. **Open-Closed Principle (User Feedback-Driven)**:
   - Created BaseAggregator with `__call__()` handling all boilerplate (decorators, type safety)
   - Users only implement simple `_aggregate(embeddings: Tensor) -> Tensor` method
   - No decorators or complex type annotations needed for extensions

2. **Separate Aggregator Classes**:
   - `MeanAggregator`: Implements `torch.mean(dim=1)` for averaging across chunks (AC 1)
   - `MaxAggregator`: Implements `torch.max(dim=1).values` for max pooling (AC 2)
   - Each class is ~5 lines of simple, focused code

3. **Dynamic Loading (User Request)**:
   - Added `get_aggregator(strategy: str)` factory function
   - Load aggregators by class name: `get_aggregator("MeanAggregator")`
   - Enables true extensibility - users can add custom aggregators and load by name
   - VectorCache.aggregate() uses dynamic loading: `cache.aggregate(strategy="MeanAggregator")`

4. **Type Safety**:
   - `@jaxtyped(typechecker=beartype)` in BaseAggregator.__call__() only
   - Extensions inherit type safety automatically - no manual decorator application

5. **VectorCache Integration**:
   - Added convenience `.aggregate(strategy="MeanAggregator")` method
   - Dynamically loads aggregator by name for flexibility

6. **Error Handling**:
   - VectorMeshError for invalid aggregator names with helpful hints
   - Validates that loaded classes inherit from BaseAggregator

**Testing Results:**
- All 44 tests passing (12 aggregation + 3 VectorCache + 29 existing)
- Coverage: 89.02% overall (improved from 88.89%)
- Aggregator coverage: 93.33%
- Test coverage includes: MeanAggregator, MaxAggregator, get_aggregator factory, custom extension pattern, Linear layer compatibility

**Architecture Compliance:**
- ✅ Inherits from VectorMeshComponent
- ✅ Frozen Pydantic configuration
- ✅ Google-style docstrings with Shapes sections
- ✅ Full type hints with jaxtyping + beartype
- ✅ Educational error messages via VectorMeshError
- ✅ Export pattern followed (BaseAggregator, MeanAggregator, MaxAggregator, get_aggregator)

**Learnings Applied:**
- Used frozen Pydantic BaseModel pattern from previous stories
- Applied jaxtyping shape annotations for tensor operations
- Responded to user feedback to simplify extension pattern
- Dynamic loading enables future learnable aggregators (Story 3.3)
- Added ruff ignore rules for jaxtyping annotations (F722, F821)

**Extension Point for Users (SIMPLIFIED after feedback):**
Users can now extend BaseAggregator with minimal boilerplate:
```python
class AttentionAggregator(BaseAggregator):
    """Custom attention-based aggregation."""

    def _aggregate(self, embeddings: Tensor) -> Tensor:
        # That's it! Just implement the core logic
        # Type safety handled automatically by __call__()
        attention_weights = self._compute_attention(embeddings)
        return torch.sum(embeddings * attention_weights, dim=1)
```

**User Feedback Incorporated:**
1. ✅ "Make it simpler - extension should be just `return torch.mean(embeddings, dim=1)`"
   - Simplified to `_aggregate()` method with no decorators
2. ✅ "Use dynamic loading - strategy='MeanAggregator' that gets loaded"
   - Added `get_aggregator()` factory function
   - VectorCache.aggregate() uses string-based loading
3. ✅ "Prepare for learnable aggregators (Epic 3.3)"
   - Factory pattern supports future custom aggregators
   - BaseAggregator design allows for parameterized aggregators later

### File List

**Created:**
- `src/vectormesh/components/aggregation.py` - BaseAggregator + MeanAggregator + MaxAggregator + get_aggregator() (198 lines)
- `tests/components/test_aggregation.py` - Comprehensive test suite (185 lines, 12 tests)

**Modified:**
- `src/vectormesh/data/cache.py` - Added `.aggregate(strategy="MeanAggregator")` method with dynamic loading
- `src/vectormesh/components/__init__.py` - Exported BaseAggregator, MeanAggregator, MaxAggregator, get_aggregator
- `src/vectormesh/__init__.py` - Added BaseAggregator, MeanAggregator, MaxAggregator, get_aggregator to public API
- `pyproject.toml` - Added ruff lint ignore rules for jaxtyping
- `tests/data/test_cache.py` - Added 3 integration tests for cache.aggregate() with new strategy names
