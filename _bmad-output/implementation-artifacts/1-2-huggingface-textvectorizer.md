# Story 1.2: HuggingFace TextVectorizer

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a researcher,
I want to convert text to vectors using HuggingFace models without managing the model lifecycle,
So that I can focus on the vectors, not the infrastructure.

## Acceptance Criteria

1. **Given** a `TextVectorizer` initialized with a supported model name (e.g., "sentence-transformers/all-MiniLM-L6-v2")
   **When** I call it with a list of strings
   **Then** it automatically downloads the model (if missing)
   **And** moves it to the correct device (MPS on Mac, CUDA if available, else CPU)
   **And** returns a typed `OneDTensor` (batch, dim) of embeddings

2. **Given** an invalid model string
   **When** I initialize `TextVectorizer`
   **Then** it raises a descriptive `VectorMeshError` (not a generic HF error)

## Tasks / Subtasks

- [x] Task 1: Implement `TextVectorizer` Component (AC: 1, 2)
  - [x] Create `src/vectormesh/components/vectorizers.py`
  - [x] Implement `TextVectorizer` inheriting from `VectorMeshComponent`
  - [x] Add Pydantic fields: `model_name: str`, `device: Optional[str] = None`
  - [x] Implement `__call__` method accepting `List[str]` and returning `TwoDTensor`

- [x] Task 2: Implement Device Auto-Detection (AC: 1)
  - [x] Create device detection logic in `__init__` or setup method
  - [x] Check for MPS availability (macOS with Apple Silicon)
  - [x] Check for CUDA availability
  - [x] Fallback to CPU if neither available
  - [x] Allow user override via `device` parameter

- [x] Task 3: Model Loading Strategy (AC: 1, 2)
  - [x] Use `SentenceTransformer` from `sentence-transformers` library as primary approach
  - [x] Handle automatic model download from HuggingFace Hub
  - [x] Wrap HuggingFace errors in `VectorMeshError` with helpful hints
  - [x] Move model to detected/specified device
  - [x] Cache model instance to avoid reloading on every call

- [x] Task 4: Batch Embedding Generation (AC: 1)
  - [x] Implement batch processing using `model.encode()`
  - [x] Convert output to PyTorch tensor
  - [x] Apply jaxtyping type annotations for shape validation
  - [x] Return `TwoDTensor` with shape `(batch, dim)`

- [x] Task 5: Error Handling & Educational Messages (AC: 2)
  - [x] Catch model loading errors (network, invalid model ID)
  - [x] Catch encoding errors (empty input, incompatible types)
  - [x] Provide hints like "Check model name on HuggingFace Hub"
  - [x] Provide fixes like "Try using 'sentence-transformers/all-MiniLM-L6-v2'"

- [x] Task 6: Testing
  - [x] Unit test: Mock `SentenceTransformer` to avoid network calls
  - [x] Unit test: Verify device detection logic
  - [x] Unit test: Verify error handling for invalid model
  - [x] Integration test (optional): Use tiny model like "prajjwal1/bert-tiny" for real encoding

## Dev Notes

### Technical Requirements

**Libraries Required:**
- `sentence-transformers>=2.0` - Primary library for text embedding models
- `torch` - Already installed, needed for tensor operations and device management
- `transformers` - Dependency of sentence-transformers, provides AutoModel/AutoTokenizer

**Version Notes:**
- `sentence-transformers/all-MiniLM-L6-v2` is the reference model (384 dimensions, 144M downloads)
- Library supports automatic device management via `device` parameter
- Returns NumPy arrays by default, need conversion to PyTorch tensors

### Architecture Compliance

**File Structure:**
- Create `src/vectormesh/components/` directory if it doesn't exist
- Create `src/vectormesh/components/__init__.py`
- Implement in `src/vectormesh/components/vectorizers.py`
- Export from `src/vectormesh/__init__.py`

**Component Design Pattern:**
- MUST inherit from `VectorMeshComponent` (Pydantic BaseModel)
- Configuration MUST be `frozen=True` (immutable)
- Use `model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)`
- The actual ML model should be stored as a private attribute or computed property

**Type Safety:**
- Input: `List[str]` (list of text strings)
- Output: `TwoDTensor` which is `Float[Tensor, "batch dim"]`
- Use `@jaxtyping.jaxtyped(typechecker=beartype)` decorator on `__call__`

**Error Strategy:**
- All errors MUST inherit from `VectorMeshError`
- Include `hint` field: Brief explanation of what likely went wrong
- Include `fix` field: Actionable suggestion for the user
- Example: `VectorMeshError(msg="Model not found", hint="Invalid HuggingFace model ID", fix="Check model name at https://huggingface.co/models")`

### Library-Specific Implementation Guidance

**SentenceTransformer Usage (from Context7 research):**

```python
from sentence_transformers import SentenceTransformer

# Loading with explicit device
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

# Encoding sentences (returns numpy array)
embeddings = model.encode(sentences)  # shape: (batch, embedding_dim)

# Device options: "cuda", "cpu", "mps", or list for multi-GPU
```

**Device Detection Pattern:**
```python
import torch

def _detect_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
```

**Model Loading Safety:**
- SentenceTransformer handles safe loading internally
- No need for `weights_only=True` - that's for raw `torch.load()`
- SentenceTransformer uses HuggingFace's safe loading mechanisms

### Previous Story Intelligence

**Learnings from Story 1.1 (Core Types & Component Base):**

1. **File Organization**: Previous story created:
   - `src/vectormesh/base.py` - Contains `VectorMeshComponent`
   - `src/vectormesh/types.py` - Contains `OneDTensor`, `TwoDTensor`, `ThreeDTensor`
   - `src/vectormesh/errors.py` - Contains `VectorMeshError` with `hint` and `fix` fields
   - `src/vectormesh/utils.py` - Contains `check_shapes` decorator

2. **Testing Patterns**:
   - Tests mirror source structure: `tests/test_*.py` for `src/vectormesh/*.py`
   - Used fixtures in `conftest.py` for shared test setup
   - Achieved 100% pass rate with 4 unit tests

3. **Export Pattern**: All public APIs exported in `src/vectormesh/__init__.py` with `__all__`

4. **Code Review Learnings**:
   - Exports are critical - don't forget to add to `__init__.py`
   - Integrate custom errors throughout (not just define them)
   - Test against API contracts, not implementation details

5. **Development Tools Used**:
   - Google Gemini for planning and implementation
   - Auto-fixed issues after code review
   - Maintained change log in story file

### Git Intelligence Summary

**Recent Commit Analysis:**
- Last commit: "sprint planning" (0c34a79) - Added sprint status and epics
- Project is in early stages, story 1.1 was completed (status: done)
- No production code commits yet beyond the foundation

**Code Patterns Established:**
- Using `src/` layout (standard Python packaging)
- Pydantic v2 for configuration validation
- Jaxtyping + Beartype for runtime shape checking
- Custom error hierarchy with educational messages

### Latest Technical Information (2026-01-01)

**Sentence Transformers Library:**
- **Most Popular Model**: `sentence-transformers/all-MiniLM-L6-v2`
  - 144.2M downloads, 4290 likes
  - 384-dimensional embeddings
  - Excellent for semantic similarity tasks
  - Supports PyTorch, TensorFlow, Rust, ONNX

**Device Management Best Practices:**
- Use `device` parameter in SentenceTransformer constructor
- Options: `"cuda"`, `"cpu"`, `"mps"`, or list for multi-GPU
- Multi-GPU support: Pass `device=["cuda:0", "cuda:1"]` to `encode()`
- Library handles device placement automatically

**API Patterns:**
- `model.encode(sentences)` returns NumPy array by default
- Shape: `(num_sentences, embedding_dim)`
- Convert to PyTorch: `torch.from_numpy(embeddings)`
- Supports batch processing with `batch_size` parameter

**Error Handling:**
- Invalid model names raise `OSError` from HuggingFace Hub
- Network errors raise `requests.exceptions.ConnectionError`
- Empty input handling varies by model - validate before encoding

**Alternative Approach (if needed):**
- Can use `transformers.AutoModel` + `AutoTokenizer` for more control
- Requires manual pooling strategy (mean, max, CLS token)
- SentenceTransformer is recommended as it handles this automatically

### Project Context Reference

**From `_bmad-output/project-context.md`:**

**MUST FOLLOW:**
- Strict type hints on all functions
- Use `@jaxtyping.jaxtyped(typechecker=beartype)` for tensor methods
- Google-style docstrings mandatory
- Prefer `einops` for reshaping (though not needed for this story)
- NEVER use `os.path`, always use `pathlib.Path`
- PascalCase for components, snake_case for methods/variables

**Testing:**
- Unit tests MUST NOT make network calls
- Mock HuggingFace API or use tiny test models
- Integration tests should use `pytest.mark.integration`
- Mirror `src/` structure in `tests/`

**Security:**
- Safe model loading is handled by SentenceTransformer library
- No need to manually set `weights_only=True` for this use case

### Implementation Strategy Recommendations

**Recommended Implementation Order:**

1. **Start Simple**: Create basic `TextVectorizer` with hardcoded CPU device
2. **Add Device Detection**: Implement auto-detection logic
3. **Add Error Handling**: Wrap all HuggingFace errors
4. **Add Type Safety**: Apply jaxtyping decorators
5. **Add Tests**: Mock SentenceTransformer, test error cases
6. **Integration Test**: Optional test with real tiny model

**Key Decision: SentenceTransformer vs AutoModel**
- **RECOMMENDED**: Use `SentenceTransformer` from `sentence-transformers` library
- **Rationale**:
  - Handles tokenization, encoding, and pooling automatically
  - Purpose-built for embedding generation
  - Simpler API, less code to maintain
  - Matches PRD's goal: "without managing model lifecycle"

**Potential Pitfalls to Avoid:**

1. **Don't**: Store model in Pydantic field directly
   - **Why**: Pydantic frozen models can't have mutable objects
   - **Do**: Use `__post_init__` or cached property to load model

2. **Don't**: Return NumPy arrays from `__call__`
   - **Why**: Type system expects PyTorch tensors
   - **Do**: Convert with `torch.from_numpy()`

3. **Don't**: Let HuggingFace errors bubble up
   - **Why**: Violates educational error requirement
   - **Do**: Catch and wrap in `VectorMeshError` with hints

4. **Don't**: Download models on import
   - **Why**: Slow startup, unexpected behavior
   - **Do**: Lazy load on first `__call__`

### References

- [Epics: Story 1.2](../../planning-artifacts/epics.md#story-12-huggingface-textvectorizer)
- [Architecture: Component Patterns](../../planning-artifacts/architecture.md#component-pattern-the-unit-of-work)
- [Architecture: Error Handling](../../planning-artifacts/architecture.md#error-handling-patterns)
- [Project Context](../../_bmad-output/project-context.md)
- [HuggingFace Model: all-MiniLM-L6-v2](https://hf.co/sentence-transformers/all-MiniLM-L6-v2)
- [SentenceTransformers Documentation](https://www.sbert.net/)

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)

### Debug Log References

N/A - No debugging required

### Completion Notes List

- ✅ Implemented `TextVectorizer` component with full Pydantic v2 configuration
- ✅ Device auto-detection: CUDA → MPS → CPU with user override capability
- ✅ Lazy-loading model pattern using `object.__setattr__` to bypass frozen config
- ✅ Comprehensive error handling wrapping OSError and generic exceptions in `VectorMeshError`
- ✅ All errors include educational `hint` and `fix` fields
- ✅ Batch embedding generation with NumPy-to-PyTorch tensor conversion
- ✅ 10 comprehensive unit tests with mocking (100% pass rate)
- ✅ Fixed beartype deprecation warning by importing List from beartype.typing
- ✅ All linting checks pass (ruff)
- ✅ No regressions in existing test suite (17/17 tests pass)
- ✅ Exported `TextVectorizer` from main `__init__.py`

**Implementation Approach:**
- Followed RED-GREEN-REFACTOR TDD cycle
- Used `_get_model()` private method for lazy loading and caching
- Leveraged `object.__setattr__` to cache model in frozen Pydantic instance
- Used `TwoDTensor` type alias instead of raw jaxtyping annotation for linter compatibility

### File List

- `src/vectormesh/components/__init__.py` (created)
- `src/vectormesh/components/vectorizers.py` (created)
- `src/vectormesh/__init__.py` (modified - added TextVectorizer export)
- `tests/components/__init__.py` (created)
- `tests/components/test_vectorizers.py` (created)
- `pyproject.toml` (modified - added sentence-transformers>=2.0.0 dependency)
