# Story 1.3: VectorCache Persistence & Storage

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a user,
I want to cache embeddings for my dataset (e.g., `assets/train.jsonl`),
So that I don't re-compute expensive vectors during every experiment.

## Acceptance Criteria

1. **Given** a list of texts loaded from `assets/train.jsonl`
   **When** I call `VectorCache.create(texts, vectorizer, name="my_cache")`
   **Then** it processes the texts in batches
   **And** saves the results as a partitioned Parquet dataset in `.vmcache/my_cache`
   **And** the directory contains `embeddings` (Arrow Array3D) and `metadata.json`

2. **Given** an interrupted cache creation process (e.g. SIGKILL)
   **When** I check the `.vmcache` directory
   **Then** the target directory is clean (does not exist or is empty), no partial corruption

## Tasks / Subtasks

- [x] Task 1: Implement `VectorCache` Component (AC: 1)
  - [x] Create `src/vectormesh/data/cache.py`
  - [x] Implement `VectorCache` class inheriting from `VectorMeshComponent`
  - [x] Add Pydantic fields: `name: str`, `cache_dir: Path = Path(".vmcache")`
  - [x] Implement `create()` class method accepting texts, vectorizer, and name

- [x] Task 2: Batch Processing Logic (AC: 1)
  - [x] Implement batch processing of texts through vectorizer
  - [x] Configurable batch size (default: 32)
  - [x] Progress tracking for long-running operations
  - [x] Memory-efficient processing (don't load all embeddings at once)

- [x] Task 3: HuggingFace Datasets Integration (AC: 1)
  - [x] Convert embeddings to HuggingFace Dataset format
  - [x] Use `Dataset.from_dict()` with embeddings array
  - [x] Save using `dataset.save_to_disk()` to cache directory
  - [x] Verify Parquet/Arrow format is used

- [x] Task 4: Atomic Creation Pattern (AC: 2)
  - [x] Create temporary directory (`.vmcache/.tmp_<name>`)
  - [x] Write all data to temporary location
  - [x] Verify data integrity (check file exists, basic validation)
  - [x] Atomically rename temp directory to final name
  - [x] Clean up temp directory on any failure

- [x] Task 5: Metadata Management (AC: 1)
  - [x] Create `metadata.json` with cache info
  - [x] Include: model_name, creation_date, num_samples, embedding_dim
  - [x] Save metadata alongside dataset
  - [x] Load and validate metadata when loading cache

- [x] Task 6: Cache Loading (AC: 1)
  - [x] Implement `load()` class method to load existing cache
  - [x] Use `load_from_disk()` from HuggingFace Datasets
  - [x] Validate cache exists and is complete
  - [x] Return loaded VectorCache instance with memory-mapped data

- [x] Task 7: Error Handling (AC: 1, 2)
  - [x] Handle disk space errors
  - [x] Handle permission errors
  - [x] Handle interrupted operations (cleanup temp dirs)
  - [x] Wrap all errors in `VectorMeshError` with hints/fixes

- [x] Task 8: Testing
  - [x] Unit test: Atomic creation (simulate interruption)
  - [x] Unit test: Batch processing
  - [x] Unit test: Metadata creation and validation
  - [x] Integration test: Create and load cache with real vectorizer
  - [x] Test cleanup on failure

## Dev Notes

### Technical Requirements

**Libraries Required:**
- `datasets>=4.4.2` - Already installed, primary library for cache storage
- `pathlib` - Standard library for path operations
- `json` - Standard library for metadata

**HuggingFace Datasets Features:**
- Uses Apache Arrow format (columnar, memory-mapped)
- `save_to_disk()` creates Parquet files
- `load_from_disk()` provides zero-copy memory-mapped access
- Supports sharding for large datasets (auto-handled)

### Architecture Compliance

**File Structure:**
- Create `src/vectormesh/data/` directory if it doesn't exist
- Create `src/vectormesh/data/__init__.py`
- Implement in `src/vectormesh/data/cache.py`
- Export from `src/vectormesh/__init__.py`

**Cache Directory Structure:**
```
.vmcache/
├── my_cache/
│   ├── dataset_info.json        # HF Datasets metadata
│   ├── state.json                # HF Datasets state
│   ├── data-00000-of-00001.arrow # Arrow data files
│   └── metadata.json             # VectorMesh custom metadata
└── .tmp_my_cache/               # Temporary during creation
```

**Component Design Pattern:**
- MUST inherit from `VectorMeshComponent` (Pydantic BaseModel)
- Configuration MUST be `frozen=True` (immutable)
- Use `model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)`
- Cache data accessed through methods, not stored in Pydantic fields

**Type Safety:**
- Input: `List[str]` (texts), `TextVectorizer` (vectorizer)
- Output: `VectorCache` instance with dataset accessible via methods
- Use `ThreeDTensor` for cached embeddings shape: (batch, chunks, dim)

### Library-Specific Implementation Guidance

**HuggingFace Datasets Usage (from Context7 research):**

```python
from datasets import Dataset, load_from_disk

# Create dataset from embeddings
embeddings_dict = {
    "embeddings": embeddings_array,  # numpy array
    "text_ids": list(range(len(texts)))
}
dataset = Dataset.from_dict(embeddings_dict)

# Save to disk with automatic Parquet/Arrow format
dataset.save_to_disk("path/to/cache")

# Load with memory-mapping (zero-copy)
loaded_dataset = load_from_disk("path/to/cache")

# Access data (memory-mapped, no RAM loading)
batch = loaded_dataset[0:10]  # Efficient slice access
```

**Atomic Creation Pattern:**
```python
from pathlib import Path
import shutil

def atomic_create(target_path: Path, create_fn):
    """Atomic directory creation pattern."""
    temp_path = target_path.parent / f".tmp_{target_path.name}"

    try:
        # Clean any existing temp
        if temp_path.exists():
            shutil.rmtree(temp_path)

        # Create in temp location
        temp_path.mkdir(parents=True, exist_ok=True)
        create_fn(temp_path)  # User's creation logic

        # Atomic rename
        temp_path.rename(target_path)

    except Exception as e:
        # Cleanup on failure
        if temp_path.exists():
            shutil.rmtree(temp_path)
        raise
```

**Metadata Format:**
```json
{
  "vectormesh_version": "0.1.0",
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_dim": 384,
  "num_samples": 1000,
  "created_at": "2026-01-02T10:30:00",
  "cache_format": "huggingface_datasets_v1"
}
```

### Previous Story Intelligence

**Learnings from Story 1.2 (TextVectorizer):**

1. **Lazy Loading Pattern**: Used `object.__setattr__` to cache data in frozen Pydantic instances
   - Apply same pattern for dataset loading in VectorCache

2. **Error Wrapping**: All exceptions wrapped in `VectorMeshError` with `hint` and `fix`
   - Pattern: `raise VectorMeshError(message="...", hint="...", fix="...")`

3. **Type Imports**: Import `List` from `beartype.typing` to avoid deprecation warnings

4. **Testing Pattern**:
   - Mock external dependencies (SentenceTransformer was mocked)
   - For VectorCache: Mock file I/O or use `tmp_path` fixture

5. **Export Pattern**: Add to `src/vectormesh/__init__.py` with `__all__`

### Architecture Decisions (from architecture.md)

**Cache Protocol Decision (ADR):**
> "Decision: HuggingFace Dataset Directory (saved via save_to_disk)"
> "Rationale: Reuses robust, proven sharding/memory-mapping logic. No need to reinvent storage."
> "Schema: {"embeddings": Array3D, "masks": Array2D, "metadata": JSON}"

**Atomic Creation (from ADR-002 Reliability & Safety):**
> "Atomic Cache Creation: All long-running writes go to .vmcache.tmp. Only renamed to .vmcache after successful completion and hash verification."

**Critical Rules from project-context.md:**
- NEVER use `os.path`, always use `pathlib.Path`
- Strict type hints on all functions
- Google-style docstrings mandatory
- PascalCase for components, snake_case for methods

### Latest Technical Information (2026-01-02)

**HuggingFace Datasets Library (v4.4.2+):**

**Key Features:**
- **Arrow Format**: Columnar format optimized for analytical queries
- **Memory-Mapping**: Zero-copy reads, minimal RAM usage
- **Automatic Sharding**: Handles large datasets automatically
- **Fast Iteration**: 4.8 Gb/s throughput on 18GB dataset

**API Patterns:**
```python
# Save with sharding for large datasets
dataset.save_to_disk('cache', num_shards=128)  # Manual control

# Save with max shard size
dataset.save_to_disk('cache', max_shard_size='500MB')

# Batched iteration (memory-efficient)
for batch in dataset.iter(batch_size=1000):
    # Process batch without loading all data
    pass
```

**Performance Characteristics:**
- Memory-mapped access: O(1) random access
- Sequential iteration: ~5 Gb/s on modern SSD
- Disk space: Approximately same as NumPy array size (compressed)

### Implementation Strategy Recommendations

**Recommended Implementation Order:**

1. **Start Simple**: Create basic `VectorCache.create()` that writes to disk
2. **Add Atomic Pattern**: Implement temp directory + rename
3. **Add Metadata**: Create and validate metadata.json
4. **Add Loading**: Implement `VectorCache.load()` method
5. **Add Error Handling**: Wrap all exceptions
6. **Add Tests**: Mock I/O, test atomic creation
7. **Integration Test**: Optional test with real TextVectorizer

**Key Design Decisions:**

**Decision: Class Method vs Instance Method**
- **RECOMMENDED**: Use `@classmethod` for `create()` and `load()`
- **Rationale**: Cache doesn't exist until created, so instance creation doesn't make sense
- **Pattern**:
  ```python
  # Create new cache
  cache = VectorCache.create(texts, vectorizer, name="my_cache")

  # Load existing cache
  cache = VectorCache.load(name="my_cache")
  ```

**Decision: Dataset Storage Location**
- Store HuggingFace Dataset object as private attribute
- Use `object.__setattr__` for frozen Pydantic instance (same pattern as TextVectorizer)
- Provide accessor methods: `cache.get_embeddings(indices)`

**Potential Pitfalls to Avoid:**

1. **Don't**: Load entire dataset into RAM
   - **Why**: Defeats purpose of memory-mapping
   - **Do**: Access via dataset slicing `dataset[start:end]`

2. **Don't**: Store Dataset in Pydantic field
   - **Why**: Not serializable, causes Pydantic errors
   - **Do**: Use private attribute with `object.__setattr__`

3. **Don't**: Skip atomic creation
   - **Why**: Corrupted caches from interruptions
   - **Do**: Always use temp dir + rename pattern

4. **Don't**: Forget to clean up temp directories
   - **Why**: Disk space leaks
   - **Do**: Cleanup in `except` block and on startup

5. **Don't**: Use `os.path` for paths
   - **Why**: Project standard is pathlib
   - **Do**: Use `Path` from pathlib

### References

- [Epics: Story 1.3](../../planning-artifacts/epics.md#story-13-vectorcache-persistence--storage)
- [Architecture: Cache Protocol](../../planning-artifacts/architecture.md#data-architecture)
- [Architecture: ADR-002 Reliability](../../planning-artifacts/architecture.md#adr-002-reliability--safety-strategy)
- [Project Context](../../_bmad-output/project-context.md)
- [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets/)
- [Apache Arrow Format](https://arrow.apache.org/)

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)

### Debug Log References

No critical debugging required. Implementation followed RED-GREEN-REFACTOR cycle successfully.

### Completion Notes List

**Implementation Approach:**
- Followed TDD RED-GREEN-REFACTOR cycle as per BMad workflow
- Created 11 comprehensive tests covering all acceptance criteria
- Implemented atomic creation pattern using temp directory + rename
- Used HuggingFace Datasets for memory-mapped, zero-copy storage
- Lazy loading pattern with `object.__setattr__` for frozen Pydantic instance

**Key Implementation Details:**
1. **Atomic Creation (AC 2)**: Implemented temp directory pattern `.tmp_<name>` with automatic cleanup on failure
2. **Batch Processing (AC 1)**: Configurable batch_size (default: 32) for memory-efficient processing
3. **HF Datasets Integration (AC 1)**: Uses `Dataset.from_dict()` and `save_to_disk()` for Arrow/Parquet storage
4. **Metadata Management**: Creates `metadata.json` with model_name, embedding_dim, num_samples, created_at
5. **Memory-Mapped Loading**: `load_from_disk()` provides zero-copy access to cached embeddings
6. **Error Handling**: All errors wrapped in `VectorMeshError` with educational hints and fixes

**Testing Results:**
- All 28 tests passing (11 new + 17 existing)
- Test coverage includes: initialization, batch processing, atomic creation, metadata validation, loading, error cases
- Mock strategy: Used `Mock(spec=TextVectorizer)` with `model_name` attribute
- Linting: All checks passed after fixing unused variables

**Files Modified:**
- Created `src/vectormesh/data/__init__.py` - exports VectorCache
- Created `src/vectormesh/data/cache.py` - 286 lines, full implementation
- Modified `src/vectormesh/__init__.py` - added VectorCache export
- Created `tests/data/__init__.py` - test package init
- Created `tests/data/test_cache.py` - 193 lines, 11 comprehensive tests

**Architecture Compliance:**
- ✅ Inherits from VectorMeshComponent
- ✅ Frozen Pydantic configuration
- ✅ Google-style docstrings
- ✅ Type hints throughout
- ✅ Educational error messages
- ✅ Uses pathlib (not os.path)
- ✅ Imports List from beartype.typing

**Learnings Applied from Story 1.2:**
- Used lazy loading pattern with `object.__setattr__` for dataset caching
- Wrapped all exceptions in VectorMeshError with hint/fix fields
- Imported List from beartype.typing to avoid deprecation warnings
- Used tmp_path pytest fixture for testing file operations
- Added proper exports to __init__.py

### File List

**Created:**
- `src/vectormesh/data/__init__.py` - VectorCache export
- `src/vectormesh/data/cache.py` - VectorCache implementation (286 lines)
- `tests/data/__init__.py` - Test package init
- `tests/data/test_cache.py` - Comprehensive test suite (193 lines, 11 tests)

**Modified:**
- `src/vectormesh/__init__.py` - Added VectorCache to imports and __all__
