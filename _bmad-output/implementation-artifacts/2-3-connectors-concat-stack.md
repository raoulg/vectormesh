# Story 2.3: Connectors (Concat/Stack)

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a model architect,
I want typed `concat` and `stack` connectors,
so that I can merge parallel branches into a single tensor for downstream processing.

## Acceptance Criteria

**AC1: GlobalConcat Implementation (Same Dimensionality)**
**Given** a `Parallel` output tuple from Story 2.1 `(TwoDTensor, TwoDTensor)`
**When** I apply `GlobalConcat(dim=1)` (feature/embedding dimension)
**Then** it returns a single `TwoDTensor` with concatenated features `[batch, combined_dim]`
**And** it validates that batch dimensions match
**And** raises educational error if `dim` parameter is missing

**AC2: GlobalConcat Error Handling (Mixed Dimensionality)**
**Given** a mixed `Parallel` output `(TwoDTensor, ThreeDTensor)` from Story 2.1
**When** I apply `GlobalConcat(dim=1)`
**Then** it raises an educational `VectorMeshError` explaining 2D/3D incompatibility
**And** provides two solution options:
  - "Option 1: Use MeanAggregator() on 3D branch to normalize dimensions for concatenation"
  - "Option 2: Use GlobalStack(dim=1) instead, which handles mixed dimensionality by creating chunk dimension"

**AC2b: GlobalConcat for 3D Tensors (Chunk Concatenation)**
**Given** a `Parallel` output tuple `(ThreeDTensor[B,C1,D], ThreeDTensor[B,C2,D])`
**When** I apply `GlobalConcat(dim=1)` (concatenating along chunk dimension)
**Then** it returns a single `ThreeDTensor` with shape `[batch, C1+C2, D]`
**And** it validates that batch dimensions match
**And** it validates that embedding dimensions match (D must be same)

**AC3: Pipeline Integration (Serial + Parallel + Concat)**
**Given** a `Serial` pipeline ending in `Parallel` from Story 2.1
**When** I append `>> GlobalConcat(dim=1)`
**Then** the final output is a single tensor with proper dimension handling
**And** type checking validates the 2D/3D compatibility at each step

**AC4: Normalized Branch Concatenation**
**Given** a normalized `Parallel` output `(TwoDTensor, TwoDTensor)` from `[TwoDVectorizer, Serial([ThreeDVectorizer, MeanAggregator])]`
**When** I apply `GlobalConcat(dim=1)`
**Then** it concatenates successfully because aggregation normalized dimensions
**And** the output dimension equals sum of input feature dimensions

**AC5: GlobalStack for Adding New Dimension (2D + 2D)**
**Given** a `Parallel` output tuple `(TwoDTensor, TwoDTensor)` with shapes `[(batch, emb1), (batch, emb2)]`
**When** I apply `GlobalStack(dim=1)` instead of GlobalConcat
**Then** it returns a `ThreeDTensor` with shape `[batch, 2, max(emb1, emb2)]`
**And** it pads smaller embeddings if emb1 != emb2
**And** it validates that batch dimensions match
**And** provides educational error if dimensions are incompatible

**AC6: GlobalStack for Extending Chunk Dimension (3D + 2D)**
**Given** a mixed `Parallel` output `(ThreeDTensor[batch, chunks, emb], TwoDTensor[batch, emb])`
**When** I apply `GlobalStack(dim=1)` to stack along chunk dimension
**Then** it returns a `ThreeDTensor` with shape `[batch, chunks+1, emb]`
**And** it properly handles the 2D tensor as a single chunk (unsqueezes to [batch, 1, emb])
**And** it validates embedding dimensions match
**And** provides educational error if embedding dimensions don't match

**AC6b: GlobalStack for 3D + 3D (Creates 4D Multi-Branch Representation)**
**Given** a `Parallel` output tuple `(ThreeDTensor[B,C1,D], ThreeDTensor[B,C2,D])`
**When** I apply `GlobalStack(dim=1)`
**Then** it returns a `FourDTensor` with shape `[batch, 2, max(C1,C2), D]`
**And** it pads shorter chunk sequences to max(C1, C2)
**And** it validates embedding dimensions match
**And** provides educational note: "Creates 4D multi-branch tensor suitable for CNN-like architectures"

**AC7: Definition-Time Validation for Full Pipeline (GlobalConcat)**
**Given** a `Serial` pipeline: `[Parallel([TwoDVectorizer, TwoDVectorizer]), GlobalConcat(dim=-1), SomeNeuralNet()]`
**When** I initialize the pipeline
**Then** it validates at definition-time:
- Parallel outputs `Tuple[TwoDTensor, TwoDTensor]`
- GlobalConcat can concatenate this tuple (same dimensionality)
- GlobalConcat outputs `TwoDTensor`
- SomeNeuralNet can accept `TwoDTensor` input
**And** raises educational error at initialization if any step is incompatible

**AC8: Definition-Time Validation Error (Mixed Dimensionality)**
**Given** a `Serial` pipeline: `[Parallel([TwoDVectorizer, ThreeDVectorizer]), GlobalConcat(dim=-1)]`
**When** I initialize the pipeline
**Then** it raises `VectorMeshError` at definition-time (not call-time)
**And** error message explains: "GlobalConcat cannot merge mixed 2D/3D tensors from Parallel branches"
**And** provides hint: "Use aggregation to normalize dimensions before concatenation"
**And** provides fix: "Add MeanAggregator() to ThreeDVectorizer branch"

**AC9: Definition-Time Validation for GlobalStack Output Compatibility**
**Given** a `Serial` pipeline: `[Parallel([TwoDVectorizer, TwoDVectorizer]), GlobalStack(dim=1), Component2D()]`
**When** I initialize the pipeline
**Then** it raises `VectorMeshError` at definition-time
**And** error explains: "GlobalStack outputs ThreeDTensor but Component2D expects TwoDTensor"
**And** provides hint: "Add aggregation after GlobalStack to convert 3D→2D"
**And** provides fix: "Insert MeanAggregator() between GlobalStack and Component2D"

## Tasks / Subtasks

- [ ] Task 1: Implement GlobalConcat connector (AC: 1, 2, 3, 4)
  - [ ] Create `GlobalConcat` class inheriting from VectorMeshComponent
  - [ ] Implement `__call__()` method accepting tuple inputs from Parallel
  - [ ] Add dimension validation for 2D/3D compatibility
  - [ ] Implement educational error for mixed dimensionality
  - [ ] Add batch dimension matching validation
  - [ ] Implement concatenation along specified dimension (default dim=-1)

- [ ] Task 2: Implement GlobalStack connector (AC: 5, 6)
  - [ ] Create `GlobalStack` class inheriting from VectorMeshComponent
  - [ ] Implement `__call__()` method accepting tuple inputs from Parallel
  - [ ] Add logic to stack along new dimension (2D+2D → 3D)
  - [ ] Add logic to extend existing dimension (3D+2D → 3D with chunks+1)
  - [ ] Implement padding for mismatched embedding dimensions
  - [ ] Add educational errors for incompatible shapes

- [ ] Task 3: Definition-Time Validation (AC: 7, 8, 9)
  - [ ] Implement output type inference for GlobalConcat (same dimensionality as inputs)
  - [ ] Implement output type inference for GlobalStack (always ThreeDTensor)
  - [ ] Add validation in Serial's component chain validator to check connector compatibility
  - [ ] Validate that previous component output (Parallel tuple) matches connector input expectations
  - [ ] Validate that connector output type matches next component input expectations
  - [ ] Implement educational errors for all validation failures at definition-time

- [ ] Task 4: Integration with Serial/Parallel combinators (AC: 3, 7, 8, 9)
  - [ ] Ensure GlobalConcat/GlobalStack work with `>>` operator
  - [ ] Test nested combinator support (Serial → Parallel → GlobalConcat)
  - [ ] Validate full pipeline type checking at definition-time
  - [ ] Test error messages for incompatible pipeline configurations

- [ ] Task 5: Testing and validation with comprehensive 2D/3D coverage
  - [ ] Unit tests for GlobalConcat with 2D+2D inputs (feature concatenation)
  - [ ] Unit tests for GlobalConcat with 3D+3D inputs (chunk concatenation)
  - [ ] Unit tests for GlobalConcat error handling with 2D+3D inputs (mixed dimensionality)
  - [ ] Unit tests for GlobalStack with 2D+2D → 3D transformation
  - [ ] Unit tests for GlobalStack with 3D+2D chunk extension
  - [ ] Unit tests for GlobalStack with 3D+3D chunk concatenation
  - [ ] Integration tests with Serial→Parallel→GlobalConcat pipelines (2D and 3D cases)
  - [ ] Integration tests with Serial→Parallel→GlobalStack pipelines
  - [ ] Test definition-time validation errors for incompatible pipelines
  - [ ] Test educational error messages for all failure modes
  - [ ] Test padding behavior for mismatched embedding dimensions
  - [ ] Validate type checking with mypy/pyright for all flows

## Dev Notes

### Critical Implementation Requirements

**🔥 ULTIMATE CONNECTOR IMPLEMENTATION - This enables advanced tensor composition for the entire framework!**

**Architecture Patterns and Constraints (from project-context.md):**
- All public API components MUST inherit from `VectorMeshComponent` (Pydantic)
- `frozen=True` is MANDATORY - state changes require creating new objects (functional style)
- Use `@jaxtyped(typechecker=beartype)` for all tensor-processing methods
- Google-style docstrings required for all public classes and methods
- **CRITICAL**: Use specific tensor types from `types.py` (`TwoDTensor`, `ThreeDTensor`, `OneDTensor`) instead of generic `Tensor`

**Composition Integration (from architecture.md ADR-001):**
- GlobalConcat and GlobalStack must integrate seamlessly with `Serial` and `Parallel` from Story 2.1
- Must support `>>` operator integration: `Parallel([...]) >> GlobalConcat(dim=-1)`
- No implicit operations - all dimension transformations must be explicit

**Type Safety Requirements:**
- GlobalConcat input: `Tuple[NDTensor, ...]` from Parallel output
- GlobalConcat output: Single `NDTensor` (2D or 3D depending on inputs)
- GlobalStack input: `Tuple[NDTensor, ...]` from Parallel output
- GlobalStack output: `ThreeDTensor` (always creates or extends 3D tensor)
- Runtime shape validation with beartype
- Educational error messages with `hint` and `fix` fields

**Error Handling Pattern:**
- Never raise generic `Exception` - subclass `VectorMeshError`
- Include `hint` field explaining the concept
- Include `fix` field suggesting likely code changes
- Example for mixed dimensionality: "Shape mismatch: Cannot concat 2D and 3D tensors. Hint: Use aggregation to normalize dimensions. Fix: Add .aggregate('mean') to 3D branch before concatenation."

**Key Innovation from User Input:**
- **GlobalStack** implements a fundamentally different merging strategy than GlobalConcat
- Instead of concatenating along embedding dimension: (batch, emb1) + (batch, emb2) → (batch, emb1+emb2)
- GlobalStack creates/extends chunk dimension: (batch, emb) + (batch, emb) → (batch, 2, emb)
- Or extends existing chunks: (batch, chunks, emb) + (batch, emb) → (batch, chunks+1, emb)
- This enables treating multiple embedding sources as a sequence rather than a combined feature vector

**2D/3D Combination Matrix for GlobalConcat (Concatenate Along Existing Dimension):**

| Input 1        | Input 2        | GlobalConcat Output | Notes                           |
|----------------|----------------|---------------------|--------------------------------|
| TwoDTensor     | TwoDTensor     | TwoDTensor[B,D1+D2] | ✅ Concat along dim=-1 (feature dimension) |
| TwoDTensor     | ThreeDTensor   | ❌ ERROR            | Mixed dimensionality - suggest aggregation OR GlobalStack |
| ThreeDTensor   | ThreeDTensor   | ✅ ThreeDTensor[B,C1+C2,D] | Concat along dim=1 (chunk dimension), embedding dims must match |

**2D/3D/4D Combination Matrix for GlobalStack (Create New Dimension / Extend Chunks):**

| Input 1        | Input 2        | GlobalStack Output  | Notes                           |
|----------------|----------------|---------------------|--------------------------------|
| TwoDTensor     | TwoDTensor     | ✅ ThreeDTensor[B,2,max(D1,D2)] | Create chunk dim, pad embeddings if D1≠D2 |
| ThreeDTensor   | TwoDTensor     | ✅ ThreeDTensor[B,C+1,D] | Extend chunks: unsqueeze 2D[B,D]→[B,1,D] |
| TwoDTensor     | ThreeDTensor   | ✅ ThreeDTensor[B,C+1,D] | Same as above (order doesn't matter) |
| ThreeDTensor   | ThreeDTensor   | ✅ FourDTensor[B,2,max(C1,C2),D] | Create branch dim, pad chunks if C1≠C2 (CNN-ready) |

**Critical Shape Flow Patterns:**
- **GlobalConcat (Concatenate Along Existing Dimension)**:
  - Uses `torch.cat()` - concatenates along EXISTING dimension
  - **2D+2D**: Concat along feature dim (dim=-1): `[B,D1] + [B,D2] → [B,D1+D2]`
  - **3D+3D**: Concat along chunk dim (dim=1): `[B,C1,D] + [B,C2,D] → [B,C1+C2,D]` (embedding dims must match)
  - **2D+3D**: ❌ ERROR - suggest aggregation OR GlobalStack

- **GlobalStack (Create New Dimension / Extend Chunks)**:
  - Uses `torch.stack()` concept - creates NEW dimension or extends existing chunk dimension
  - **2D+2D**: Create chunk dim: `[B,D1] + [B,D2] → [B,2,max(D1,D2)]` (pads if D1≠D2)
  - **3D+2D or 2D+3D**: Extend chunks: `[B,C,D] + [B,D] → [B,C+1,D]` (unsqueeze 2D first)
  - **3D+3D**: Create branch dim: `[B,C1,D] + [B,C2,D] → [B,2,max(C1,C2),D]` (4D multi-branch, pads if C1≠C2)

- **Key Semantic Difference**:
  - **GlobalConcat**: Merge along existing dimension (combine features or combine chunks)
  - **GlobalStack**: Create/extend chunk dimension (treat inputs as separate chunks)

- **Validation**: Both must validate batch dimensions match across all inputs

**Definition-Time Validation Strategy (CRITICAL!):**
- **Type Inference**: GlobalConcat/GlobalStack must declare output types based on inputs
  - GlobalConcat with `Tuple[TwoDTensor, TwoDTensor]` → outputs `TwoDTensor`
  - GlobalConcat with `Tuple[ThreeDTensor, ThreeDTensor]` → outputs `ThreeDTensor`
  - GlobalStack with `Tuple[TwoDTensor, ...]` → outputs `ThreeDTensor` (creates chunk dim)
  - GlobalStack with `Tuple[ThreeDTensor, TwoDTensor]` → outputs `ThreeDTensor` (extends chunks)
  - GlobalStack with `Tuple[ThreeDTensor, ThreeDTensor]` → outputs `FourDTensor` (creates branch dim)
- **Serial Validation**: When GlobalConcat/GlobalStack is in a Serial chain, validate:
  1. Previous component output type matches connector input expectations
  2. If previous is Parallel, infer tuple types from branch outputs
  3. Connector can handle that specific tuple (dimensionality compatibility)
  4. Connector output type matches next component input expectations
- **Fail Fast**: All validation happens at pipeline initialization, not at call-time
- **Educational Errors**: Provide specific fix suggestions for incompatible configurations

**Example Validation Flow:**
```python
# At initialization, Serial validates the full chain:
pipeline = Serial([
    Parallel([TwoDVectorizer("m1"), ThreeDVectorizer("m2")]),  # → Tuple[2D, 3D]
    GlobalConcat(dim=-1),  # ❌ FAILS HERE: Cannot concat mixed 2D/3D
    SomeNN()
])
# Error: "GlobalConcat cannot merge mixed 2D/3D tensors from Parallel branches.
#  Hint: Normalize dimensions before concatenation.
#  Fix: Parallel([TwoDVectorizer('m1'), Serial([ThreeDVectorizer('m2'), MeanAggregator()])])"

# Valid pipeline:
pipeline = Serial([
    Parallel([TwoDVectorizer("m1"), TwoDVectorizer("m2")]),  # → Tuple[2D, 2D] ✓
    GlobalConcat(dim=-1),  # → 2D ✓
    SomeNN()  # Expects 2D ✓
])
```

**Validation Implementation Requirements:**
- GlobalConcat/GlobalStack need `infer_output_type(input_type)` method
- Serial's `validate_component_chain()` must handle connectors specially
- Check if component is GlobalConcat/GlobalStack and previous is Parallel
- Infer Parallel's tuple output from its branches
- Validate connector compatibility with that tuple
- Propagate connector's output type to next component validation

### Technical Requirements from Epic Analysis

**From Epic 2 Business Context:**
- Enable "merging multiple vector signals" for sophisticated processing
- Support both feature concatenation (GlobalConcat) and sequence stacking (GlobalStack)
- Foundation for downstream processing after parallel branches

**Integration with Previous Stories:**
- **Story 1.1**: Base types (`TwoDTensor`, `ThreeDTensor`) are already defined
- **Story 2.1**: `Parallel` combinator outputs tuple format that GlobalConcat/GlobalStack consume
- **Story 2.1**: VectorMeshComponent pattern and educational errors established

**Integration with Future Stories:**
- **Story 2.2**: `>>` operator will work with GlobalConcat/GlobalStack (e.g., `parallel >> GlobalConcat(dim=-1)`)
- **Story 2.4**: Visualization will show concat/stack operations in pipeline diagrams
- **Story 2.5**: Gating mechanisms may use connectors for multi-path merging

**Git Intelligence from Recent Commits:**
- Recent focus on "2d/3d vector consistency" (commits bc7e694, 0961180)
- "review 2-1 combinators" (commit 3ca8e38) - review process for combinator work
- Pattern: Comprehensive testing for all components, especially around 2D/3D handling

### Source Tree Components to Touch

**New Files to Create:**
- `src/vectormesh/components/connectors.py` - GlobalConcat and GlobalStack implementations

**Existing Files to Modify:**
- `src/vectormesh/__init__.py` - Add GlobalConcat, GlobalStack to public API exports
- `src/vectormesh/components/__init__.py` - Export connectors from components module

**Testing Files to Create:**
- `tests/components/test_connectors.py` - Comprehensive unit and integration tests

**Testing Standards:**
- Unit tests STRICTLY NO network access - mock all HF calls
- Test tensor shapes symbolically (e.g., `(B, S, E)`) not just success
- Mirror test structure to `src/` structure 1:1
- Integration tests marked with `@pytest.mark.integration`
- Test educational error messages for all failure modes

### Project Structure Notes

**Alignment with Unified Project Structure:**
- Follows `src/vectormesh/components/` layout as specified in architecture.md
- Connectors directory for merging/combining operations
- Consistent with existing VectorMeshComponent pattern from Stories 1.1 and 2.1
- Matches established educational error patterns

**Detected Conflicts or Variances:**
- None - this story naturally extends the combinator foundation from Story 2.1
- Compatible with existing Parallel tuple output format
- Sets pattern for future connector components (if needed)

**File Location Rationale:**
- `connectors.py` contains GlobalConcat and GlobalStack as they work together as merging primitives
- Separate from `combinators.py` to distinguish composition (Serial/Parallel) from merging (Concat/Stack)
- Follows architecture decision for related components in same file

### Library and Framework Requirements

**Core Dependencies (from pyproject.toml):**
- **PyTorch**: Latest stable - tensor concatenation, stacking, and padding operations
- **Pydantic v2+**: Component configuration with `frozen=True` requirement
- **jaxtyping**: Tensor shape annotations (`Tuple[TwoDTensor, ...]`, `ThreeDTensor` patterns)
- **beartype**: Runtime type checking for tensor operations
- **einops**: Tensor reshaping operations (preferred over bare `.view()`)

**Development Tools:**
- **ruff**: Linting and code formatting (zero violations required)
- **mypy/pyright**: Static type checking (strict mode required)
- **pytest**: Testing framework with integration test markers

**Architecture Compliance Requirements:**
1. **VectorMeshComponent Pattern**: All connectors inherit from this base
2. **Frozen Configuration**: No mutable state, functional composition style
3. **Educational Errors**: VectorMeshError with hint/fix fields for all failures
4. **Type Safety**: Full jaxtyping + beartype validation on all tensor operations
5. **Google-style Docstrings**: Required for all public methods with `Shapes:` sections

**Critical Implementation Notes:**
- Use `torch.cat()` for GlobalConcat along specified dimension
- Use `torch.stack()` for GlobalStack creating new dimension
- Use `torch.nn.functional.pad()` for padding mismatched embeddings in GlobalStack
- Use `einops.rearrange()` for dimension transformations if needed
- Always validate batch dimensions match before operations
- Always validate embedding dimensions match for GlobalStack

### File Structure Requirements

**src/vectormesh/components/connectors.py:**
```python
from typing import Union, Tuple
from beartype.typing import List
from jaxtyping import jaxtyped
from beartype import beartype as typechecker
from pydantic import Field
import torch

from vectormesh.types import VectorMeshComponent, TwoDTensor, ThreeDTensor, FourDTensor, NDTensor, VectorMeshError

class GlobalConcat(VectorMeshComponent):
    """Concatenate parallel branch outputs along specified dimension.

    Merges multiple tensor outputs from Parallel combinator into a single
    tensor by concatenating along the specified dimension (typically feature
    dimension). Requires all inputs to have same dimensionality (all 2D or
    all 3D after aggregation).

    Args:
        dim: Dimension along which to concatenate (default: -1 for features)

    Example:
        ```python
        parallel = Parallel([TwoDVectorizer("model1"), TwoDVectorizer("model2")])
        concat = GlobalConcat(dim=-1)
        pipeline = parallel >> concat
        result = pipeline(["Hello"])  # TwoDTensor[batch, dim1+dim2]
        ```

    Shapes:
        Input: Tuple[TwoDTensor[B,D1], TwoDTensor[B,D2], ...] (same dimensionality)
        Output: TwoDTensor[B, D1+D2+...] (concatenated features)
    """

    dim: int = Field(default=-1, description="Dimension along which to concatenate")

    @classmethod
    def infer_output_type(cls, input_tuple_types: Tuple[type, ...]) -> type:
        """Infer output tensor type from input tuple types.

        For definition-time validation in Serial chains. Returns the tensor
        type that will be output given the input tuple types.

        Args:
            input_tuple_types: Tuple of tensor types (e.g., (TwoDTensor, TwoDTensor))

        Returns:
            Output tensor type (same dimensionality as inputs)

        Raises:
            VectorMeshError: If mixed dimensionality (2D+3D) detected
        """
        # Check all same dimensionality, return that type
        # Raise educational error if mixed
        pass

    @jaxtyped(typechecker=typechecker)
    def __call__(self, inputs: Tuple[NDTensor, ...]) -> NDTensor:
        """Concatenate tensors along specified dimension.

        Validates dimensionality compatibility and batch dimension matching,
        then concatenates all tensors along the specified dimension.

        Args:
            inputs: Tuple of tensors from Parallel combinator output

        Returns:
            Single concatenated tensor

        Raises:
            VectorMeshError: If dimensionality mismatch or batch dimensions don't match

        Shapes:
            Input: Tuple of same-dimensionality tensors
            Output: Single tensor with concatenated dimension
        """
        # Implementation with educational errors for mixed dimensionality
        pass


class GlobalStack(VectorMeshComponent):
    """Stack parallel branch outputs along new or existing dimension.

    Merges multiple tensor outputs by stacking along a dimension, either
    creating a new chunk dimension (2D→3D) or extending an existing chunk
    dimension (3D+2D→3D). Handles padding for mismatched embedding dimensions.

    Args:
        dim: Dimension along which to stack (default: 1 for chunks)

    Example:
        ```python
        # Create new chunk dimension
        parallel = Parallel([TwoDVectorizer("model1"), TwoDVectorizer("model2")])
        stack = GlobalStack(dim=1)
        result = stack(parallel(["Hello"]))  # ThreeDTensor[batch, 2, emb]

        # Extend chunk dimension
        parallel_mixed = Parallel([ThreeDVectorizer("model"), TwoDVectorizer("model2")])
        stack = GlobalStack(dim=1)
        result = stack(parallel_mixed(["Hello"]))  # ThreeDTensor[batch, chunks+1, emb]
        ```

    Shapes:
        Input 2D+2D: Tuple[TwoDTensor[B,E1], TwoDTensor[B,E2]] → ThreeDTensor[B,2,max(E1,E2)]
        Input 3D+2D: Tuple[ThreeDTensor[B,C,E], TwoDTensor[B,E]] → ThreeDTensor[B,C+1,E]
    """

    dim: int = Field(default=1, description="Dimension along which to stack")

    @classmethod
    def infer_output_type(cls, input_tuple_types: Tuple[type, ...]) -> type:
        """Infer output tensor type from input tuple types.

        For definition-time validation in Serial chains. GlobalStack returns
        ThreeDTensor for 2D inputs or mixed 2D/3D, FourDTensor for 3D+3D.

        Args:
            input_tuple_types: Tuple of tensor types (e.g., (TwoDTensor, TwoDTensor))

        Returns:
            ThreeDTensor for 2D+2D or 3D+2D cases (creates/extends chunks)
            FourDTensor for 3D+3D case (creates branch dimension)

        Raises:
            VectorMeshError: If embedding dimensions incompatible
        """
        # Check if all inputs are 3D
        all_3d = all(t == ThreeDTensor for t in input_tuple_types)
        if all_3d:
            return FourDTensor  # 3D+3D creates 4D multi-branch
        else:
            return ThreeDTensor  # 2D+2D or mixed 2D/3D creates/extends 3D

    @jaxtyped(typechecker=typechecker)
    def __call__(self, inputs: Tuple[NDTensor, ...]) -> Union[ThreeDTensor, FourDTensor]:
        """Stack tensors along specified dimension, creating or extending chunk dimension.

        Handles padding for mismatched dimensions and proper dimension
        management for all cases: 2D→3D, 3D extension, and 3D→4D multi-branch.

        Args:
            inputs: Tuple of tensors from Parallel combinator output

        Returns:
            ThreeDTensor for 2D+2D or 3D+2D cases
            FourDTensor for 3D+3D case (multi-branch representation)

        Raises:
            VectorMeshError: If shapes incompatible or dimensions don't align

        Shapes:
            2D+2D: [B,E1], [B,E2] → ThreeDTensor[B,2,max(E1,E2)] (pad embeddings)
            3D+2D: [B,C,E], [B,E] → ThreeDTensor[B,C+1,E] (unsqueeze 2D first)
            3D+3D: [B,C1,E], [B,C2,E] → FourDTensor[B,2,max(C1,C2),E] (pad chunks, create branch dim)
        """
        # Implementation with padding and dimension handling for all cases
        pass
```

**Implementation Strategy:**
- **GlobalConcat**:
  - `infer_output_type()`: Check all inputs same dimensionality, return that type or raise error
  - `__call__()`: Validate same dimensionality, then `torch.cat(inputs, dim=self.dim)`
  - Validate batch dimensions match across all inputs
  - Educational errors for mixed dimensionality
- **GlobalStack**:
  - `infer_output_type()`: Returns `ThreeDTensor` for 2D/mixed inputs, `FourDTensor` for 3D+3D
  - `__call__()`: Route based on dimensionality combinations:
    - **2D+2D**: Pad to max(D1,D2), unsqueeze to [B,1,D], stack → [B,2,max(D1,D2)]
    - **3D+2D**: Unsqueeze 2D to [B,1,E], concatenate along chunk dim → [B,C+1,E]
    - **3D+3D**: Pad to max(C1,C2), stack along new branch dim → [B,2,max(C1,C2),E] (4D!)
  - Validate batch and embedding dimensions match, educational errors for incompatibilities
- **Serial Integration**:
  - Modify `Serial.validate_component_chain()` to handle connectors
  - When connector detected, check if previous component is Parallel
  - Infer Parallel's tuple output types from branch types
  - Call connector's `infer_output_type()` to validate compatibility
  - Use connector's output type to validate next component in chain
  - Fail fast with educational errors at definition-time

### References

**Architecture Documents:**
- [ADR-001: Composition Syntax](../../planning-artifacts/architecture.md#adr-001-composition-syntax) - Serial/Parallel integration requirements
- [Component Pattern](../../planning-artifacts/architecture.md#component-pattern-the-unit-of-work) - VectorMeshComponent inheritance requirement
- [Error Handling Patterns](../../planning-artifacts/architecture.md#error-handling-patterns) - Educational error requirements

**Epic Requirements:**
- [Epic 2: Advanced Composition & Architecture](../../planning-artifacts/epics.md#epic-2-advanced-composition--architecture) - FR8 (Connectors) coverage
- [Story 2.3: Connectors](../../planning-artifacts/epics.md#story-23-connectors-concatstack) - User story and acceptance criteria source

**Project Context:**
- [Component Architecture Rules](../../project-context.md#framework-specific-rules) - VectorMeshComponent, frozen=True requirements
- [Type Safety Rules](../../project-context.md#language-specific-rules) - jaxtyping + beartype requirements
- [Testing Rules](../../project-context.md#testing-rules) - Unit vs integration test requirements

**Previous Stories:**
- [Story 2.1: Combinators](./2-1-combinators-serial-parallel.md) - Parallel tuple output format and integration patterns
- [Story 1.1: Core Types](./1-1-core-types-component-base.md) - Foundation types (TwoDTensor, ThreeDTensor)

**User Innovation:**
- User suggested GlobalStack as alternative to GlobalConcat for different composition semantics
- Stacking along new dimension treats embeddings as sequence rather than combined features
- Enables (batch, emb) + (batch, emb) → (batch, 2, emb) instead of (batch, emb+emb)

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.5

### Debug Log References

N/A - Story creation phase

### Completion Notes List

**🎉 STORY 2.3 CONTEXT CREATION COMPLETED - ULTIMATE CONNECTOR IMPLEMENTATION GUIDE READY**

**✅ All Story Requirements Documented:**
- **AC1-4**: GlobalConcat for merging parallel branches with educational errors
- **AC5-6**: GlobalStack for sequence-based composition (user innovation integrated)
- **AC7-9**: Definition-time validation for full pipeline type safety (CRITICAL!)
- **Integration**: Comprehensive Serial/Parallel integration patterns from Story 2.1
- **Type Safety**: Full jaxtyping + beartype validation requirements specified
- **Educational Errors**: Complete error handling patterns with hint/fix fields

**✅ Critical Developer Guardrails Established:**
1. **VectorMeshComponent Pattern**: Frozen Pydantic configuration, no mutable state
2. **Definition-Time Validation**: `infer_output_type()` method for type inference in Serial chains
3. **2D/3D Combination Matrices**: Explicit behavior for all input combinations
4. **User Innovation**: GlobalStack implements alternative composition semantics
5. **Type Safety**: Use TwoDTensor/ThreeDTensor, never generic Tensor
6. **Fail Fast**: All validation at pipeline initialization, not call-time
7. **Testing Strategy**: Comprehensive coverage of all 2D/3D flows and validation errors

**✅ Architecture Compliance:**
- **ADR-001 Integration**: Works with Serial/Parallel and >> operator
- **Component Pattern**: VectorMeshComponent inheritance mandatory
- **Error Handling**: Educational VectorMeshError with hint/fix fields
- **Code Quality**: Google-style docstrings, ruff compliance, strict type checking

**✅ Previous Story Intelligence Applied:**
- **Story 2.1**: Parallel tuple output format is the foundation for connectors
- **Story 2.1**: VectorMeshComponent and educational error patterns established
- **Story 1.1**: TwoDTensor/ThreeDTensor types provide type safety backbone
- **Recent Git Patterns**: Focus on 2D/3D consistency and comprehensive testing

**✅ User Innovation Integrated:**
- **GlobalStack Concept**: Alternative to concatenation along embedding dimension
- **Sequence Semantics**: (batch, emb) + (batch, emb) → (batch, 2, emb)
- **Chunk Extension**: (batch, chunks, emb) + (batch, emb) → (batch, chunks+1, emb)
- **Use Case**: Treating multiple embeddings as sequence rather than combined features

**✅ Technical Requirements Specified:**
- **GlobalConcat**: Validates same dimensionality, concatenates along dim=-1
- **GlobalStack**: Handles 2D+2D (create chunks), 3D+2D (extend chunks), 3D+3D (concat chunks)
- **Type Inference**: `infer_output_type()` classmethod for definition-time validation
- **Serial Integration**: Modified component chain validation to handle connectors
- **Padding Strategy**: GlobalStack pads mismatched embedding dimensions
- **Batch Validation**: All operations validate batch dimensions match
- **Fail Fast**: Educational errors at definition-time for all incompatibility cases

**✅ Integration Points Mapped:**
- Extends Story 2.1 Parallel combinator with tuple consumption
- Foundation for Story 2.2 >> operator integration
- Will be visualized in Story 2.4 pipeline diagrams
- May be used by Story 2.5 gating mechanisms for multi-path merging

**✅ File Structure Defined:**
- `src/vectormesh/components/connectors.py` - GlobalConcat and GlobalStack
- `tests/components/test_connectors.py` - Comprehensive unit and integration tests
- Exports in `__init__.py` files for public API

**✅ Quality Gates:**
- pyright strict mode: zero errors required
- ruff: zero violations required
- Test coverage ≥90% for new modules
- All acceptance criteria must be validated

**Ultimate Context Engine Analysis Completed - Comprehensive Developer Guide Created**

### File List

**Files to Create:**
- `/Users/rgrouls/code/MADS/courses/packages/vectormesh/src/vectormesh/components/connectors.py`
- `/Users/rgrouls/code/MADS/courses/packages/vectormesh/tests/components/test_connectors.py`

**Files to Modify:**
- `/Users/rgrouls/code/MADS/courses/packages/vectormesh/src/vectormesh/__init__.py` - Add GlobalConcat, GlobalStack exports
- `/Users/rgrouls/code/MADS/courses/packages/vectormesh/src/vectormesh/components/__init__.py` - Export connectors