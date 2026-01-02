# Story 1.1: Core Types & Component Base (Foundation)

Status: done

## Story

As a developer,
I want a strict type system and component base class,
So that I catch shape errors early and enforce configuration validation.

## Acceptance Criteria

1. **Given** a class inheriting from `VectorMeshComponent`
   **When** it is instantiated with invalid configuration types
   **Then** Pydantic v2 raises a validation error immediately
   **And** the configuration is immutable (frozen)

2. **Given** a method decorated with `@jaxtyping.jaxtyped`
   **When** I pass a tensor of the wrong shape (e.g., 2D instead of 1D)
   **Then** `beartype` raises a detailed `TypeCheckError` describing the shape mismatch

3. **Given** a usage of `OneDTensor`, `TwoDTensor`, or `ThreeDTensor`
   **When** I inspect the type hints
   **Then** they resolve to the correct `Float[Tensor, ...]` jaxtyping definition

## Tasks / Subtasks

- [x] Task 1: Create `VectorMeshComponent` base class (AC: 1)
  - [x] Implement `VectorMeshComponent` inheriting from `pydantic.BaseModel`
  - [x] Set `model_config` to `frozen=True` and `arbitrary_types_allowed=True`
  - [x] Verify validation ensures correct types on instantiation

- [x] Task 2: Define Tensor Types (AC: 3)
  - [x] Create `src/vectormesh/types.py`
  - [x] Define `OneDTensor` as `Float[Tensor, "dim"]`
  - [x] Define `TwoDTensor` as `Float[Tensor, "batch dim"]`
  - [x] Define `ThreeDTensor` as `Float[Tensor, "batch seq dim"]`
  - [x] Add explicit docstrings explaining usage

- [x] Task 3: Implement Runtime Shape Checking (AC: 2)
  - [x] Ensure `jaxtyping` and `beartype` decorators work seamlessly
  - [x] Add a utility or documentation for standard decoration pattern `@jaxtyping.jaxtyped(typechecker=beartype)`

- [x] Task 4: Create Custom Error System
  - [x] Implement `VectorMeshError` base class
  - [x] Ensure it supports friendly error messages (NFR3)

## Dev Notes

### Technical Requirements
- **Library**: `pydantic>=2.0`
- **Library**: `jaxtyping`, `beartype`
- **Library**: `torch` (for Tensor type)
- **Base Class**: `VectorMeshComponent` must be the root for all future components.

### Architecture Compliance
- **File Structure**:
  - `src/vectormesh/base.py`: `VectorMeshComponent`
  - `src/vectormesh/types.py`: Tensor definitions
  - `src/vectormesh/errors.py`: Error classes
- **Type Safety**: STRICT definition time and runtime time checking.

### References
- [Epics: Story 1.1](_bmad-output/planning-artifacts/epics.md#story-11-core-types--component-base-foundation)
- [Architecture: Typed Tensors](_bmad-output/architecture.md#decision-2-strict-typing-with-jaxtyping--beartype)
- [Project Context](_bmad-output/project-context.md)

## Dev Agent Record

### Agent Model Used
- Spec-Writer: Google Gemini (Planning)
- Implementer: Google Gemini (Dev)

### Completion Notes List
- Implemented `VectorMeshComponent` with strict Pydantic v2 validation.
- Defined `OneDTensor`, `TwoDTensor`, `ThreeDTensor` in `vectormesh.types`.
- Created `vectormesh.utils.check_shapes` decorator for runtime validation.
- Created `vectormesh.errors.VectorMeshError` with `hint` and `fix` fields.
- Verified all components with 4 unit tests (100% pass).

### File List
- `src/vectormesh/base.py`
- `src/vectormesh/types.py`
- `src/vectormesh/utils.py`
- `src/vectormesh/errors.py`
- `tests/test_base.py`
- `tests/test_types.py`
- `tests/test_runtime_check.py`
- `tests/test_errors.py`
### Senior Developer Review (AI)
- **Outcome**: Fixes Applied
- **Date**: 2026-01-01
- **Severity**: 1 High, 1 Medium, 1 Low
- **Resolution**:
  - [x] High: Exported public API in `src/vectormesh/__init__.py`.
  - [x] Medium: Integrated `VectorMeshError` in `utils.py` for friendlier errors.
  - [x] Low: Hardened `tests/test_types.py` against string representation changes.

### Change Log
- 2026-01-01: Auto-fixed code review issues (Exports, Utils, Tests).
