# Story 3.1: RegexVectorizer

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a feature engineer,
I want to create binary vectors based on regex pattern matches,
so that I can extract explicit features (e.g., legal articles, specific terms) regardless of varied formulations.

## Acceptance Criteria

1.  **Given** a dictionary of patterns (e.g., for Dutch law articles)
    **When** I initialize `RegexVectorizer(patterns)`
    **Then** it validates the regexes are valid python re patterns (raises specifically `VectorMeshError` on invalid syntax)
    **And** the configuration is immutable (frozen).

2.  **Given** a list of strings
    **When** I call `vectorizer.vectorize(texts)` (or `__call__`)
    **Then** it returns a `TwoDTensor` of shape `(batch, num_patterns)` (consistent with other vectorizers where last dim is features)
    **And** values are 1.0 (match) or 0.0 (no match).

3.  **Given** specific legal text examples:
    - "artikel 265 Boek 3 van het Burgerlijk Wetboek"
    - "artikel 7:2 Burgerlijk Wetboek"
    - "artikel 7:26 lid 3 van het Burgerlijk Wetboek"
    - "artikelen 6:251 en 6:252 Burgerlijk Wetboek"
    - "artikel 55 Wet Bodembescherming"
    **And** a negative example: "artikel 6.5 en 31.5 van de koopovereenkomst" (should NOT be detected as law article)
    **When** configured with appropriate patterns
    **Then** it correctly identifies the law articles (including "6:251" and "6:252") while ignoring the "koopovereenkomst" reference.

4.  **Given** invalid regex syntax in patterns
    **When** initializing
    **Then** it raises `VectorMeshError` with a helpful `hint` (e.g., "Check regex syntax") and `fix`.

## Tasks / Subtasks

- [x] Task 1: Create `RegexVectorizer` component backbone
    - [x] Create `src/vectormesh/components/regex.py`
    - [x] Define `RegexVectorizer` inheriting from `VectorMeshComponent`
    - [x] Define `patterns` field as `Dict[str, str]`
    - [x] Implement `model_post_init` (or Pydantic validator) to compile and validate regexes immediately using `re.compile`
    - [x] Ensure validation raises `VectorMeshError` for bad patterns

- [x] Task 2: Implement Vectorization Logic
    - [x] Implement `__call__` method decorated with `@jaxtyping.jaxtyped(typechecker=beartype)`
    - [x] Accept `List[str]` and return `TwoDTensor[Batch, NumPatterns]` (Float)
    - [x] Iterate through texts and patterns to build the binary tensor
    - [x] Use `torch.tensor` for output

- [x] Task 3: Unit Testing & Validation (Strict)
    - [x] Create `tests/components/test_regex.py`
    - [x] Test valid pattern initialization
    - [x] Test invalid pattern initialization (expect `VectorMeshError` with hint)
    - [x] Test vectorization correctness with the specific Dutch law examples from the Acceptance Criteria
    - [x] Verify output shape is `(batch, num_patterns)` and type is `TwoDTensor`

- [x] Task 4: Export & Integration
    - [x] Export `RegexVectorizer` in `src/vectormesh/components/__init__.py`
    - [x] Export in `src/vectormesh/__init__.py`

## Dev Notes

- **Architecture Compliance**:
    - Inherit from `VectorMeshComponent` (Pydantic v2).
    - Use `frozen=True` in `model_config`.
    - Use `src/vectormesh/components/regex.py`.
    - **Performance**: While `re` is used now, keep the internal implementation encapsulated.
    - **Typing**: Strict `jaxtyping` + `beartype` is mandatory. Output must be `TwoDTensor` (Batch, Dim).

- **Error Handling**:
    - Catch `re.error` during validation and wrap in `VectorMeshError`.
    - Provide actionable hints.

- **Source Tree Components to Touch**:
    - `[NEW] src/vectormesh/components/regex.py`
    - `[MOD] src/vectormesh/components/__init__.py`
    - `[MOD] src/vectormesh/__init__.py`
    - `[NEW] tests/components/test_regex.py`

### Project Structure Notes

- Alignment with unified project structure: `src/vectormesh` layout.
- Naming: `RegexVectorizer` (PascalCase), `regex.py` (snake_case).

### References

- [Epics: Story 3.1](file:///Users/rgrouls/code/MADS/courses/packages/vectormesh/_bmad-output/planning-artifacts/epics.md)
- [Architecture: Components](file:///Users/rgrouls/code/MADS/courses/packages/vectormesh/_bmad-output/planning-artifacts/architecture.md)

## Dev Agent Record

### Agent Model Used

Antigravity (Gemini 2.0 Flash)

### Debug Log References

- None

### Completion Notes List

- Refined story based on user feedback.
- Updated to use specific Dutch law examples.
- Corrected output type to `TwoDTensor` for consistency.

### File List

- `src/vectormesh/components/regex.py`
- `tests/components/test_regex.py`
