---
project_name: 'vectormesh'
user_name: 'raoul'
date: '2026-01-01'
sections_completed: ['technology_stack', 'language_rules', 'framework_rules', 'testing_rules', 'code_quality_rules', 'workflow_rules', 'critical_rules']
existing_patterns_found: 5
---

# Project Context for AI Agents

_This file contains critical rules and patterns that AI agents must follow when implementing code in this project. Focus on unobvious details that agents might otherwise miss._

---

## Technology Stack & Versions

- **Language Runtime**: Python 3.12+ (Managed by `uv`)
- **Core ML Framework**: PyTorch (latest stable)
- **Data & Validation**:
  - **Pydantic v2+**: Strict runtime validation for all components.
  - **Jaxtyping + Beartype**: Mandatory for tensor shape checking.
  - **HuggingFace Datasets**: Primary cache storage (Parquet/Arrow).
- **Optimization**: Hyperopt (for search spaces).
- **Dev Tooling**: `ty` (Project management), `ruff` (Linting), `mypy` (Static Analysis).

## Critical Implementation Rules

### Language-Specific Rules (Typed Python)

- **Strict Type Checking**: All function signatures MUST be fully typed. Use `Any` only as a last resort.
- **Runtime Shape Guards**: Decorate all tensor-processing methods with `@jaxtyping.jaxtyped(typechecker=beartype)`.
  - *Example*: `def forward(self, x: Float[Tensor, "batch d_model"]) -> Float[Tensor, "batch d_model"]:`
- **Error Handling**: Never raise generic `Exception`. sub-class `VectorMeshError` and include `hint` and `fix` fields for educational clarity.
- **Docstrings**: Google-style docstrings are mandatory for all public API methods.

### Framework-Specific Rules

- **Component Architecture**: Public API components MUST inherit from `VectorMeshComponent` (Pydantic). Internal utils may be raw functions.
- **Configuration**: `frozen=True` is MANDATORY. State changes require creating new objects (functional style).
- **Composition Syntax**:
  - Use `>>` ONLY for linear sequencing.
  - Use `Serial` or `Parallel` explicitly for branching/complex topologies.
- **Tensor Operations**: Prefer `einops` for reshaping. Avoid bare `view()` without shape comments.

### Testing Rules

- **Unit vs Integration**:
  - **Unit Tests**: STRICTLY NO network access. Mock all HF calls or use `bert-tiny-random`.
  - **Integration Tests**: Mark with `@pytest.mark.integration`. Network allowed but prefer `vcrpy` cassettes to ensure determinism.
- **Cache Isolation**: All tests writing to cache MUST use `tmp_path` fixture. Never touch user's `~/.vmcache`.
- **Shape Verification**: Verify tensor shapes symbolically (e.g., `(B, S, E)`), not just for running without error.
- **Mirror Structure**: Tests must mirror `src/` structure 1:1.

### Code Quality & Style Rules

- **Linting**: Conforming to `ruff` defaults is mandatory. Sort imports (`isort`) and remove unused variables (`F401`) automatically.
- **Naming Conventions**:
  - **Components**: `PascalCase` (e.g., `TextVectorizer`, `GateSkip`).
  - **Methods/Vars**: `snake_case`.
  - **Constants**: `UPPER_CASE`.
- **Docstrings**: Google-style docstrings required for all public classes and methods. Must include `Args:`, `Returns:`, and `Shapes:` (for tensors).
- **File Structure**: One Component per file is preferred over giant monoliths.

### Development Workflow Rules

- **Branch Naming**: Use strictly `feat/`, `fix/`, `docs/`, `refactor/` prefixes.
  - *Example*: `feat/add-gate-skip-component`
- **Commit Messages**: Conventional Commits style is MANDATORY.
  - *Good*: `feat: implement GateSkip component`
  - *Bad*: `added gating`
- **PR Requirements**: All PRs must pass `ruff check .` and `pytest` locally before requesting review.

### Critical Don't-Miss Rules

- **Anti-Pattern (Shapes)**: NEVER rely on implicit broadcasting. Use `einops` or explicit operations with shape comments.
- **Anti-Pattern (Paths)**: NEVER use raw strings for paths. ALWAYS use `pathlib.Path`.
- **Model Loading Strategy**: Prefer `zoo`. ALLOW arbitrary HF strings.
  - **Stage 1 (Fast)**: Fetch `AutoConfig` to validate shapes (`hidden_size`).
  - **Stage 2 (Fallback)**: Only run **Runtime Introspection** (dummy pass) if config is ambiguous.
  - **Cache**: Persist verified metadata to avoid repeated fetches.
- **Security**: ALWAYS load weights with `torch.load(..., weights_only=True)` to prevent pickle RCE.
- **Edge Case**: If a model is missing from `zoo`, raise `ModelNotFoundError`. DO NOT fallback to auto-download.
