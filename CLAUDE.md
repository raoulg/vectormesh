# Claude Code Instructions for VectorMesh

Always use Context7 MCP when I need library/API documentation, code generation, setup or configuration steps without me having to explicitly ask.

## Package Management

**ALWAYS use `uv` for package management, never `pip`.**

- Install dependencies: `uv sync`
- Add packages: `uv add <package>`
- Remove packages: `uv remove <package>`
- Run scripts: `uv run <command>`

## Code Quality Standards

### Design Principles
- **SRP** (Single Responsibility Principle)
- **Open-Closed Principle**
- Keep modules focused and composable

### Type Safety
- **always** use typehint, and run linters
- **use Pydantic** for type validation and input/output contracts
- Define clear schemas for model inputs/outputs
- **TENSOR TYPES**: ALWAYS use the specific tensor types from `types.py` (TwoDTensor, ThreeDTensor, OneDTensor) instead of generic `Tensor` or jaxtyping `Float[Tensor, "..."]`. This ensures consistency across the codebase and better linting support.

### Path Handling
- **Always use `pathlib`** - never use `os` for path operations

### Code Quality
- Enforce linting from day one
- Follow consistent code style

## Architecture Focus

VectorMesh is an SDK for vector-space experimentation with HuggingFace models:
- Clear separation: vectorization → connectors → custom architectures
- Pydantic-enforced model input/output expectations
- Use Hugging Face MCP for model discovery and validation
- Focus on composability and extensibility
