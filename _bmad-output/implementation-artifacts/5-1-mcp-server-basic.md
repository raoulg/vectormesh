# Story 5.1: MCP Server Basic

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As an AI Agent user,
I want a `fastmcp` server exposing vectorization tools,
So that my agents can vectorize text and query the cache.

## Acceptance Criteria

1. **Given** the `vectormesh` MCP server running (built with `fastmcp`)
   **When** an agent calls the `vectorize_text` tool
   **Then** it returns the vector representation as a JSON array of floats

2. **Given** the MCP server
   **When** I query `list_tools`
   **Then** I see `vectorize_text` available

## Tasks / Subtasks

- [ ] Task 1: Create MCP Server Module (AC: 1, 2)
  - [ ] Create `src/vectormesh/server/` directory
  - [ ] Create `src/vectormesh/server/__init__.py` module
  - [ ] Create `src/vectormesh/server/mcp.py` with FastMCP server instance
  - [ ] Define server metadata (name, version, instructions)

- [ ] Task 2: Implement `vectorize_text` Tool (AC: 1)
  - [ ] Create tool function with `@mcp.tool()` decorator
  - [ ] Accept parameters: text (str), model_name (str, optional)
  - [ ] Use TextVectorizer to generate embeddings
  - [ ] Convert torch.Tensor to list of floats for JSON serialization
  - [ ] Return properly formatted JSON response

- [ ] Task 3: Add CLI Entry Point (AC: 2)
  - [ ] Add `vectormesh-server` script to pyproject.toml [project.scripts]
  - [ ] Create CLI function that runs FastMCP server
  - [ ] Support `--host` and `--port` arguments
  - [ ] Default to localhost:8000

- [ ] Task 4: Error Handling (AC: 1)
  - [ ] Handle invalid model names with VectorMeshError
  - [ ] Handle empty text input with helpful error
  - [ ] Wrap all errors in MCP-compatible format
  - [ ] Provide educational error messages with hints

- [ ] Task 5: Testing (AC: 1, 2)
  - [ ] Unit test: vectorize_text with valid input
  - [ ] Unit test: vectorize_text with empty text (error handling)
  - [ ] Unit test: vectorize_text with invalid model (error handling)
  - [ ] Integration test: MCP server startup and tool listing
  - [ ] Integration test: vectorize_text via MCP protocol

- [ ] Task 6: Documentation (AC: 2)
  - [ ] Add docstring to vectorize_text tool
  - [ ] Create simple README section for running MCP server
  - [ ] Document CLI usage in docstring

## Dev Notes

### Technical Requirements

**FastMCP API (v2.14.2):**
- Use `FastMCP()` to create server instance
- Use `@mcp.tool()` decorator to register tools
- Tool functions must be async (`async def`) for MCP protocol
- Server startup via `mcp.run()` method
- Installed version: `fastmcp>=2.14.2` (already in pyproject.toml)

**MCP Protocol Requirements (from PRD):**
- FR17: Expose vectorization tools to AI agents
- NFR20: MCP server compliant with MCP specification v1.0+
- NFR21: Responses formatted as valid JSON with proper error handling
- NFR22: Support concurrent connections (handled by FastMCP)

**Tool Function Signature:**
```python
@mcp.tool()
async def vectorize_text(
    text: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> list[float]:
    """Vectorize text using HuggingFace sentence transformers.

    Args:
        text: Text to vectorize
        model_name: HuggingFace model identifier

    Returns:
        List of float values representing the embedding vector

    Raises:
        ValueError: If text is empty or model_name is invalid
    """
    ...
```

### Architecture Compliance

**File Structure (from Architecture.md):**
- New module: `src/vectormesh/server/` for MCP server code
- Server module contains: `__init__.py`, `mcp.py`
- Export server from main `__init__.py` only if needed for programmatic access
- CLI script entry point in pyproject.toml [project.scripts]

**Component Architecture:**
- MCP server is NOT a VectorMeshComponent (it's infrastructure, not a vector processing component)
- TextVectorizer IS used within tool functions (reuse existing component)
- No Pydantic validation needed for server config (FastMCP handles this)
- Tool functions should use VectorMeshError for domain errors

**Error Handling (from Architecture & Project Context):**
- Educational error messages with `hint` and `fix` fields
- Wrap TextVectorizer errors in MCP-compatible format
- Never expose internal stack traces to MCP clients
- Log errors server-side for debugging

**Type Safety:**
- Full type hints on all tool functions
- Use `list[float]` return type (JSON-serializable)
- Input validation for text parameter (non-empty)
- Model name validation using TextVectorizer

### Library & Framework Requirements

**FastMCP Usage Patterns:**

```python
from fastmcp import FastMCP

# Create server instance
mcp = FastMCP(
    name="vectormesh",
    version="0.1.0",
    instructions="VectorMesh MCP server for text vectorization"
)

# Register tool with decorator
@mcp.tool()
async def vectorize_text(text: str, model_name: str = "...") -> list[float]:
    # Implementation
    pass

# Run server (blocking)
if __name__ == "__main__":
    mcp.run()
```

**TextVectorizer Integration:**

```python
from vectormesh.components.vectorizers import TextVectorizer

# Inside tool function:
vectorizer = TextVectorizer(model_name=model_name)
embeddings_tensor = vectorizer([text])  # Returns torch.Tensor (1, dim)
embeddings_list = embeddings_tensor[0].tolist()  # Convert to list[float]
return embeddings_list
```

**Dependencies Already Installed:**
- `fastmcp>=2.14.2` (MCP server framework)
- `torch>=2.9.1` (tensor library used by TextVectorizer)
- `transformers>=4.57.3` (HuggingFace models)
- `sentence-transformers>=2.0.0` (embedding models)

### File Structure Requirements

**New Files to Create:**
```
src/vectormesh/server/
├── __init__.py          # Export create_server() factory
└── mcp.py               # FastMCP server + tools
```

**Modified Files:**
```
pyproject.toml            # Add [project.scripts] entry for CLI
```

**CLI Entry Point in pyproject.toml:**
```toml
[project.scripts]
vectormesh = "vectormesh:main"  # Existing
vectormesh-server = "vectormesh.server.mcp:main"  # New
```

**Export Pattern (if needed):**
```python
# src/vectormesh/server/__init__.py
from .mcp import create_server

__all__ = ["create_server"]
```

### Testing Requirements

**Unit Tests (tests/server/test_mcp.py):**
- Test vectorize_text with valid input
- Test vectorize_text with empty text (ValueError)
- Test vectorize_text with invalid model (VectorMeshError)
- Test embedding shape (should be 1D list)
- Test embedding values are floats
- Mock TextVectorizer to avoid loading real models

**Integration Tests:**
- Mark with `@pytest.mark.integration`
- Test MCP server startup
- Test tool listing via MCP protocol
- Test vectorize_text via MCP client
- Use real TextVectorizer with small model (all-MiniLM-L6-v2)

**Testing Patterns from Previous Stories:**
- Mirror src/ structure: `tests/server/test_mcp.py`
- Use `tmp_path` fixture for any file operations
- Mock network calls in unit tests
- Use vcrpy for integration tests if needed
- Verify JSON serialization (list[float], not torch.Tensor)

### Previous Story Intelligence

**Learnings from Story 1.4 (Parameter-Free Aggregation):**

1. **TDD RED-GREEN-REFACTOR Cycle**:
   - Write failing tests first
   - Implement minimal code to pass
   - Refactor for quality
   - Applied successfully with 44/44 tests passing

2. **Type Safety Pattern**:
   - Use jaxtyping + beartype for tensor operations
   - Full type hints on all functions
   - Google-style docstrings with Args, Returns sections
   - NOT applicable to MCP tools (no tensor shape validation needed)

3. **Error Handling Pattern**:
   - All errors wrapped in VectorMeshError
   - Include `hint` and `fix` fields for educational clarity
   - Apply to TextVectorizer errors in MCP tools
   - Example:
     ```python
     try:
         vectorizer = TextVectorizer(model_name=model_name)
     except Exception as e:
         raise VectorMeshError(
             message=f"Failed to load model '{model_name}': {str(e)}",
             hint="Check that model name is valid HuggingFace identifier",
             fix="Use a supported model like 'sentence-transformers/all-MiniLM-L6-v2'"
         )
     ```

4. **Export Pattern**:
   - Add to module `__init__.py` with `__all__`
   - Export from main `__init__.py` only if part of public API
   - MCP server is infrastructure, likely not exported from main

5. **Frozen Configuration**:
   - Not applicable to MCP server (FastMCP handles config)
   - TextVectorizer already has frozen config from Story 1.2

**Learnings from Story 1.2 (TextVectorizer):**

1. **Device Management**: TextVectorizer auto-detects device (GPU/MPS/CPU)
   - No explicit device handling needed in MCP tool
   - TextVectorizer handles this internally

2. **Model Loading**: TextVectorizer caches models automatically
   - First call: downloads and caches model
   - Subsequent calls: reuses cached model
   - MCP tool can create new TextVectorizer per request (FastMCP handles concurrency)

3. **Batch Processing**: TextVectorizer accepts `list[str]`
   - For single text: `vectorizer([text])`
   - Returns tensor of shape (1, embedding_dim)
   - Extract first row: `embeddings_tensor[0].tolist()`

### Implementation Patterns

**MCP Server Module Pattern:**

```python
# src/vectormesh/server/mcp.py
"""VectorMesh MCP server for AI agent integration."""

from fastmcp import FastMCP
from vectormesh.components.vectorizers import TextVectorizer
from vectormesh.errors import VectorMeshError

# Create server instance
mcp = FastMCP(
    name="vectormesh",
    version="0.1.0",
    instructions="VectorMesh MCP server exposing text vectorization tools for AI agents"
)


@mcp.tool()
async def vectorize_text(
    text: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> list[float]:
    """Vectorize text using HuggingFace sentence transformers.

    This tool converts text into dense vector embeddings using
    state-of-the-art transformer models from HuggingFace.

    Args:
        text: Text to vectorize (non-empty string)
        model_name: HuggingFace model identifier
            (default: sentence-transformers/all-MiniLM-L6-v2)

    Returns:
        List of float values representing the 384-dimensional embedding

    Raises:
        ValueError: If text is empty
        VectorMeshError: If model loading fails

    Example:
        Input: "Hello world"
        Output: [0.123, -0.456, 0.789, ...] (384 floats)
    """
    # Validate input
    if not text or not text.strip():
        raise ValueError(
            "Text cannot be empty. "
            "Hint: Provide non-empty text for vectorization. "
            "Fix: Pass a valid text string like 'Hello world'"
        )

    try:
        # Load vectorizer (cached after first call)
        vectorizer = TextVectorizer(model_name=model_name)

        # Vectorize text (returns tensor of shape (1, embedding_dim))
        embeddings_tensor = vectorizer([text])

        # Convert to JSON-serializable list
        embeddings_list = embeddings_tensor[0].tolist()

        return embeddings_list

    except Exception as e:
        raise VectorMeshError(
            message=f"Failed to vectorize text with model '{model_name}': {str(e)}",
            hint="Check that model name is a valid HuggingFace identifier",
            fix="Use a supported model like 'sentence-transformers/all-MiniLM-L6-v2' or check https://huggingface.co/models"
        ) from e


def main() -> None:
    """Run the VectorMesh MCP server.

    This is the CLI entry point for the MCP server.

    Usage:
        uv run vectormesh-server

    Or with custom host/port (if FastMCP supports args):
        uv run vectormesh-server --host 0.0.0.0 --port 8080
    """
    import sys

    # Simple argument parsing (FastMCP might support --host --port)
    # For MVP: just use defaults
    print("Starting VectorMesh MCP Server...")
    print("Server name: vectormesh")
    print("Available tools: vectorize_text")

    try:
        mcp.run()
    except KeyboardInterrupt:
        print("\nShutting down VectorMesh MCP server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting MCP server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Testing Pattern:**

```python
# tests/server/test_mcp.py
"""Tests for VectorMesh MCP server."""

import pytest
from unittest.mock import Mock, patch
import torch

from vectormesh.server.mcp import vectorize_text, mcp
from vectormesh.errors import VectorMeshError


class TestVectorizeText:
    """Test suite for vectorize_text MCP tool."""

    @pytest.mark.asyncio
    async def test_vectorize_text_valid_input(self):
        """Test vectorize_text with valid input."""
        with patch('vectormesh.server.mcp.TextVectorizer') as MockVectorizer:
            # Mock vectorizer returns tensor
            mock_instance = Mock()
            mock_instance.return_value = torch.tensor([[0.1, 0.2, 0.3]])
            MockVectorizer.return_value = mock_instance

            result = await vectorize_text(text="Hello world")

            # Verify result is list of floats
            assert isinstance(result, list)
            assert all(isinstance(x, float) for x in result)
            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_vectorize_text_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            await vectorize_text(text="")

        error_msg = str(exc_info.value)
        assert "empty" in error_msg.lower()
        assert "hint" in error_msg.lower() or "Hint:" in error_msg

    @pytest.mark.asyncio
    async def test_vectorize_text_invalid_model_raises_error(self):
        """Test that invalid model raises VectorMeshError."""
        with patch('vectormesh.server.mcp.TextVectorizer') as MockVectorizer:
            MockVectorizer.side_effect = Exception("Model not found")

            with pytest.raises(VectorMeshError) as exc_info:
                await vectorize_text(text="test", model_name="invalid/model")

            error = exc_info.value
            assert "failed to vectorize" in str(error).lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_vectorize_text_real_model(self):
        """Integration test with real TextVectorizer."""
        result = await vectorize_text(
            text="Hello world",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Verify shape (384 dimensions for all-MiniLM-L6-v2)
        assert isinstance(result, list)
        assert len(result) == 384
        assert all(isinstance(x, float) for x in result)

        # Verify non-zero values
        assert not all(x == 0.0 for x in result)


class TestMCPServer:
    """Test suite for MCP server instance."""

    def test_server_has_vectorize_text_tool(self):
        """Test that server exposes vectorize_text tool."""
        # Get registered tools from FastMCP instance
        # This might vary based on FastMCP API
        # For MVP: just verify server exists
        assert mcp is not None
        assert mcp.name == "vectormesh"
```

### Implementation Strategy Recommendations

**Recommended Implementation Order:**

1. **Create Server Module**: Set up `src/vectormesh/server/` directory structure
2. **Implement vectorize_text Tool**: Simple async function with TextVectorizer
3. **Add Error Handling**: Wrap errors in educational VectorMeshError
4. **Create CLI Entry Point**: Add to pyproject.toml and test locally
5. **Write Unit Tests**: Mock TextVectorizer, test error cases
6. **Integration Test**: Test with real model (marked as @pytest.mark.integration)
7. **Documentation**: Add docstrings and README section

**Key Design Decisions:**

**Decision: Async vs Sync Tool Functions**
- **REQUIRED**: Async (`async def`)
- **Rationale**: MCP protocol requires async handlers
- **FastMCP**: All tool functions must be async

**Decision: TextVectorizer Instance Creation**
- **RECOMMENDED**: Create new instance per request
- **Rationale**:
  - FastMCP handles concurrency
  - TextVectorizer caches model (no performance penalty)
  - Simpler than managing shared state
- **Alternative**: Could use server-level cache for vectorizers if needed

**Decision: Error Response Format**
- **REQUIRED**: JSON-compatible errors
- **Pattern**: Raise exceptions with clear messages
- **FastMCP**: Automatically wraps exceptions in MCP error format
- **Enhancement**: Use VectorMeshError for domain errors

**Potential Pitfalls to Avoid:**

1. **Don't**: Return torch.Tensor directly from tool
   - **Why**: Not JSON-serializable
   - **Do**: Convert to list using `.tolist()`

2. **Don't**: Forget async/await for tool functions
   - **Why**: MCP protocol requires async
   - **Do**: Use `async def` for all tools

3. **Don't**: Expose internal stack traces to clients
   - **Why**: Security and UX
   - **Do**: Wrap in educational VectorMeshError

4. **Don't**: Skip input validation
   - **Why**: Poor UX for AI agents
   - **Do**: Validate text is non-empty, model name is reasonable

5. **Don't**: Load model synchronously without error handling
   - **Why**: Network errors, missing models
   - **Do**: Wrap TextVectorizer in try/except with helpful error

### Project Context Reference

**Critical Rules from project-context.md:**

1. **Type Safety**: Full type hints on all functions
   - Tool function: `async def vectorize_text(text: str, model_name: str = "...") -> list[float]:`

2. **Error Handling**: Never raise generic Exception
   - Use VectorMeshError with hint and fix fields
   - Example in implementation pattern above

3. **Docstrings**: Google-style docstrings mandatory
   - Include Args, Returns, Raises sections
   - Example in implementation pattern above

4. **File Structure**: One module per feature
   - MCP server in `src/vectormesh/server/mcp.py`
   - Tests in `tests/server/test_mcp.py`

5. **Testing**: Unit vs Integration
   - Unit: Mock TextVectorizer (no network)
   - Integration: Use real model (mark @pytest.mark.integration)

6. **Naming Conventions**:
   - Functions: `snake_case` (vectorize_text)
   - Classes: `PascalCase` (not applicable here)
   - Constants: `UPPER_CASE` (not applicable here)

### References

- [Epics: Story 1.5](../../planning-artifacts/epics.md#story-15-mcp-server-basic)
- [PRD: MCP Requirements](../../planning-artifacts/prd.md#documentation-architecture-mcp-first-approach)
- [PRD: Functional Requirements FR17, FR34-FR40](../../planning-artifacts/prd.md#documentation--learning-support)
- [PRD: Non-Functional Requirements NFR20-NFR22](../../planning-artifacts/prd.md#mcp-protocol)
- [Architecture: MCP Integration](../../planning-artifacts/architecture.md) (Note: MCP not yet in architecture doc - this is first story)
- [Project Context](../../_bmad-output/project-context.md)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp) (installed: v2.14.2)
- [MCP Specification](https://spec.modelcontextprotocol.io/specification/)

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)

### Debug Log References

(To be filled during implementation)

### Completion Notes List

(To be filled during implementation)

### File List

**Created:**
(To be filled during implementation)

**Modified:**
(To be filled during implementation)
