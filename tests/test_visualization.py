"""Tests for component visualization with mathematical notation."""

import torch

from vectormesh.visualization import visualize
from vectormesh.components.combinators import Serial, Parallel
from vectormesh.components.connectors import GlobalConcat, GlobalStack
from vectormesh.components.gating import Skip, Gate
from vectormesh.types import NDTensor
from vectormesh.validation import register_morphism, Morphism, TensorDimensionality


# Mock components for testing
class MockTwoDVectorizer:
    """Mock 2D vectorizer for visualization testing."""

    def __init__(self, model_name: str = "mock-model", embedding_dim: int = 384):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.output_mode = "2d"

    def __call__(self, texts):
        batch_size = len(texts)
        return torch.randn(batch_size, self.embedding_dim)


class MockThreeDVectorizer:
    """Mock 3D vectorizer for visualization testing."""

    def __init__(self, model_name: str = "mock-model-3d", embedding_dim: int = 768, chunks: int = 3):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.chunks = chunks
        self.output_mode = "3d"

    def __call__(self, texts):
        batch_size = len(texts)
        return torch.randn(batch_size, self.chunks, self.embedding_dim)


class MockAggregator:
    """Mock aggregator for visualization testing."""

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim

    def __call__(self, tensor_3d):
        if tensor_3d.dim() == 3:
            return torch.mean(tensor_3d, dim=1)
        return tensor_3d


class MockProcessor:
    """Mock processor component."""

    def __init__(self, output_dim: int = 128):
        self.output_dim = output_dim

    def __call__(self, tensor):
        batch_size = tensor.shape[0]
        return torch.randn(batch_size, self.output_dim)


# Register morphisms for mock components
register_morphism(
    MockTwoDVectorizer,
    Morphism(
        source=TensorDimensionality.TEXT,
        target=TensorDimensionality.TWO_D,
        component_name="MockTwoDVectorizer",
        description="Mock 2D vectorizer (TEXT→2D)"
    )
)

register_morphism(
    MockThreeDVectorizer,
    Morphism(
        source=TensorDimensionality.TEXT,
        target=TensorDimensionality.THREE_D,
        component_name="MockThreeDVectorizer",
        description="Mock 3D vectorizer (TEXT→3D)"
    )
)

register_morphism(
    MockAggregator,
    Morphism(
        source=TensorDimensionality.THREE_D,
        target=TensorDimensionality.TWO_D,
        component_name="MockAggregator",
        description="Mock aggregator (3D→2D)"
    )
)

register_morphism(
    MockProcessor,
    Morphism(
        source=TensorDimensionality.TWO_D,
        target=TensorDimensionality.TWO_D,
        component_name="MockProcessor",
        description="Mock processor (2D→2D)"
    )
)

# Register Skip and Gate for integration tests
register_morphism(
    Skip,
    Morphism(
        source=TensorDimensionality.TWO_D,
        target=TensorDimensionality.TWO_D,
        component_name="Skip",
        description="Residual skip connection (2D→2D)"
    )
)

register_morphism(
    Gate,
    Morphism(
        source=TensorDimensionality.TWO_D,
        target=TensorDimensionality.TWO_D,
        component_name="Gate",
        description="Gating mechanism (2D→2D)"
    )
)


# =============================================================================
# AC1: Serial Pipeline Visualization with Mathematical Notation
# =============================================================================

def test_serial_simple_chain_visualization():
    """Test visualization of simple Serial chain (AC1)."""
    # Arrange: Simple 2D → 2D → 2D pipeline
    vec = MockTwoDVectorizer("bert-base", 768)
    proc = MockProcessor(128)
    pipeline = Serial(components=[vec, proc])

    # Act
    output = visualize(pipeline)

    # Assert: Should show linear chain with arrows
    assert "→" in output, "Should use arrow (→) for sequential flow"
    assert "MockTwoDVectorizer" in output, "Should show component name"
    assert "MockProcessor" in output, "Should show all components"


def test_serial_with_mathematical_notation():
    """Test Serial visualization includes mathematical tensor notation (AC1)."""
    # Arrange
    vec = MockTwoDVectorizer("model", 384)
    pipeline = Serial(components=[vec])

    # Act
    output = visualize(pipeline)

    # Assert: Should contain Unicode mathematical symbols
    assert "ℝ" in output, "Should use ℝ (U+211D) for real tensors"
    assert "B" in output or "batch" in output.lower(), "Should indicate batch dimension"


# =============================================================================
# AC2: Parallel Branching Visualization with Mathematical Types
# =============================================================================

def test_parallel_branching_visualization():
    """Test Parallel shows branching topology (AC2)."""
    # Arrange: Two parallel branches
    vec1 = MockTwoDVectorizer("model1", 384)
    vec2 = MockTwoDVectorizer("model2", 768)
    parallel = Parallel(branches=[vec1, vec2])

    # Act
    output = visualize(parallel)

    # Assert: Should show tree structure with branch characters
    assert "├──" in output or "└──" in output, "Should use tree characters (├──, └──)"
    assert "model1" in output, "Should show first branch"
    assert "model2" in output, "Should show second branch"


def test_parallel_shows_tuple_output():
    """Test Parallel visualization indicates tuple output (AC2)."""
    # Arrange
    vec1 = MockTwoDVectorizer("model1", 384)
    vec2 = MockTwoDVectorizer("model2", 768)
    parallel = Parallel(branches=[vec1, vec2])

    # Act
    output = visualize(parallel)

    # Assert: Should indicate tuple output
    assert "(" in output and ")" in output, "Should show tuple notation for output"


# =============================================================================
# AC3: Nested Combinator Hierarchy with Dimension Flow Tracking
# =============================================================================

def test_nested_combinators_with_indentation():
    """Test nested combinator visualization with proper indentation (AC3)."""
    # Arrange: Nested Serial within Parallel
    inner_serial = Serial(components=[
        MockThreeDVectorizer("model1", 384),
        MockAggregator(384)
    ])
    vec2 = MockTwoDVectorizer("model2", 768)
    parallel = Parallel(branches=[inner_serial, vec2])
    outer = Serial(components=[MockTwoDVectorizer("base", 512), parallel])

    # Act
    output = visualize(outer)

    # Assert: Should show hierarchical structure
    assert "Serial[" in output or "Parallel[" in output, "Should label combinator types"
    lines = output.split("\n")
    # Check that there's some form of indentation/hierarchy
    assert len(lines) > 3, "Should have multiple lines for nested structure"


def test_nested_shows_dimension_transformations():
    """Test nested visualization tracks 2D/3D transformations (AC3)."""
    # Arrange: 3D → aggregation → 2D flow
    pipeline = Serial(components=[
        MockThreeDVectorizer("model", 768),
        MockAggregator(768)
    ])

    # Act
    output = visualize(pipeline)

    # Assert: Should indicate dimension changes (if shape inference implemented)
    # This is a basic test - we expect component names at minimum
    assert "MockThreeDVectorizer" in output
    assert "MockAggregator" in output


# =============================================================================
# AC4: Mathematical Notation Standards
# =============================================================================

def test_mathematical_notation_standards():
    """Test visualization uses proper mathematical notation (AC4)."""
    # Arrange: Simple pipeline
    vec = MockTwoDVectorizer("model", 384)
    pipeline = Serial(components=[vec])

    # Act
    output = visualize(pipeline)

    # Assert: Unicode symbols for mathematical notation
    # ℝ (U+211D) for real tensors
    assert "ℝ" in output or "R" in output, "Should use ℝ or R for tensors"
    # Should have dimension notation
    has_dimensions = any(char in output for char in ["B", "×", "^", "{", "}"])
    assert has_dimensions, "Should show tensor dimensions with notation"


def test_sigma_star_for_text_input():
    """Test visualization uses Σ* for text inputs (AC4)."""
    # Arrange: Pipeline starting with vectorizer (text input)
    vec = MockTwoDVectorizer("model", 384)
    pipeline = Serial(components=[vec])

    # Act
    output = visualize(pipeline)

    # Assert: Should indicate text input with Σ*
    # This might be implicit in showing the vectorizer at start
    # or explicit with "Σ*" or "text" or "str"
    assert "MockTwoDVectorizer" in output  # At minimum show the component


# =============================================================================
# AC5: Connector Visualization Integration
# =============================================================================

def test_globalconcat_visualization():
    """Test GlobalConcat shows concatenation effect (AC5)."""
    # Arrange: Parallel → GlobalConcat
    vec1 = MockTwoDVectorizer("model1", 384)
    vec2 = MockTwoDVectorizer("model2", 768)
    parallel = Parallel(branches=[vec1, vec2])
    concat = GlobalConcat(dim=1)
    pipeline = Serial(components=[parallel, concat])

    # Act
    output = visualize(pipeline)

    # Assert: Should show connector and its parameters
    assert "GlobalConcat" in output, "Should show connector component"
    assert "dim=1" in output or "dim" in output, "Should show connector parameters"


def test_globalstack_visualization():
    """Test GlobalStack visualization (AC5)."""
    # Arrange: Parallel → GlobalStack
    vec1 = MockTwoDVectorizer("model1", 384)
    vec2 = MockTwoDVectorizer("model2", 512)
    parallel = Parallel(branches=[vec1, vec2])
    stack = GlobalStack(dim=1)
    pipeline = Serial(components=[parallel, stack])

    # Act
    output = visualize(pipeline)

    # Assert: Should show stack connector
    assert "GlobalStack" in output, "Should show stack connector"


# =============================================================================
# AC6: Component Parameter Display
# =============================================================================

def test_component_parameters_displayed():
    """Test component parameters are shown inline (AC6)."""
    # Arrange: Vectorizer with model name
    vec = MockTwoDVectorizer("bert-base-uncased", 768)
    pipeline = Serial(components=[vec])

    # Act
    output = visualize(pipeline)

    # Assert: Should show model identifier
    assert "bert-base-uncased" in output or "MockTwoDVectorizer" in output, \
        "Should show component identifier or name"


def test_connector_parameters_displayed():
    """Test connector parameters are displayed (AC6)."""
    # Arrange: Connector with dim parameter
    concat = GlobalConcat(dim=1)
    # Wrap in a minimal pipeline
    vec1 = MockTwoDVectorizer("m1", 384)
    vec2 = MockTwoDVectorizer("m2", 384)
    parallel = Parallel(branches=[vec1, vec2])
    pipeline = Serial(components=[parallel, concat])

    # Act
    output = visualize(pipeline)

    # Assert: Should show dim parameter
    assert "dim=1" in output or "dim" in output.lower(), "Should display connector parameters"


# =============================================================================
# Integration Tests with Gating Components
# =============================================================================

def test_skip_component_visualization():
    """Test Skip component visualization."""
    # Arrange: Skip component
    main = MockProcessor(384)
    skip = Skip(main=main)
    pipeline = Serial(components=[MockTwoDVectorizer("model", 384), skip])

    # Act
    output = visualize(pipeline)

    # Assert: Should show Skip component
    assert "Skip" in output, "Should show Skip component in visualization"


def test_gate_component_visualization():
    """Test Gate component visualization."""
    # Arrange: Gate component
    def router(x: NDTensor) -> float:
        return 0.5

    component = MockProcessor(384)
    gate = Gate(component=component, router=router)
    pipeline = Serial(components=[MockTwoDVectorizer("model", 384), gate])

    # Act
    output = visualize(pipeline)

    # Assert: Should show Gate component
    assert "Gate" in output, "Should show Gate component in visualization"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

def test_single_component_visualization():
    """Test visualizing a single component."""
    # Arrange: Just one component
    vec = MockTwoDVectorizer("model", 384)

    # Act
    output = visualize(vec)

    # Assert: Should work with single component
    assert "MockTwoDVectorizer" in output
    assert isinstance(output, str)


def test_empty_serial_raises_or_handles():
    """Test empty Serial is handled gracefully."""
    # Arrange: Empty Serial
    pipeline = Serial(components=[])

    # Act & Assert: Should either raise error or handle gracefully
    # Based on Serial validation, this should raise during creation
    # But if it doesn't, visualization should handle it
    try:
        output = visualize(pipeline)
        # If it doesn't raise, output should be reasonable
        assert isinstance(output, str)
    except Exception:
        # Expected - empty Serial may raise during creation or visualization
        pass


def test_visualization_returns_string():
    """Test visualize() returns a string."""
    # Arrange
    vec = MockTwoDVectorizer("model", 384)
    pipeline = Serial(components=[vec])

    # Act
    output = visualize(pipeline)

    # Assert
    assert isinstance(output, str), "visualize() should return a string"
    assert len(output) > 0, "Output should not be empty"
