"""Tests for gating mechanisms: Skip and Gate components."""

import pytest
import torch

from vectormesh.types import VectorMeshError, NDTensor
from vectormesh.components.gating import Skip, Gate
from vectormesh.validation import register_morphism, Morphism, TensorDimensionality


class MockComponent:
    """Mock component for testing gating mechanisms."""

    def __init__(self, input_dim: int = 384, output_dim: int = 384):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __call__(self, input_data: NDTensor) -> NDTensor:
        """Return tensor with output_dim."""
        batch_size = input_data.shape[0]
        return torch.randn(batch_size, self.output_dim)


class MockProjection:
    """Mock projection component for dimension matching."""

    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __call__(self, input_data: NDTensor) -> NDTensor:
        """Project input to output dimension."""
        batch_size = input_data.shape[0]
        return torch.randn(batch_size, self.output_dim)


# Register morphisms for Skip, Gate, and MockComponent
# Skip and Gate preserve tensor dimensionality (2D→2D, 3D→3D)
# They can be used as identity morphisms in the composition system
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

# Register MockComponent for integration tests
register_morphism(
    MockComponent,
    Morphism(
        source=TensorDimensionality.TWO_D,
        target=TensorDimensionality.TWO_D,
        component_name="MockComponent",
        description="Mock component for testing (2D→2D)"
    )
)


# =============================================================================
# Skip Component Tests
# =============================================================================

def test_skip_matching_shapes():
    """Test Skip with input and main output matching shapes (AC1)."""
    # Arrange: Create Skip with component that preserves dimensions
    main = MockComponent(input_dim=384, output_dim=384)
    skip = Skip(main=main)

    # Act: Process input
    input_data = torch.randn(2, 384)  # Batch of 2, dim 384
    output = skip(input_data)

    # Assert: Output shape matches input, LayerNorm applied
    assert output.shape == (2, 384), f"Expected (2, 384), got {output.shape}"
    assert output is not input_data, "Output should be new tensor (not in-place)"


def test_skip_with_projection():
    """Test Skip with projection for dimension mismatch (AC2)."""
    # Arrange: Main changes dimensions 768 → 512, projection handles it
    main = MockComponent(input_dim=768, output_dim=512)
    projection = MockProjection(input_dim=768, output_dim=512)
    skip = Skip(main=main, projection=projection)

    # Act: Process input with dimension 768
    input_data = torch.randn(2, 768)
    output = skip(input_data)

    # Assert: Output has dimension 512 (from main path)
    assert output.shape == (2, 512), f"Expected (2, 512), got {output.shape}"


def test_skip_shape_mismatch_error_without_projection():
    """Test Skip raises VectorMeshError on shape mismatch without projection (AC2)."""
    # Arrange: Main changes dimensions without projection
    main = MockComponent(input_dim=768, output_dim=512)
    skip = Skip(main=main)  # No projection provided

    # Act & Assert: Should raise educational error
    input_data = torch.randn(2, 768)
    with pytest.raises(VectorMeshError) as exc_info:
        skip(input_data)

    # Verify error has educational fields
    error = exc_info.value
    assert "shape mismatch" in str(error).lower()
    assert error.hint is not None, "Error should have hint field"
    assert error.fix is not None, "Error should have fix field"
    assert "projection" in error.fix.lower(), "Fix should mention projection"


def test_skip_normalization_applied():
    """Test Skip applies LayerNorm to result (AC1)."""
    # Arrange
    main = MockComponent(input_dim=384, output_dim=384)
    skip = Skip(main=main)
    input_data = torch.randn(2, 384)

    # Act
    output = skip(input_data)

    # Assert: Output should be normalized (check mean/std approximately)
    # LayerNorm normalizes to mean=0, std=1 along last dimension
    mean = output.mean(dim=1)

    # Mean should be close to 0 (with some tolerance)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5), \
        "LayerNorm should normalize mean to ~0"


def test_skip_with_projection_shape_still_mismatch():
    """Test Skip raises error if projection output still doesn't match main output."""
    # Arrange: Projection that doesn't actually fix the mismatch
    main = MockComponent(input_dim=768, output_dim=512)
    bad_projection = MockProjection(input_dim=768, output_dim=256)  # Wrong output!
    skip = Skip(main=main, projection=bad_projection)

    # Act & Assert
    input_data = torch.randn(2, 768)
    with pytest.raises(VectorMeshError) as exc_info:
        skip(input_data)

    error = exc_info.value
    assert "shape mismatch" in str(error).lower()


# =============================================================================
# Gate Component Tests
# =============================================================================

def test_gate_with_scalar_router():
    """Test Gate with router returning scalar value (AC3)."""
    # Arrange: Router that returns scalar (global gating)
    def scalar_router(input_data: NDTensor) -> float:
        return 0.5  # Gate with 50% signal

    component = MockComponent(input_dim=384, output_dim=384)
    gate = Gate(component=component, router=scalar_router)

    # Act
    input_data = torch.randn(2, 384)
    output = gate(input_data)

    # Assert: Output is gated (component_output * 0.5)
    assert output.shape == (2, 384), f"Expected (2, 384), got {output.shape}"
    # Since router returns 0.5, output magnitude should be ~half of ungated
    # (This is a weak test, but validates basic gating behavior)


def test_gate_with_tensor_router():
    """Test Gate with router returning per-element gates (AC3)."""
    # Arrange: Router that returns tensor (per-element gating)
    def tensor_router(input_data: NDTensor) -> NDTensor:
        # Return same-shaped tensor of gate values
        return torch.ones_like(input_data) * 0.8

    component = MockComponent(input_dim=384, output_dim=384)
    gate = Gate(component=component, router=tensor_router)

    # Act
    input_data = torch.randn(2, 384)
    output = gate(input_data)

    # Assert
    assert output.shape == (2, 384), f"Expected (2, 384), got {output.shape}"


def test_gate_router_required():
    """Test Gate requires router parameter (no default pass-through) (AC3)."""
    # Arrange & Act & Assert: Creating Gate without router should fail
    component = MockComponent()

    with pytest.raises(Exception):  # Pydantic validation error
        Gate(component=component)  # Missing required 'router' field


def test_gate_shape_mismatch_error():
    """Test Gate raises error when router returns incompatible tensor shape."""
    # Arrange: Router that returns wrong-shaped tensor
    def bad_router(input_data: NDTensor) -> NDTensor:
        # Return wrong shape (should match output shape)
        return torch.randn(2, 128)  # Wrong dim!

    component = MockComponent(input_dim=384, output_dim=384)
    gate = Gate(component=component, router=bad_router)

    # Act & Assert
    input_data = torch.randn(2, 384)
    with pytest.raises(VectorMeshError) as exc_info:
        gate(input_data)

    error = exc_info.value
    assert "shape" in str(error).lower() or "broadcast" in str(error).lower()
    assert error.hint is not None
    assert error.fix is not None


def test_gate_allows_scalar_tensor_broadcast():
    """Test Gate allows scalar-like tensors to broadcast."""
    # Arrange: Router that returns scalar tensor (should broadcast)
    def scalar_tensor_router(input_data: NDTensor) -> NDTensor:
        return torch.tensor(0.5)  # Scalar tensor

    component = MockComponent(input_dim=384, output_dim=384)
    gate = Gate(component=component, router=scalar_tensor_router)

    # Act
    input_data = torch.randn(2, 384)
    output = gate(input_data)

    # Assert: Should work (scalar broadcasts)
    assert output.shape == (2, 384)


# =============================================================================
# Integration Tests with Combinators
# =============================================================================

def test_skip_in_serial_pipeline():
    """Test Skip integrates with Serial combinator (AC4)."""
    from vectormesh.components.combinators import Serial

    # Arrange: Serial pipeline with Skip
    component1 = MockComponent(input_dim=384, output_dim=384)
    component2 = MockComponent(input_dim=384, output_dim=384)
    skip = Skip(main=component1)

    pipeline = Serial(components=[skip, component2])

    # Act
    input_data = torch.randn(2, 384)
    output = pipeline(input_data)

    # Assert: Pipeline executes successfully
    assert output.shape == (2, 384)


def test_gate_in_serial_pipeline():
    """Test Gate integrates with Serial combinator (AC4)."""
    from vectormesh.components.combinators import Serial

    # Arrange
    def simple_router(x: NDTensor) -> float:
        return 0.7

    component1 = MockComponent(input_dim=384, output_dim=384)
    component2 = MockComponent(input_dim=384, output_dim=384)
    gate = Gate(component=component1, router=simple_router)

    pipeline = Serial(components=[gate, component2])

    # Act
    input_data = torch.randn(2, 384)
    output = pipeline(input_data)

    # Assert
    assert output.shape == (2, 384)


def test_skip_and_gate_in_parallel_branches():
    """Test Skip and Gate work in Parallel branches (AC4)."""
    from vectormesh.components.combinators import Parallel

    # Arrange: Use same output dimensions to avoid jaxtyping dimension binding issues
    def router_fn(x: NDTensor) -> float:
        return 0.6

    skip = Skip(main=MockComponent(input_dim=384, output_dim=384))
    gate = Gate(component=MockComponent(input_dim=384, output_dim=384), router=router_fn)

    parallel = Parallel(branches=[skip, gate])

    # Act
    input_data = torch.randn(2, 384)
    output = parallel(input_data)

    # Assert: Parallel returns tuple
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert output[0].shape == (2, 384)  # Skip branch
    assert output[1].shape == (2, 384)  # Gate branch


# =============================================================================
# Pydantic Validation Tests
# =============================================================================

def test_skip_frozen_pydantic_model():
    """Test Skip follows frozen=True Pydantic pattern (AC4)."""
    # Arrange
    main = MockComponent()
    skip = Skip(main=main)

    # Act & Assert: Attempting to modify should fail
    with pytest.raises(Exception):  # Pydantic ValidationError
        skip.main = MockComponent()


def test_gate_frozen_pydantic_model():
    """Test Gate follows frozen=True Pydantic pattern (AC4)."""
    # Arrange
    def router(x): return 0.5
    component = MockComponent()
    gate = Gate(component=component, router=router)

    # Act & Assert
    with pytest.raises(Exception):
        gate.component = MockComponent()
