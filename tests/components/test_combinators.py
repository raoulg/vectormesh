"""Tests for Serial and Parallel combinators."""

import pytest
import torch
from unittest.mock import Mock, patch
from beartype.roar import BeartypeCallHintParamViolation

from vectormesh.types import TwoDTensor, ThreeDTensor, VectorMeshError
from vectormesh.components.combinators import Serial, Parallel
from vectormesh.validation import register_morphism, Morphism, TensorDimensionality


class MockTwoDVectorizer:
    """Mock 2D vectorizer for testing."""

    def __init__(self, name: str = "mock", output_dim: int = 384):
        self.name = name
        self.output_dim = output_dim

    def __call__(self, texts):
        """Return mock 2D tensor."""
        batch_size = len(texts)
        return torch.randn(batch_size, self.output_dim)


class MockThreeDVectorizer:
    """Mock 3D vectorizer for testing."""

    def __init__(self, name: str = "mock", output_dim: int = 768, chunks: int = 3):
        self.name = name
        self.output_dim = output_dim
        self.chunks = chunks

    def __call__(self, texts):
        """Return mock 3D tensor."""
        batch_size = len(texts)
        return torch.randn(batch_size, self.chunks, self.output_dim)


class MockMeanAggregator:
    """Mock aggregator that converts 3D to 2D."""

    def __init__(self, output_dim: int = 768):
        self.output_dim = output_dim

    def __call__(self, tensor_3d):
        """Aggregate 3D tensor to 2D by mean pooling."""
        if tensor_3d.dim() == 3:
            # True aggregation: [batch, chunks, dim] -> [batch, dim]
            return torch.mean(tensor_3d, dim=1)
        elif tensor_3d.dim() == 2:
            # Pass-through for 2D input (processor case)
            return tensor_3d
        else:
            # Just return appropriate shape for any other case
            batch_size = tensor_3d.shape[0] if tensor_3d.dim() > 0 else 1
            return torch.randn(batch_size, self.output_dim)


# Register morphisms for mock components
register_morphism(MockTwoDVectorizer, Morphism(
    source=TensorDimensionality.TEXT,
    target=TensorDimensionality.TWO_D,
    component_name="MockTwoDVectorizer",
    description="Mock text → 2D embedding morphism"
))

register_morphism(MockThreeDVectorizer, Morphism(
    source=TensorDimensionality.TEXT,
    target=TensorDimensionality.THREE_D,
    component_name="MockThreeDVectorizer",
    description="Mock text → 3D embedding morphism"
))

register_morphism(MockMeanAggregator, Morphism(
    source=TensorDimensionality.THREE_D,
    target=TensorDimensionality.TWO_D,
    component_name="MockMeanAggregator",
    description="Mock 3D → 2D aggregation morphism"
))

# Register combinators themselves as components
from unittest.mock import Mock
from vectormesh.components.combinators import Serial, Parallel

# Mock objects for generic testing - we'll assume they're 2D→2D processors
register_morphism(Mock, Morphism(
    source=TensorDimensionality.TWO_D,
    target=TensorDimensionality.TWO_D,
    component_name="Mock",
    description="Generic mock component (2D → 2D)"
))


class TestSerial:
    """Test Serial combinator implementation."""

    def test_serial_creation_works(self):
        """Serial should be importable and creatable."""
        from vectormesh.components.combinators import Serial
        # Should be able to create empty Serial (will fail on execution)
        serial = Serial(components=[])
        assert isinstance(serial, Serial)

    def test_serial_accepts_component_list(self):
        """Serial should accept a list of components."""
        from vectormesh.components.combinators import Serial
        vectorizer = MockThreeDVectorizer()  # 3D output
        aggregator = MockMeanAggregator()    # 3D input → 2D output

        pipeline = Serial(components=[vectorizer, aggregator])
        assert len(pipeline.components) == 2

    def test_serial_sequential_flow_2d_to_2d(self):
        """Serial should process components sequentially (2D → 2D)."""
        # Test case: TwoDVectorizer alone (Text → 2D)
        from vectormesh.components.combinators import Serial
        vectorizer = MockTwoDVectorizer("model1", 384)

        pipeline = Serial(components=[vectorizer])
        result = pipeline(["Hello world"])

        # Should be 2D output from final component
        assert result.shape == (1, 384)

    def test_serial_sequential_flow_3d_to_2d(self):
        """Serial should handle 3D → Aggregator → 2D flow."""
        from vectormesh.components.combinators import Serial
        vectorizer = MockThreeDVectorizer("model1", 768, chunks=3)
        aggregator = MockMeanAggregator(768)

        pipeline = Serial(components=[vectorizer, aggregator])
        result = pipeline(["Long document"])

        # Should be 2D output after aggregation
        assert result.shape == (1, 768)

    def test_serial_inherits_from_vectormesh_component(self):
        """Serial should inherit from VectorMeshComponent."""
        from vectormesh.components.combinators import Serial
        from vectormesh.types import VectorMeshComponent
        assert issubclass(Serial, VectorMeshComponent)

    def test_serial_type_validation_with_beartype(self):
        """Serial should provide educational errors for component failures."""
        from vectormesh.components.combinators import Serial
        pipeline = Serial(components=[MockTwoDVectorizer()])

        # This should raise VectorMeshError with educational information
        with pytest.raises(VectorMeshError) as exc_info:
            pipeline(123)  # Wrong input type

        error = exc_info.value
        assert "failed to process input" in str(error).lower()
        assert error.hint is not None
        assert error.fix is not None


class TestParallel:
    """Test Parallel combinator implementation."""

    def test_parallel_creation_works(self):
        """Parallel should be importable and creatable."""
        from vectormesh.components.combinators import Parallel
        # Should be able to create empty Parallel (will fail on execution)
        parallel = Parallel(branches=[])
        assert isinstance(parallel, Parallel)

    def test_parallel_accepts_branch_list(self):
        """Parallel should accept a list of branches."""
        from vectormesh.components.combinators import Parallel
        branch1 = MockTwoDVectorizer("model1", 384)
        branch2 = MockTwoDVectorizer("model2", 768)

        pipeline = Parallel(branches=[branch1, branch2])
        assert len(pipeline.branches) == 2

    def test_parallel_same_dimensionality_2d_plus_2d(self):
        """Parallel should handle two 2D vectorizers (AC2)."""
        from vectormesh.components.combinators import Parallel
        branch1 = MockTwoDVectorizer("model1", 384)
        branch2 = MockTwoDVectorizer("model2", 768)

        pipeline = Parallel(branches=[branch1, branch2])
        result = pipeline(["Hello world"])

        # Should return tuple of TwoDTensors
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].shape == (1, 384)
        assert result[1].shape == (1, 768)

    def test_parallel_mixed_dimensionality_2d_plus_3d_agg(self):
        """Parallel should handle mixed dimensionality with aggregation (AC2b)."""
        from vectormesh.components.combinators import Serial, Parallel

        branch1 = MockTwoDVectorizer("2d-model", 384)
        # Branch2: 3D → Aggregator serial chain
        branch2_components = [
            MockThreeDVectorizer("3d-model", 768, chunks=3),
            MockMeanAggregator(768)
        ]

        branch2 = Serial(components=branch2_components)
        pipeline = Parallel(branches=[branch1, branch2])
        result = pipeline(["Hello world"])

        # Should return tuple of two TwoDTensors
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].shape == (1, 384)  # 2D output
        assert result[1].shape == (1, 768)  # 2D output after aggregation

    def test_parallel_broadcast_to_all_branches(self):
        """Parallel should broadcast input to all branches independently."""
        from vectormesh.components.combinators import Parallel

        branch1 = MockTwoDVectorizer("model1")
        branch2 = MockTwoDVectorizer("model2")

        pipeline = Parallel(branches=[branch1, branch2])
        texts = ["Hello", "World"]
        result = pipeline(texts)

        # Each branch should receive the same input
        assert len(result) == 2
        assert result[0].shape[0] == 2  # batch size
        assert result[1].shape[0] == 2  # batch size

    def test_parallel_inherits_from_vectormesh_component(self):
        """Parallel should inherit from VectorMeshComponent."""
        from vectormesh.components.combinators import Parallel
        from vectormesh.types import VectorMeshComponent
        assert issubclass(Parallel, VectorMeshComponent)

    def test_parallel_tuple_output_format(self):
        """Parallel should return tuple format for future Concat compatibility."""
        from vectormesh.components.combinators import Parallel

        branch1 = MockTwoDVectorizer("model1", 384)
        branch2 = MockTwoDVectorizer("model2", 768)

        pipeline = Parallel(branches=[branch1, branch2])
        result = pipeline(["test"])

        # Output should be tuple[TwoDTensor, TwoDTensor]
        assert isinstance(result, tuple)
        assert all(isinstance(tensor, torch.Tensor) for tensor in result)


class TestNestedCombinators:
    """Test nested combinator scenarios (AC4, AC4b, AC4c)."""

    def test_nested_2d_only_pipeline(self):
        """Test nested 2D processing pipeline (AC4)."""
        # Serial([Parallel([TwoDVectorizer, TwoDVectorizer]), GlobalConcat])
        # Note: GlobalConcat is Story 2.3, so we'll mock it
        from vectormesh.components.combinators import Serial, Parallel

        parallel_2d = Parallel(branches=[
            MockTwoDVectorizer("model1", 384),
            MockTwoDVectorizer("model2", 768)
        ])

        # Mock GlobalConcat for testing
        mock_concat = Mock()
        mock_concat.return_value = torch.randn(1, 384 + 768)  # Combined dimensions

        pipeline = Serial(components=[parallel_2d, mock_concat])
        result = pipeline(["test"])

        # Should flow: input → Parallel → tuple → Concat → TwoDTensor
        assert result.shape == (1, 384 + 768)

    def test_nested_mixed_dimensionality(self):
        """Test mixed pipeline with 2D/3D compatibility (AC4b)."""
        from vectormesh.components.combinators import Serial, Parallel

        # Parallel with mixed branches
        parallel_mixed = Parallel(branches=[
            MockTwoDVectorizer("2d-model", 384),
            MockThreeDVectorizer("3d-model", 768)
        ])

        # Mock processor that handles (TwoDTensor, ThreeDTensor)
        mock_processor = Mock()
        mock_processor.return_value = torch.randn(1, 512)

        pipeline = Serial(components=[parallel_mixed, mock_processor])
        result = pipeline(["test"])

        assert result.shape == (1, 512)

    def test_complex_multi_level_nesting(self):
        """Test complex multi-level nesting (AC4c)."""
        from vectormesh.components.combinators import Serial, Parallel

        # Complex nested structure
        inner_serial = Serial(components=[
            MockThreeDVectorizer("raw-transformer", 768),
            MockMeanAggregator(768)
        ])

        parallel_complex = Parallel(branches=[
            MockTwoDVectorizer("sentence-transformer", 384),
            inner_serial
        ])

        mock_concat = Mock()
        mock_concat.return_value = torch.randn(1, 384 + 768)

        mock_processor = Mock()
        mock_processor.return_value = torch.randn(1, 256)

        pipeline = Serial(components=[parallel_complex, mock_concat, mock_processor])
        result = pipeline(["test"])

        assert result.shape == (1, 256)


class TestShapeValidationAndErrors:
    """Test type safety and educational error handling (AC3)."""

    def test_serial_shape_validation_with_beartype(self):
        """Serial should validate tensor shapes with beartype (AC3)."""
        from vectormesh.components.combinators import Serial

        pipeline = Serial(components=[MockTwoDVectorizer()])

        # Wrong input type should raise VectorMeshError due to component failure
        # (beartype validation happens inside component execution)
        with pytest.raises(VectorMeshError):
            pipeline(123)  # Not a list of strings

    def test_incompatible_components_raise_educational_error(self):
        """Serial with incompatible components should raise VectorMeshError with hint/fix."""
        from vectormesh.components.combinators import Serial

        # Use registered mock components that are incompatible
        vectorizer = MockTwoDVectorizer()  # TEXT → 2D
        aggregator = MockMeanAggregator()  # 3D → 2D (incompatible!)

        # This should fail at construction time due to morphism incompatibility
        with pytest.raises(ValueError) as exc_info:
            pipeline = Serial(components=[vectorizer, aggregator])

        error = str(exc_info.value)
        assert "non-composable" in error.lower()
        assert "hint:" in error.lower()
        assert "fix:" in error.lower()

    def test_parallel_input_compatibility_validation(self):
        """Parallel should validate input compatibility for broadcast."""
        from vectormesh.components.combinators import Parallel

        pipeline = Parallel(branches=[MockTwoDVectorizer(), MockTwoDVectorizer()])

        # Should handle list of strings correctly
        result = pipeline(["test1", "test2"])
        assert isinstance(result, tuple)

        # Should raise error for incompatible input
        with pytest.raises((VectorMeshError, BeartypeCallHintParamViolation)):
            pipeline(123)  # Not a list


if __name__ == "__main__":
    pytest.main([__file__])