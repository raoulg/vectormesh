"""Tests for GlobalConcat and GlobalStack connectors."""

import pytest
import torch
from vectormesh.components.connectors import GlobalConcat, GlobalStack
from vectormesh.types import TwoDTensor, ThreeDTensor, FourDTensor, VectorMeshError


class TestGlobalConcat:
    """Test suite for GlobalConcat connector."""

    def test_concat_2d_same_dimensionality(self):
        """Test GlobalConcat with 2D+2D inputs (AC1)."""
        # Arrange
        batch_size = 4
        dim1, dim2 = 768, 512
        tensor1: TwoDTensor = torch.randn(batch_size, dim1)
        tensor2: TwoDTensor = torch.randn(batch_size, dim2)
        concat = GlobalConcat(dim=1)

        # Act
        result = concat((tensor1, tensor2))

        # Assert
        assert result.shape == (batch_size, dim1 + dim2)
        assert isinstance(result, torch.Tensor)

    def test_concat_missing_dim_parameter(self):
        """Test GlobalConcat raises error when dim parameter is missing (AC1)."""
        # This test verifies that dim is a required parameter
        # For now, we have a default, but story says it should be required
        # We'll implement this requirement
        pass  # Will implement after basic functionality

    def test_concat_mixed_dimensionality_error(self):
        """Test GlobalConcat error for 2D+3D mixed inputs (AC2)."""
        # Arrange
        batch_size = 4
        tensor_2d: TwoDTensor = torch.randn(batch_size, 768)
        tensor_3d: ThreeDTensor = torch.randn(batch_size, 10, 768)
        concat = GlobalConcat(dim=1)

        # Act & Assert
        with pytest.raises(VectorMeshError) as exc_info:
            concat((tensor_2d, tensor_3d))

        # Verify error message contains educational guidance
        error = exc_info.value
        assert "mixed" in str(error).lower() or "2d" in str(error).lower()
        assert error.hint is not None
        assert error.fix is not None
        assert "MeanAggregator" in error.fix or "GlobalStack" in error.fix

    def test_concat_3d_same_dimensionality(self):
        """Test GlobalConcat with 3D+3D inputs along chunk dimension (AC2b)."""
        # Arrange
        batch_size = 4
        chunks1, chunks2 = 10, 15
        embed_dim = 768
        tensor1: ThreeDTensor = torch.randn(batch_size, chunks1, embed_dim)
        tensor2: ThreeDTensor = torch.randn(batch_size, chunks2, embed_dim)
        concat = GlobalConcat(dim=1)  # Concatenate along chunk dimension

        # Act
        result = concat((tensor1, tensor2))

        # Assert
        assert result.shape == (batch_size, chunks1 + chunks2, embed_dim)
        assert isinstance(result, torch.Tensor)

    def test_concat_batch_dimension_mismatch_error(self):
        """Test GlobalConcat raises error when batch dimensions don't match."""
        # Arrange
        tensor1: TwoDTensor = torch.randn(4, 768)
        tensor2: TwoDTensor = torch.randn(8, 512)  # Different batch size
        concat = GlobalConcat(dim=1)

        # Act & Assert
        with pytest.raises(VectorMeshError) as exc_info:
            concat((tensor1, tensor2))

        assert "batch" in str(exc_info.value).lower()


class TestGlobalStack:
    """Test suite for GlobalStack connector."""

    def test_stack_2d_creates_3d(self):
        """Test GlobalStack with 2D+2D creates 3D tensor (AC5)."""
        # Arrange
        batch_size = 4
        embed1, embed2 = 768, 512
        tensor1: TwoDTensor = torch.randn(batch_size, embed1)
        tensor2: TwoDTensor = torch.randn(batch_size, embed2)
        stack = GlobalStack(dim=1)

        # Act
        result = stack((tensor1, tensor2))

        # Assert
        max_embed = max(embed1, embed2)
        assert result.shape == (batch_size, 2, max_embed)
        assert isinstance(result, torch.Tensor)

    def test_stack_3d_2d_extends_chunks(self):
        """Test GlobalStack with 3D+2D extends chunk dimension (AC6)."""
        # Arrange
        batch_size = 4
        chunks = 10
        embed_dim = 768
        tensor_3d: ThreeDTensor = torch.randn(batch_size, chunks, embed_dim)
        tensor_2d: TwoDTensor = torch.randn(batch_size, embed_dim)
        stack = GlobalStack(dim=1)

        # Act
        result = stack((tensor_3d, tensor_2d))

        # Assert
        assert result.shape == (batch_size, chunks + 1, embed_dim)
        assert isinstance(result, torch.Tensor)

    def test_stack_3d_3d_creates_4d(self):
        """Test GlobalStack with 3D+3D creates 4D multi-branch tensor (AC6b)."""
        # Arrange
        batch_size = 4
        chunks1, chunks2 = 10, 15
        embed_dim = 768
        tensor1: ThreeDTensor = torch.randn(batch_size, chunks1, embed_dim)
        tensor2: ThreeDTensor = torch.randn(batch_size, chunks2, embed_dim)
        stack = GlobalStack(dim=1)

        # Act
        result = stack((tensor1, tensor2))

        # Assert
        max_chunks = max(chunks1, chunks2)
        assert result.shape == (batch_size, 2, max_chunks, embed_dim)
        assert isinstance(result, torch.Tensor)

    def test_stack_embedding_dimension_mismatch_error(self):
        """Test GlobalStack raises error when embedding dimensions don't match (AC6)."""
        # Arrange
        batch_size = 4
        tensor_3d: ThreeDTensor = torch.randn(batch_size, 10, 768)
        tensor_2d: TwoDTensor = torch.randn(batch_size, 512)  # Different embed dim
        stack = GlobalStack(dim=1)

        # Act & Assert
        with pytest.raises(VectorMeshError) as exc_info:
            stack((tensor_3d, tensor_2d))

        assert "embedding" in str(exc_info.value).lower() or "dimension" in str(exc_info.value).lower()


class TestGlobalConcatDefinitionTimeValidation:
    """Test suite for GlobalConcat definition-time validation (AC7-8)."""

    def test_infer_output_type_2d_inputs(self):
        """Test infer_output_type returns TwoDTensor for 2D+2D (AC7)."""
        # Act
        output_type = GlobalConcat.infer_output_type((TwoDTensor, TwoDTensor))

        # Assert
        assert output_type == TwoDTensor

    def test_infer_output_type_3d_inputs(self):
        """Test infer_output_type returns ThreeDTensor for 3D+3D (AC7)."""
        # Act
        output_type = GlobalConcat.infer_output_type((ThreeDTensor, ThreeDTensor))

        # Assert
        assert output_type == ThreeDTensor

    def test_infer_output_type_mixed_dimensionality_error(self):
        """Test infer_output_type raises error for mixed 2D+3D (AC8)."""
        # Act & Assert
        with pytest.raises(VectorMeshError) as exc_info:
            GlobalConcat.infer_output_type((TwoDTensor, ThreeDTensor))

        # Verify error contains educational guidance
        error = exc_info.value
        assert "mixed" in str(error).lower() or "2d" in str(error).lower()
        assert error.hint is not None
        assert error.fix is not None

    def test_infer_output_type_empty_inputs_error(self):
        """Test infer_output_type raises error for empty tuple (AC8)."""
        # Act & Assert
        with pytest.raises(VectorMeshError) as exc_info:
            GlobalConcat.infer_output_type(())

        assert "at least one" in str(exc_info.value).lower()


class TestGlobalStackDefinitionTimeValidation:
    """Test suite for GlobalStack definition-time validation (AC7-8)."""

    def test_infer_output_type_2d_inputs(self):
        """Test infer_output_type returns ThreeDTensor for 2D+2D (AC7)."""
        # Act
        output_type = GlobalStack.infer_output_type((TwoDTensor, TwoDTensor))

        # Assert
        assert output_type == ThreeDTensor

    def test_infer_output_type_mixed_2d_3d_inputs(self):
        """Test infer_output_type returns ThreeDTensor for 3D+2D (AC7)."""
        # Act
        output_type = GlobalStack.infer_output_type((ThreeDTensor, TwoDTensor))

        # Assert
        assert output_type == ThreeDTensor

    def test_infer_output_type_3d_inputs(self):
        """Test infer_output_type returns FourDTensor for 3D+3D (AC7)."""
        # Act
        output_type = GlobalStack.infer_output_type((ThreeDTensor, ThreeDTensor))

        # Assert
        assert output_type == FourDTensor

    def test_infer_output_type_empty_inputs_error(self):
        """Test infer_output_type raises error for empty tuple (AC8)."""
        # Act & Assert
        with pytest.raises(VectorMeshError) as exc_info:
            GlobalStack.infer_output_type(())

        assert "at least one" in str(exc_info.value).lower()


class TestConnectorIntegration:
    """Integration tests with Parallel and Serial combinators (AC9)."""

    def test_parallel_2d_2d_with_concat(self):
        """Test Parallel with 2D vectorizers followed by GlobalConcat."""
        pytest.skip("Requires real vectorizer implementations")

    def test_parallel_3d_3d_with_concat(self):
        """Test Parallel with 3D vectorizers followed by GlobalConcat."""
        pytest.skip("Requires real vectorizer implementations")

    def test_parallel_2d_2d_with_stack(self):
        """Test Parallel with 2D vectorizers followed by GlobalStack."""
        pytest.skip("Requires real vectorizer implementations")

    def test_parallel_mixed_with_stack(self):
        """Test Parallel with mixed 3D+2D followed by GlobalStack."""
        pytest.skip("Requires real vectorizer implementations")

    def test_full_pipeline_concat_then_aggregator(self):
        """Test full pipeline: Parallel → GlobalConcat → MeanAggregator."""
        pytest.skip("Requires real vectorizer and aggregator implementations")
