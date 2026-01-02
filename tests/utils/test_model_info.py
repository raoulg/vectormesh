"""Tests for model introspection module."""
import pytest
from vectormesh.utils.model_info import get_model_metadata, ModelMetadata
from vectormesh.types import VectorMeshError


class TestGetModelMetadata:
    """Test cases for get_model_metadata function."""

    def test_get_model_metadata_sentence_transformer(self):
        """Test AutoConfig introspection for sentence-transformer."""
        metadata = get_model_metadata("sentence-transformers/all-MiniLM-L6-v2")
        assert metadata.output_mode == "2d"
        assert metadata.max_position_embeddings == 512
        assert metadata.hidden_size == 384
        assert metadata.pooling_strategy == "mean"

    def test_get_model_metadata_raw_transformer(self):
        """Test AutoConfig introspection for raw transformer."""
        metadata = get_model_metadata("bert-base-uncased")
        assert metadata.output_mode == "3d"
        assert metadata.max_position_embeddings == 512
        assert metadata.hidden_size == 768
        assert metadata.pooling_strategy is None  # No built-in pooling

    def test_get_model_metadata_large_context(self):
        """Test large context window detection."""
        # Using a model that should have large context window
        metadata = get_model_metadata("microsoft/DialoGPT-medium")
        assert metadata.max_position_embeddings >= 1024  # Large context
        assert metadata.output_mode == "3d"

    def test_get_model_metadata_invalid_model(self):
        """Test error handling for nonexistent model."""
        with pytest.raises(VectorMeshError) as exc_info:
            get_model_metadata("invalid/nonexistent-model")

        error = exc_info.value
        assert "failed to load config" in str(error).lower()
        assert error.hint is not None
        assert error.fix is not None

    def test_model_metadata_immutable(self):
        """Test that ModelMetadata is frozen/immutable."""
        metadata = ModelMetadata(
            model_id="test",
            max_position_embeddings=512,
            hidden_size=768,
            output_mode="2d",
            pooling_strategy="mean"
        )

        # Should not be able to modify
        with pytest.raises(Exception):  # ValidationError or similar
            metadata.hidden_size = 1024

    def test_model_metadata_validation(self):
        """Test ModelMetadata validates output_mode values."""
        # Valid output_mode
        metadata = ModelMetadata(
            model_id="test",
            max_position_embeddings=512,
            hidden_size=768,
            output_mode="2d"
        )
        assert metadata.output_mode == "2d"

        # Invalid output_mode should raise validation error
        with pytest.raises(Exception):  # ValidationError
            ModelMetadata(
                model_id="test",
                max_position_embeddings=512,
                hidden_size=768,
                output_mode="invalid"  # Not "2d" or "3d"
            )