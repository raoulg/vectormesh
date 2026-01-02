"""Tests for vectorizer compatibility and optimistic loading."""
import pytest
import torch
from unittest.mock import Mock, patch

from vectormesh.components.vectorizers import TwoDVectorizer, ThreeDVectorizer
from vectormesh.types import VectorMeshError


class TestTwoDVectorizerCompatibility:
    """Test TwoDVectorizer compatibility checks."""

    @patch("vectormesh.components.vectorizers.get_model_metadata")
    @patch("vectormesh.components.vectorizers.SentenceTransformer")
    def test_twod_vectorizer_happy_path(self, mock_sentence_transformer, mock_get_metadata):
        """Test successful optimistic loading."""
        # Mock successful load
        mock_model = Mock()
        mock_model.encode.return_value = torch.randn(2, 384)
        mock_sentence_transformer.return_value = mock_model

        # Mock metadata for strict check
        mock_metadata = Mock()
        mock_metadata.output_mode = "2d"
        mock_get_metadata.return_value = mock_metadata

        vectorizer = TwoDVectorizer(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Verify call works
        result = vectorizer(["test"])
        assert result.shape == (2, 384)
        
        # Verify strict check was called
        mock_get_metadata.assert_called()

    @patch("vectormesh.components.vectorizers.get_model_metadata")
    @patch("vectormesh.components.vectorizers.SentenceTransformer")
    def test_twod_vectorizer_fails_with_3d_model(self, mock_sentence_transformer, mock_get_metadata):
        """Test failure when trying to load a 3D model with TwoDVectorizer."""
        # Mock loading failure
        mock_sentence_transformer.side_effect = Exception("Model not found or invalid")
        
        # Mock introspection confirming it's a 3D model
        mock_metadata = Mock()
        mock_metadata.output_mode = "3d"
        mock_get_metadata.return_value = mock_metadata

        with pytest.raises(VectorMeshError) as exc_info:
            vectorizer = TwoDVectorizer(model_name="bert-base-uncased")
            vectorizer(["test"]) # Trigger optimistic load

        error = exc_info.value
        assert "TwoDVectorizer requires a 2D model" in str(error)
        assert "This model produces 3D output" in error.hint
        assert "ThreeDVectorizer" in error.fix

    @patch("vectormesh.components.vectorizers.get_model_metadata")
    @patch("vectormesh.components.vectorizers.SentenceTransformer")
    def test_twod_vectorizer_fails_generic(self, mock_sentence_transformer, mock_get_metadata):
        """Test generic failure when it IS a 2D model but still fails to load."""
        # Mock loading failure
        mock_sentence_transformer.side_effect = Exception("Connection error")
        
        # Mock introspection confirming it IS a 2D model
        mock_metadata = Mock()
        mock_metadata.output_mode = "2d"
        mock_get_metadata.return_value = mock_metadata

        with pytest.raises(VectorMeshError) as exc_info:
            vectorizer = TwoDVectorizer(model_name="valid-2d-model")
            vectorizer(["test"])

        error = exc_info.value
        # Should raise the generic loading error, NOT the compatibility error
        assert "Failed to load user-specified 2D model" in str(error)


class TestThreeDVectorizerCompatibility:
    """Test ThreeDVectorizer compatibility checks."""

    @patch("vectormesh.components.vectorizers.get_model_metadata")
    @patch("vectormesh.components.vectorizers.AutoTokenizer")
    @patch("vectormesh.components.vectorizers.AutoModel")
    def test_threed_vectorizer_happy_path(self, mock_auto_model, mock_auto_tokenizer, mock_get_metadata):
        """Test successful optimistic loading."""
        # Mock successful load
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.device = torch.device("cpu")
        mock_model.to.return_value = mock_model  # Support fluent interface
        mock_model.return_value = Mock(last_hidden_state=torch.randn(1, 3, 768))
        mock_auto_model.from_pretrained.return_value = mock_model

        # Mock metadata needed for chunking (context_window, etc)
        mock_metadata = Mock()
        mock_metadata.max_position_embeddings = 512
        mock_metadata.output_mode = "3d"
        mock_get_metadata.return_value = mock_metadata

        vectorizer = ThreeDVectorizer(model_name="bert-base-uncased")
        
        # Verify call works; triggers lazy load
        vectorizer(["test"])
        
        # Verify model loaded

    @patch("vectormesh.components.vectorizers.get_model_metadata")
    @patch("vectormesh.components.vectorizers.AutoModel")
    def test_threed_vectorizer_fails_with_2d_model(self, mock_auto_model, mock_get_metadata):
        """Test failure when trying to load a 2D model with ThreeDVectorizer."""
        # Note: We rely on PRE-CHECK to fail before loading.
        # So we don't necessarily need mock_auto_model to fail,
        # but we DO want to verify it wasn't called.
        
        # Mock introspection confirming it's a 2D model
        mock_metadata = Mock()
        mock_metadata.output_mode = "2d"
        mock_get_metadata.return_value = mock_metadata

        with pytest.raises(VectorMeshError) as exc_info:
            vectorizer = ThreeDVectorizer(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorizer(["test"]) 

        # Verify we didn't waste time downloading the model
        mock_auto_model.from_pretrained.assert_not_called()

        error = exc_info.value
        assert "ThreeDVectorizer requires a 3D model" in str(error)
        assert "This model produces 2D output" in error.hint
        assert "TwoDVectorizer" in error.fix