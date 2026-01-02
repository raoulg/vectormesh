"""Basic tests for TextVectorizer properties."""
import pytest
from unittest.mock import Mock, patch

from vectormesh.components.vectorizers import TextVectorizer


def test_text_vectorizer_properties():
    """Test basic properties without network calls."""
    with patch("vectormesh.utils.model_info.get_model_metadata") as mock_get_metadata:
        # Mock the metadata
        mock_metadata = Mock()
        mock_metadata.output_mode = "2d"
        mock_metadata.hidden_size = 384
        mock_metadata.max_position_embeddings = 512
        mock_metadata.pooling_strategy = "mean"
        mock_get_metadata.return_value = mock_metadata

        vectorizer = TextVectorizer(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Test properties
        assert vectorizer.output_mode == "2d"
        assert vectorizer.embedding_dim == 384
        assert vectorizer.context_window == 512
        assert vectorizer.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert vectorizer.auto_chunk is True


def test_text_vectorizer_3d_properties():
    """Test 3D model properties."""
    with patch("vectormesh.utils.model_info.get_model_metadata") as mock_get_metadata:
        # Mock the metadata for 3D model
        mock_metadata = Mock()
        mock_metadata.output_mode = "3d"
        mock_metadata.hidden_size = 768
        mock_metadata.max_position_embeddings = 512
        mock_get_metadata.return_value = mock_metadata

        vectorizer = TextVectorizer(
            model_name="bert-base-uncased",
            chunk_size=256  # Override default
        )

        # Test properties
        assert vectorizer.output_mode == "3d"
        assert vectorizer.embedding_dim == 768
        assert vectorizer.context_window == 256  # Should use override
        assert vectorizer.auto_chunk is True


def test_text_vectorizer_auto_chunk_false():
    """Test auto_chunk=False parameter."""
    with patch("vectormesh.utils.model_info.get_model_metadata") as mock_get_metadata:
        mock_metadata = Mock()
        mock_metadata.output_mode = "3d"
        mock_metadata.hidden_size = 768
        mock_metadata.max_position_embeddings = 512
        mock_get_metadata.return_value = mock_metadata

        vectorizer = TextVectorizer(
            model_name="bert-base-uncased",
            auto_chunk=False
        )

        assert vectorizer.auto_chunk is False
        assert vectorizer.output_mode == "3d"