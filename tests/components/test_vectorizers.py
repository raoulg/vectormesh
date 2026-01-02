"""Tests for TextVectorizer component."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from vectormesh.components.vectorizers import TextVectorizer
from vectormesh.errors import VectorMeshError


class TestTextVectorizer:
    """Test suite for TextVectorizer component."""

    def test_textvectorizer_initialization_with_model_name(self):
        """Test that TextVectorizer can be initialized with a model name."""
        vectorizer = TextVectorizer(model_name="sentence-transformers/all-MiniLM-L6-v2")
        assert vectorizer.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert vectorizer.device is None  # Should be None until auto-detected

    def test_textvectorizer_initialization_with_device_override(self):
        """Test that device can be explicitly set."""
        vectorizer = TextVectorizer(
            model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        assert vectorizer.device == "cpu"

    @patch("vectormesh.components.vectorizers.SentenceTransformer")
    def test_textvectorizer_encode_returns_correct_shape(self, mock_st):
        """Test that encoding returns correctly shaped tensor."""
        # Setup mock
        mock_model = Mock()
        mock_embeddings = np.array([[0.1] * 384, [0.2] * 384, [0.3] * 384])  # 3x384
        mock_model.encode.return_value = mock_embeddings
        mock_st.return_value = mock_model

        # Test
        vectorizer = TextVectorizer(model_name="test-model", device="cpu")
        result = vectorizer(["text1", "text2", "text3"])

        # Verify
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 384)
        mock_model.encode.assert_called_once_with(["text1", "text2", "text3"])

    @patch("vectormesh.components.vectorizers.SentenceTransformer")
    def test_textvectorizer_device_auto_detection(self, mock_st):
        """Test that device is auto-detected when not specified."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1] * 384])  # Mock encode output
        mock_st.return_value = mock_model

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                vectorizer = TextVectorizer(model_name="test-model")
                # Trigger model loading by calling
                vectorizer(["test"])
                # Should have detected CPU as device
                # Check that SentenceTransformer was called with detected device
                call_args = mock_st.call_args
                assert call_args[1]["device"] == "cpu"

    @patch("vectormesh.components.vectorizers.SentenceTransformer")
    def test_textvectorizer_mps_device_detection(self, mock_st):
        """Test MPS device detection on macOS."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1] * 384])  # Mock encode output
        mock_st.return_value = mock_model

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=True):
                vectorizer = TextVectorizer(model_name="test-model")
                vectorizer(["test"])
                # Should have detected MPS
                call_args = mock_st.call_args
                assert call_args[1]["device"] == "mps"

    @patch("vectormesh.components.vectorizers.SentenceTransformer")
    def test_textvectorizer_cuda_device_detection(self, mock_st):
        """Test CUDA device detection."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1] * 384])  # Mock encode output
        mock_st.return_value = mock_model

        with patch("torch.cuda.is_available", return_value=True):
            vectorizer = TextVectorizer(model_name="test-model")
            vectorizer(["test"])
            # Should have detected CUDA
            call_args = mock_st.call_args
            assert call_args[1]["device"] == "cuda"

    @patch("vectormesh.components.vectorizers.SentenceTransformer")
    def test_textvectorizer_invalid_model_raises_vectormesh_error(self, mock_st):
        """Test that invalid model names raise VectorMeshError with helpful message."""
        mock_st.side_effect = OSError("Model not found")

        with pytest.raises(VectorMeshError) as exc_info:
            vectorizer = TextVectorizer(model_name="invalid/model-name")
            vectorizer(["test"])  # Trigger model loading

        error = exc_info.value
        assert "hint" in str(error).lower() or hasattr(error, "hint")
        assert "fix" in str(error).lower() or hasattr(error, "fix")

    def test_textvectorizer_frozen_config(self):
        """Test that TextVectorizer configuration is immutable."""
        vectorizer = TextVectorizer(model_name="test-model")
        with pytest.raises(Exception):  # Pydantic will raise ValidationError or similar
            vectorizer.model_name = "different-model"

    @patch("vectormesh.components.vectorizers.SentenceTransformer")
    def test_textvectorizer_model_caching(self, mock_st):
        """Test that model is loaded only once and cached."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1] * 384])
        mock_st.return_value = mock_model

        vectorizer = TextVectorizer(model_name="test-model", device="cpu")

        # Call multiple times
        vectorizer(["text1"])
        vectorizer(["text2"])
        vectorizer(["text3"])

        # SentenceTransformer should be instantiated only once
        assert mock_st.call_count == 1

    @patch("vectormesh.components.vectorizers.SentenceTransformer")
    def test_textvectorizer_empty_input_handling(self, mock_st):
        """Test handling of empty input list."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([]).reshape(0, 384)
        mock_st.return_value = mock_model

        vectorizer = TextVectorizer(model_name="test-model", device="cpu")
        result = vectorizer([])

        assert result.shape == (0, 384)
