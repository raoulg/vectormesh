"""Tests for curated model registry."""
import pytest
from unittest.mock import patch, Mock

from vectormesh.zoo.models import (
    ZooModel, MPNET, LABSE, MINILM,
    BGE_GEMMA2, E5_MISTRAL, DISTILUSE,
    ESSENTIAL_MODELS, EXTENDED_MODELS, ALL_MODELS
)
from vectormesh.utils.model_info import get_model_metadata


class TestZooModel:
    """Test CuratedModel dataclass."""

    def test_zoo_model_immutable(self):
        """Test that CuratedModel is frozen/immutable."""
        model = ZooModel(
            model_id="test",
            context_window=512,
            embedding_dim=768,
            output_mode="2d",
            description="Test model"
        )

        # Should not be able to modify
        with pytest.raises(Exception):  # dataclass frozen error
            model.embedding_dim = 1024

    def test_zoo_model_validation(self):
        """Test CuratedModel validates output_mode values."""
        # Valid output_mode
        model = ZooModel(
            model_id="test",
            context_window=512,
            embedding_dim=768,
            output_mode="2d",
            description="Test"
        )
        assert model.output_mode == "2d"

        # Valid 3d mode
        model3d = ZooModel(
            model_id="test",
            context_window=512,
            embedding_dim=768,
            output_mode="3d",
            description="Test"
        )
        assert model3d.output_mode == "3d"


class TestModelConstants:
    """Test model constants against automated metadata extraction."""

    @patch("vectormesh.utils.model_info.AutoConfig.from_pretrained")
    def test_models_match_metadata_logic(self, mock_config):
        """Verify that curated models match what get_model_metadata would derive."""
        # This ensures our hardcoded constants aren't drifting from our own extraction logic.
        
        for model in ALL_MODELS:
            # Create a mock config that mimics the properties we expect for this model
            # This is validating that IF the model is what we think it is, ZooModel is defined correctly.
            mock_conf = Mock()
            mock_conf.max_position_embeddings = model.context_window
            mock_conf.hidden_size = model.embedding_dim
            
            # For 2D models, we expect sentence-transformers pooling
            if model.output_mode == "2d":
                mock_conf.pooling_mode_mean_tokens = True
            else:
                del mock_conf.pooling_mode_mean_tokens
            
            mock_config.return_value = mock_conf

            # Run extraction
            metadata = get_model_metadata(model.model_id)

            # Assert our hardcoded constant matches the extracted metadata
            assert metadata.output_mode == model.output_mode, f"Mismatch for {model.model_id}"
            assert metadata.max_position_embeddings == model.context_window
            assert metadata.hidden_size == model.embedding_dim

    @pytest.mark.integration
    def test_zoo_models_match_huggingface_reality(self):
        """Integration test: Verify curated constants against REAL HuggingFace API.
        
        This test actually calls the HF Hub (via get_model_metadata) to ensure
        that our hardcoded ZooModel definitions are accurate to reality.
        """
        for model in ALL_MODELS:
            # Fetch real metadata
            print(f"Fetching metadata for {model.model_id}...")
            # Note: get_model_metadata downloads config.json (<10kb) which is fast enough for integration test
            real_metadata = get_model_metadata(model.model_id)
            
            assert real_metadata.model_id == model.model_id
            assert real_metadata.max_position_embeddings == model.context_window
            assert real_metadata.hidden_size == model.embedding_dim
            assert real_metadata.output_mode == model.output_mode