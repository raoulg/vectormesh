"""Tests for curated model registry."""
import pytest
from unittest.mock import patch

from vectormesh.zoo.models import (
    CuratedModel, MPNET, QWEN_0_6B, LABSE, MINILM,
    BGE_GEMMA2, E5_MISTRAL, QWEN_8B, DISTILUSE,
    MVP_MODELS, GROWTH_MODELS, ALL_MODELS
)
from vectormesh.utils.model_info import get_model_metadata


class TestCuratedModel:
    """Test CuratedModel dataclass."""

    def test_curated_model_immutable(self):
        """Test that CuratedModel is frozen/immutable."""
        model = CuratedModel(
            model_id="test",
            context_window=512,
            embedding_dim=768,
            output_mode="2d",
            description="Test model"
        )

        # Should not be able to modify
        with pytest.raises(Exception):  # dataclass frozen error
            model.embedding_dim = 1024

    def test_curated_model_validation(self):
        """Test CuratedModel validates output_mode values."""
        # Valid output_mode
        model = CuratedModel(
            model_id="test",
            context_window=512,
            embedding_dim=768,
            output_mode="2d",
            description="Test"
        )
        assert model.output_mode == "2d"

        # Valid 3d mode
        model3d = CuratedModel(
            model_id="test",
            context_window=512,
            embedding_dim=768,
            output_mode="3d",
            description="Test"
        )
        assert model3d.output_mode == "3d"


class TestModelConstants:
    """Test individual model constants."""

    def test_mvp_models_structure(self):
        """Test MVP model constants have correct structure."""
        assert len(MVP_MODELS) == 4

        # Test MPNET
        assert MPNET.model_id == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        assert MPNET.output_mode == "2d"
        assert MPNET.context_window == 512
        assert MPNET.embedding_dim == 768
        assert "dutch" in MPNET.description.lower() or "multilingual" in MPNET.description.lower()

        # Test QWEN
        assert QWEN_0_6B.output_mode == "3d"
        assert QWEN_0_6B.context_window == 32768

        # Test LABSE
        assert LABSE.output_mode == "2d"
        assert "109 languages" in LABSE.description

        # Test MINILM
        assert MINILM.output_mode == "2d"
        assert MINILM.embedding_dim == 384

    def test_growth_models_structure(self):
        """Test Growth models have correct structure."""
        assert len(GROWTH_MODELS) == 6

        # Test BGE Gemma2
        assert BGE_GEMMA2.output_mode == "3d"
        assert BGE_GEMMA2.context_window == 8192
        assert BGE_GEMMA2.embedding_dim == 3584

    def test_all_models_collection(self):
        """Test ALL_MODELS includes all models."""
        assert len(ALL_MODELS) == 10
        assert len(ALL_MODELS) == len(MVP_MODELS) + len(GROWTH_MODELS)

        # Test that MVP models are in ALL_MODELS
        for mvp_model in MVP_MODELS:
            assert mvp_model in ALL_MODELS

        # Test that Growth models are in ALL_MODELS
        for growth_model in GROWTH_MODELS:
            assert growth_model in ALL_MODELS

    def test_model_ids_unique(self):
        """Test that all model IDs are unique."""
        model_ids = [model.model_id for model in ALL_MODELS]
        assert len(model_ids) == len(set(model_ids))  # No duplicates

    def test_output_modes_valid(self):
        """Test that all models have valid output modes."""
        for model in ALL_MODELS:
            assert model.output_mode in ["2d", "3d"]

    def test_context_windows_positive(self):
        """Test that all context windows are positive integers."""
        for model in ALL_MODELS:
            assert isinstance(model.context_window, int)
            assert model.context_window > 0

    def test_embedding_dims_positive(self):
        """Test that all embedding dimensions are positive integers."""
        for model in ALL_MODELS:
            assert isinstance(model.embedding_dim, int)
            assert model.embedding_dim > 0

    def test_descriptions_exist(self):
        """Test that all models have descriptions."""
        for model in ALL_MODELS:
            assert isinstance(model.description, str)
            assert len(model.description) > 0


class TestModelMetadataValidation:
    """Test curated model metadata matches AutoConfig when possible."""

    @pytest.mark.parametrize("model", MVP_MODELS)
    @patch("vectormesh.utils.model_info.get_model_metadata")
    def test_mvp_model_metadata_consistency(self, mock_get_metadata, model):
        """Test MVP model metadata is internally consistent."""
        # We can't easily test against real AutoConfig in unit tests,
        # so we test internal consistency
        assert model.model_id.startswith(("sentence-transformers/", "Qwen/", "bert-", "xlm-"))

        # 2D models should be sentence-transformers
        if model.output_mode == "2d":
            assert "sentence-transformers" in model.model_id

        # 3D models should not be sentence-transformers
        if model.output_mode == "3d":
            assert "sentence-transformers" not in model.model_id

    def test_model_registry_importable(self):
        """Test that model constants can be imported."""
        from vectormesh.zoo.models import MPNET, ALL_MODELS

        assert MPNET is not None
        assert len(ALL_MODELS) == 10