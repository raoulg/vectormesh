"""Tests for VectorCache component."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from vectormesh.data.cache import VectorCache
from vectormesh.components.vectorizers import BaseVectorizer, TwoDVectorizer, ThreeDVectorizer
from vectormesh.errors import VectorMeshError


def load_jsonl(file_path: Path, limit: int = None) -> list[str]:
    """Load texts from JSONL file.

    Args:
        file_path: Path to JSONL file
        limit: Optional limit on number of records to load

    Returns:
        List of text strings from the 'text' field
    """
    texts = []
    with open(file_path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            record = json.loads(line)
            texts.append(record["text"])
    return texts


class TestVectorCache:
    """Test suite for VectorCache component."""

    def test_vectorcache_initialization(self):
        """Test that VectorCache can be initialized."""
        cache = VectorCache(name="test_cache")
        assert cache.name == "test_cache"
        assert cache.cache_dir == Path(".vmcache")

    def test_vectorcache_custom_cache_dir(self):
        """Test custom cache directory."""
        cache = VectorCache(name="test", cache_dir=Path("/custom/path"))
        assert cache.cache_dir == Path("/custom/path")

    def test_vectorcache_create_basic(self, tmp_path):
        """Test basic cache creation."""
        # Mock vectorizer
        mock_vectorizer = Mock(spec=BaseVectorizer)
        mock_vectorizer.return_value = torch.tensor([[0.1] * 384, [0.2] * 384])
        mock_vectorizer.model_name = "test-model"  # Add model_name attribute
        mock_vectorizer.output_mode = "2d"  # Add output_mode attribute

        texts = ["text1", "text2"]

        cache = VectorCache.create(
            texts=texts,
            vectorizer=mock_vectorizer,
            name="test_cache",
            cache_dir=tmp_path
        )

        assert cache.name == "test_cache"
        assert (tmp_path / "test_cache").exists()
        assert (tmp_path / "test_cache" / "metadata.json").exists()

    def test_vectorcache_batch_processing(self, tmp_path):
        """Test that texts are processed in batches."""
        mock_vectorizer = Mock(spec=BaseVectorizer)
        mock_vectorizer.model_name = "test-model"  # Add model_name attribute
        mock_vectorizer.output_mode = "2d"  # Add output_mode attribute
        # Return different embeddings for each batch
        mock_vectorizer.side_effect = [
            torch.tensor([[0.1] * 384, [0.2] * 384]),  # Batch 1
            torch.tensor([[0.3] * 384])  # Batch 2
        ]

        texts = ["text1", "text2", "text3"]

        _ = VectorCache.create(
            texts=texts,
            vectorizer=mock_vectorizer,
            name="test_batch",
            cache_dir=tmp_path,
            batch_size=2
        )

        # Should have been called twice (2 texts, then 1 text)
        assert mock_vectorizer.call_count == 2
        assert (tmp_path / "test_batch").exists()

    def test_vectorcache_atomic_creation(self, tmp_path):
        """Test atomic creation pattern - no partial corruption."""
        mock_vectorizer = Mock(spec=BaseVectorizer)
        mock_vectorizer.output_mode = "2d"
        mock_vectorizer.model_name = "test-model"
        mock_vectorizer.side_effect = Exception("Simulated failure")

        texts = ["text1", "text2"]

        with pytest.raises(VectorMeshError):
            VectorCache.create(
                texts=texts,
                vectorizer=mock_vectorizer,
                name="test_atomic",
                cache_dir=tmp_path
            )

        # Final cache dir should NOT exist
        assert not (tmp_path / "test_atomic").exists()
        # Temp dir should also be cleaned up
        temp_dirs = list(tmp_path.glob(".tmp_*"))
        assert len(temp_dirs) == 0

    def test_vectorcache_metadata_creation(self, tmp_path):
        """Test metadata.json is created with correct info."""
        mock_vectorizer = Mock(spec=BaseVectorizer)
        mock_vectorizer.return_value = torch.tensor([[0.1] * 384, [0.2] * 384])
        mock_vectorizer.model_name = "test-model"
        mock_vectorizer.output_mode = "2d"

        texts = ["text1", "text2"]

        _ = VectorCache.create(
            texts=texts,
            vectorizer=mock_vectorizer,
            name="test_meta",
            cache_dir=tmp_path
        )

        metadata_path = tmp_path / "test_meta" / "metadata.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["model_name"] == "test-model"
        assert metadata["output_mode"] == "2d"
        assert metadata["num_samples"] == 2
        assert metadata["embedding_dim"] == 384
        assert "created_at" in metadata

    def test_vectorcache_load_existing(self, tmp_path):
        """Test loading an existing cache."""
        # First create a cache
        mock_vectorizer = Mock(spec=BaseVectorizer)
        mock_vectorizer.return_value = torch.tensor([[0.1] * 384, [0.2] * 384])
        mock_vectorizer.model_name = "test-model"
        mock_vectorizer.output_mode = "2d"

        VectorCache.create(
            texts=["text1", "text2"],
            vectorizer=mock_vectorizer,
            name="test_load",
            cache_dir=tmp_path
        )

        # Now load it
        loaded_cache = VectorCache.load(name="test_load", cache_dir=tmp_path)

        assert loaded_cache.name == "test_load"
        assert loaded_cache.cache_dir == tmp_path
        assert loaded_cache.output_mode == "2d"

    def test_vectorcache_load_nonexistent_raises_error(self, tmp_path):
        """Test that loading nonexistent cache raises error."""
        with pytest.raises(VectorMeshError) as exc_info:
            VectorCache.load(name="nonexistent", cache_dir=tmp_path)

        error = exc_info.value
        assert "hint" in str(error).lower() or hasattr(error, "hint")

    def test_vectorcache_get_embeddings(self, tmp_path):
        """Test retrieving embeddings from cache."""
        mock_vectorizer = Mock(spec=BaseVectorizer)
        embeddings = torch.tensor([[0.1] * 384, [0.2] * 384, [0.3] * 384])
        mock_vectorizer.return_value = embeddings
        mock_vectorizer.model_name = "test-model"
        mock_vectorizer.output_mode = "2d"

        cache = VectorCache.create(
            texts=["text1", "text2", "text3"],
            vectorizer=mock_vectorizer,
            name="test_get",
            cache_dir=tmp_path
        )

        # Get all embeddings
        result = cache.get_embeddings()
        assert result.shape[0] == 3  # 3 samples
        assert result.shape[1] == 384  # 384 dimensions

        # Get specific indices
        result_slice = cache.get_embeddings(indices=[0, 2])
        assert result_slice.shape[0] == 2

    def test_vectorcache_frozen_config(self):
        """Test that VectorCache configuration is immutable."""
        cache = VectorCache(name="test")
        with pytest.raises(Exception):  # Pydantic ValidationError
            cache.name = "different"

    def test_vectorcache_empty_texts_handling(self, tmp_path):
        """Test handling of empty text list."""
        mock_vectorizer = Mock(spec=BaseVectorizer)
        mock_vectorizer.output_mode = "2d"

        with pytest.raises(VectorMeshError) as exc_info:
            VectorCache.create(
                texts=[],
                vectorizer=mock_vectorizer,
                name="test_empty",
                cache_dir=tmp_path
            )

        error = exc_info.value
        assert "empty" in str(error).lower()

    @pytest.mark.integration
    def test_vectorcache_integration_with_jsonl(self, tmp_path):
        """Integration test: Load JSONL data and create cache with real vectorizer.

        This test demonstrates the complete flow:
        1. Load JSONL file (assets/train.jsonl)
        2. Extract text fields
        3. Create VectorCache with real TextVectorizer
        4. Load cache and retrieve embeddings

        Note: This test is marked as 'integration' and uses a real model,
        so it's slower and requires internet connection on first run.
        Run with: pytest -m integration
        """
        # Path to JSONL file
        jsonl_path = Path(__file__).parent.parent.parent / "assets" / "train.jsonl"

        # Skip if file doesn't exist
        if not jsonl_path.exists():
            pytest.skip(f"JSONL file not found: {jsonl_path}")

        # Load texts from JSONL (limit to 10 for faster testing)
        texts = load_jsonl(jsonl_path, limit=10)
        assert len(texts) == 10
        assert all(isinstance(t, str) for t in texts)

        # Create real vectorizer
        vectorizer = TwoDVectorizer(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create cache with real data and real vectorizer
        cache = VectorCache.create(
            texts=texts,
            vectorizer=vectorizer,
            name="integration_test_cache",
            cache_dir=tmp_path,
            batch_size=5  # Process in 2 batches
        )

        # Verify cache was created
        assert cache.name == "integration_test_cache"
        assert (tmp_path / "integration_test_cache").exists()

        # Verify metadata
        metadata = cache.get_metadata()
        assert metadata["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert metadata["output_mode"] == "2d"
        assert metadata["num_samples"] == 10
        assert metadata["embedding_dim"] == 384

        # Get embeddings
        embeddings = cache.get_embeddings()
        assert embeddings.shape == (10, 384)

        # Verify embeddings are valid (not all zeros)
        assert not torch.allclose(embeddings, torch.zeros_like(embeddings))

        # Test loading from disk
        loaded_cache = VectorCache.load(
            name="integration_test_cache",
            cache_dir=tmp_path
        )

        # Get embeddings from loaded cache
        loaded_embeddings = loaded_cache.get_embeddings()
        assert torch.allclose(embeddings, loaded_embeddings)
        assert loaded_cache.output_mode == "2d"

        # Test partial retrieval
        partial = loaded_cache.get_embeddings(indices=[0, 5, 9])
        assert partial.shape == (3, 384)

    def test_vectorcache_aggregate_mean(self, tmp_path):
        """Test VectorCache.aggregate() with MeanAggregator strategy."""
        mock_vectorizer = Mock(spec=ThreeDVectorizer)
        # Return embeddings with 3D shape (batch=2, chunks=1, dim=4)
        embeddings = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0]],
            [[5.0, 6.0, 7.0, 8.0]]
        ])
        mock_vectorizer.return_value = embeddings
        mock_vectorizer.model_name = "test-model"
        mock_vectorizer.output_mode = "3d"

        cache = VectorCache.create(
            texts=["text1", "text2"],
            vectorizer=mock_vectorizer,
            name="test_agg_mean",
            cache_dir=tmp_path
        )

        # Test mean aggregation
        result = cache.aggregate(strategy="MeanAggregator")
        assert result.shape == (2, 4)
        # With single chunk, mean should equal the original values
        assert torch.allclose(result, torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))

    def test_vectorcache_aggregate_max(self, tmp_path):
        """Test VectorCache.aggregate() with MaxAggregator strategy."""
        mock_vectorizer = Mock(spec=ThreeDVectorizer)
        # Return embeddings with 3D shape (batch=2, chunks=1, dim=4)
        embeddings = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0]],
            [[5.0, 6.0, 7.0, 8.0]]
        ])
        mock_vectorizer.return_value = embeddings
        mock_vectorizer.model_name = "test-model"
        mock_vectorizer.output_mode = "3d"

        cache = VectorCache.create(
            texts=["text1", "text2"],
            vectorizer=mock_vectorizer,
            name="test_agg_max",
            cache_dir=tmp_path
        )

        # Test max aggregation
        result = cache.aggregate(strategy="MaxAggregator")
        assert result.shape == (2, 4)
        # With single chunk, max should equal the original values
        assert torch.allclose(result, torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))

    def test_vectorcache_aggregate_invalid_strategy(self, tmp_path):
        """Test that invalid aggregation strategy raises error."""
        mock_vectorizer = Mock(spec=ThreeDVectorizer)
        embeddings = torch.tensor([[[1.0, 2.0]]])
        mock_vectorizer.return_value = embeddings
        mock_vectorizer.model_name = "test-model"
        mock_vectorizer.output_mode = "3d"

        cache = VectorCache.create(
            texts=["text"],
            vectorizer=mock_vectorizer,
            name="test_invalid",
            cache_dir=tmp_path
        )

        # Invalid strategy should raise VectorMeshError from get_aggregator
        from vectormesh.errors import VectorMeshError
        with pytest.raises(VectorMeshError):
            cache.aggregate(strategy="InvalidAggregator")

    def test_vectorcache_aggregate_on_2d_raises_error(self, tmp_path):
        """Test that calling aggregate on a 2D cache raises an error."""
        mock_vectorizer = Mock(spec=BaseVectorizer)
        mock_vectorizer.return_value = torch.randn(2, 384)
        mock_vectorizer.model_name = "test-2d"
        mock_vectorizer.output_mode = "2d"

        cache = VectorCache.create(
            texts=["text1", "text2"],
            vectorizer=mock_vectorizer,
            name="test_agg_2d",
            cache_dir=tmp_path
        )

        from vectormesh.errors import VectorMeshError
        with pytest.raises(VectorMeshError) as exc_info:
            cache.aggregate()
        
        error = exc_info.value
        assert "Cannot aggregate 2D embeddings" in str(error)
