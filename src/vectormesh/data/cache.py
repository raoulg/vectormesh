"""Vector caching using HuggingFace Datasets for persistent storage."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, TypeVar, Generic

import numpy as np
import torch
from beartype.typing import List
from datasets import Dataset, load_from_disk
from pydantic import ConfigDict, Field

from vectormesh.types import VectorMeshComponent, VectorMeshError
from vectormesh.components.vectorizers import Vectorizer

TVectorizer = TypeVar("TVectorizer", bound=Vectorizer)

class VectorCache(VectorMeshComponent, Generic[TVectorizer]):
    """Cache embeddings to disk using HuggingFace Datasets (Arrow/Parquet format).

    This component provides persistent storage for text embeddings with atomic
    creation, memory-mapped loading, and automatic cleanup on failures.

    Args:
        name: Cache identifier (used as directory name)
        cache_dir: Root directory for all caches (default: .vmcache)

    Example:
        ```python
        # Create cache
        vectorizer = TwoDVectorizer(model_name="sentence-transformers/all-MiniLM-L6-v2")
        cache = VectorCache.create(
            texts=["hello", "world"],
            vectorizer=vectorizer,
            name="my_cache"
        )

        # Load cache later
        cache = VectorCache.load(name="my_cache")
        embeddings = cache.get_embeddings()
        ```

    Note:
        Uses atomic creation pattern to prevent corrupted caches from interruptions.
        All data is memory-mapped for zero-copy access.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    cache_dir: Path = Field(default=Path(".vmcache"))

    # Private cached dataset (not part of Pydantic fields)
    _dataset: Optional[Dataset] = None
    _metadata: Optional[dict] = None

    @classmethod
    def create(
        cls,
        texts: List[str],
        vectorizer: TVectorizer,
        name: str,
        cache_dir: Path = Path(".vmcache"),
        batch_size: int = 32,
    ) -> "VectorCache[TVectorizer]":
        """Create a new vector cache from texts.

        Args:
            texts: List of text strings to vectorize and cache
            vectorizer: TextVectorizer instance to use for embedding generation
            name: Cache identifier
            cache_dir: Root directory for caches
            batch_size: Number of texts to process per batch

        Returns:
            VectorCache instance with data loaded

        Raises:
            VectorMeshError: If cache creation fails

        Shapes:
            Input: List[str] with N texts
            Output: Cache with (N, embedding_dim) embeddings
        """
        # Validate inputs
        if not texts:
            raise VectorMeshError(
                message="Cannot create cache from empty text list",
                hint="Provide at least one text string",
                fix="Pass a non-empty list to VectorCache.create()",
            )

        # Setup paths
        cache_dir = Path(cache_dir)
        final_path = cache_dir / name
        temp_path = cache_dir / f".tmp_{name}"

        try:
            # Atomic creation: work in temp directory
            if temp_path.exists():
                shutil.rmtree(temp_path)

            temp_path.mkdir(parents=True, exist_ok=True)

            # Batch process texts through vectorizer
            all_embeddings = []
            num_batches = (len(texts) + batch_size - 1) // batch_size

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]

                # Vectorize batch
                batch_embeddings = vectorizer(batch_texts)
                all_embeddings.append(batch_embeddings.cpu().numpy())

            # Concatenate all batches
            embeddings_array = np.vstack(all_embeddings)

            # Create HuggingFace Dataset
            dataset_dict = {
                "embeddings": embeddings_array,
                "text_id": list(range(len(texts))),
            }
            dataset = Dataset.from_dict(dataset_dict)

            # Save dataset to temp location
            dataset.save_to_disk(str(temp_path / "dataset"))

            # Create metadata
            metadata = {
                "vectormesh_version": "0.1.0",
                "model_name": vectorizer.model_name,
                "output_mode": vectorizer.output_mode,  # Persist the mode (2d/3d)
                "embedding_dim": int(embeddings_array.shape[1]),
                "num_samples": len(texts),
                "created_at": datetime.now().isoformat(),
                "cache_format": "huggingface_datasets_v1",
            }

            metadata_path = temp_path / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Atomic rename: only if everything succeeded
            if final_path.exists():
                shutil.rmtree(final_path)
            temp_path.rename(final_path)

            # Load and return cache
            return cls.load(name=name, cache_dir=cache_dir)

        except VectorMeshError:
            # Re-raise our custom errors
            if temp_path.exists():
                shutil.rmtree(temp_path)
            raise

        except Exception as e:
            # Cleanup temp directory on any failure
            if temp_path.exists():
                shutil.rmtree(temp_path)

            raise VectorMeshError(
                message=f"Failed to create cache '{name}': {str(e)}",
                hint="Check disk space, permissions, and vectorizer functionality",
                fix="Ensure you have write permissions and sufficient disk space in the cache directory",
            ) from e

    @classmethod
    def load(cls, name: str, cache_dir: Path = Path(".vmcache")) -> "VectorCache":
        """Load an existing vector cache.

        Args:
            name: Cache identifier
            cache_dir: Root directory for caches

        Returns:
            VectorCache instance with memory-mapped data

        Raises:
            VectorMeshError: If cache doesn't exist or is corrupted
        """
        cache_dir = Path(cache_dir)
        cache_path = cache_dir / name

        # Validate cache exists
        if not cache_path.exists():
            raise VectorMeshError(
                message=f"Cache '{name}' not found in {cache_dir}",
                hint="Cache has not been created yet",
                fix=f"Create the cache first using VectorCache.create(..., name='{name}')",
            )

        # Validate metadata exists
        metadata_path = cache_path / "metadata.json"
        if not metadata_path.exists():
            raise VectorMeshError(
                message=f"Cache '{name}' is corrupted (missing metadata.json)",
                hint="Cache creation may have been interrupted",
                fix="Delete the corrupted cache directory and recreate it",
            )

        try:
            # Load metadata
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Load dataset (memory-mapped)
            dataset = load_from_disk(str(cache_path / "dataset"))

            # Create instance
            instance = cls(name=name, cache_dir=cache_dir)

            # Cache dataset and metadata (bypass frozen config)
            object.__setattr__(instance, "_dataset", dataset)
            object.__setattr__(instance, "_metadata", metadata)

            return instance

        except Exception as e:
            raise VectorMeshError(
                message=f"Failed to load cache '{name}': {str(e)}",
                hint="Cache may be corrupted or incompatible",
                fix="Try recreating the cache or check file permissions",
            ) from e

    def get_embeddings(
        self, indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Retrieve embeddings from the cache.

        Args:
            indices: Optional list of indices to retrieve (None = all)

        Returns:
            Tensor of embeddings with shape (N, embedding_dim)

        Raises:
            VectorMeshError: If cache not loaded properly
        """
        if self._dataset is None:
            raise VectorMeshError(
                message="Cache dataset not loaded",
                hint="Load the cache first using VectorCache.load()",
                fix="Call VectorCache.load() before accessing embeddings",
            )

        try:
            if indices is None:
                # Get all embeddings
                embeddings_np = np.array(self._dataset["embeddings"])
            else:
                # Get specific indices
                subset = self._dataset.select(indices)
                embeddings_np = np.array(subset["embeddings"])

            return torch.from_numpy(embeddings_np).float()

        except Exception as e:
            raise VectorMeshError(
                message=f"Failed to retrieve embeddings: {str(e)}",
                hint="Cache data may be corrupted",
                fix="Recreate the cache if the error persists",
            ) from e

    def get_metadata(self) -> dict:
        """Get cache metadata.

        Returns:
            Dictionary containing cache metadata

        Raises:
            VectorMeshError: If metadata not loaded
        """
        if self._metadata is None:
            raise VectorMeshError(
                message="Cache metadata not loaded",
                hint="Load the cache first using VectorCache.load()",
                fix="Call VectorCache.load() before accessing metadata",
            )

        return self._metadata.copy()

    @property
    def output_mode(self) -> str:
        """Get the output mode (2d or 3d) of the cached embeddings."""
        if self._metadata is None:
            # Should have been loaded by load() or create()
            # But if accessed before load(), maybe error?
            # It's safer to rely on get_metadata raising error.
             return self.get_metadata().get("output_mode", "2d") # Default to 2d for backward compat
        return self._metadata.get("output_mode", "2d")

    def aggregate(self, strategy: str = "MeanAggregator") -> torch.Tensor:
        """Aggregate embeddings using specified aggregation strategy.

        Convenience method that loads an aggregator by name and applies it to
        the cached embeddings. For custom aggregation logic, define a class
        inheriting from BaseAggregator.

        Args:
            strategy: Aggregator class name (e.g., "MeanAggregator", "MaxAggregator")

        Returns:
            Aggregated tensor of shape (batch, dim)

        Raises:
            VectorMeshError: If aggregator not found

        Shapes:
            Input: get_embeddings() returns (batch, chunks, dim)
            Output: (batch, dim)

        Example:
            ```python
            cache = VectorCache.load(name="my_cache")

            # Use built-in aggregator
            mean_result = cache.aggregate(strategy="MeanAggregator")
            max_result = cache.aggregate(strategy="MaxAggregator")

            # Or use custom aggregator (must be defined and imported)
            custom_result = cache.aggregate(strategy="MyCustomAggregator")
            ```
        """
        from vectormesh.components.aggregation import get_aggregator

        if self.output_mode == "2d":
            raise VectorMeshError(
                message="Cannot aggregate 2D embeddings",
                hint="Aggregation is only for 3D (chunked) embeddings to reduce them to 2D",
                fix="Use the embeddings directly via get_embeddings(), they are already pooled."
            )

        embeddings = self.get_embeddings()
        aggregator = get_aggregator(strategy)
        return aggregator(embeddings)
