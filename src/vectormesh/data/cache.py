import hashlib
import json
import shutil
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Generic, Optional, TypeVar, get_args, get_type_hints

from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk
from loguru import logger

from vectormesh.types import Cachable, VectorMeshError

from .vectorizers import BaseVectorizer

TVectorizer = TypeVar("TVectorizer", bound=BaseVectorizer)


class VectorCache(Cachable, Generic[TVectorizer]):
    name: str
    cache_dir: Path
    dataset: Optional[Dataset] = None
    metadata: Optional[dict] = None

    @classmethod
    def create(
        cls,
        cache_dir: Path,
        vectorizer: TVectorizer,
        dataset: Dataset,
        dataset_tag: str = "default",
        features: Optional[Features] = None,
        vector_batch: int = 32,
        map_batch: int = 32,
        column_name: Optional[str] = None,
        remove_columns: Optional[list[str]] = None,
    ) -> "VectorCache[TVectorizer]":
        """
        Args:
            cache_dir (Path): the location to store the cache
            vectorizer (TVectorizer): the vectorizer to apply to the dataset to vectorize the text
            dataset (Dataset): the dataset that provides the text to vectorize
            dataset_tag (Optional[str], optional): a tag used to identify preprocessing and versions.
                This will be used to create a new cache folder cache_dire/dataset_tag_{column_name} inside cache_dir.
                Will look for existing metadata.json in cache_dir/dataset_tag to extend
            features (Optional[Features], optional): the features of the provided dataset. Will be handled automatically if not provided. Defaults to None.
            vector_batch (Optional[int], optional): the batchsize of the vectorizer (eg the huggingface model). Defaults to 32.
            map_batch (Optional[int], optional): The batchsize for the mapping over the dataset. Defaults to 32.
            column_name (Optional[str], optional): how to store the output of the vectorizer in the dataset. If not provided, will use vectorizer.col_name. Defaults to None.
            remove_columns (Optional[list[str]], optional): source columns to drop from the cached dataset. Useful to keep a distributable cache small, e.g. remove_columns=["image"] after embedding. Defaults to None.

        Returns:
            VectorCache[TVectorizer]
        """
        column_name = cls._resolve_column(vectorizer, column_name)
        logger.info(f"Using embedding column: {column_name}")

        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created cache directory at {cache_dir}")

        if features is None:
            features = cls.get_features(
                dataset, cls.get_dtensor(vectorizer), embedding_column=column_name
            )
        # Drop any requested source columns from the output schema so it matches
        # the dataset returned by map(..., remove_columns=...).
        for col in remove_columns or []:
            features.pop(col, None)

        now = datetime.now().strftime("%Y%m%d%H%M%S")
        cachetag = f"{now}_{dataset_tag}_{column_name}"
        filepath = cache_dir / cachetag
        logger.info(f"Starting {cachetag}")

        try:
            new_dataset = cls._vectorize(
                dataset, vectorizer, features, vector_batch, map_batch, remove_columns
            )
            metadata = cls._build_metadata(
                vectorizer,
                column_name,
                features,
                num_observations=len(new_dataset),
                dataset=new_dataset,
            )
            # check for existing metadata to update
            metadata = cls.update_metadata(cache_dir / dataset_tag, metadata)
            cls._write(new_dataset, filepath, metadata)
        except Exception as e:
            # save_to_disk writes a *directory*, so remove the tree (not unlink).
            shutil.rmtree(filepath, ignore_errors=True)
            raise VectorMeshError(f"Failed to create cache at {filepath}") from e

        new_dataset.set_format(type="torch")
        logger.success(f"Cache saved to {filepath}")
        return cls(
            name=cachetag,
            cache_dir=cache_dir,
            dataset=new_dataset,
            metadata=metadata,
        )

    @staticmethod
    def _resolve_column(vectorizer, column_name: Optional[str]) -> str:
        """Pick the output column: the explicit override, else vectorizer.col_name."""
        if column_name is None:
            if not vectorizer.col_name:
                raise VectorMeshError(
                    "column_name must be provided if vectorizer.col_name is not set."
                )
            column_name = vectorizer.col_name
        return column_name

    @classmethod
    def _vectorize(
        cls,
        dataset: Dataset,
        vectorizer,
        features: Features,
        vector_batch: int,
        map_batch: int,
        remove_columns: Optional[list[str]],
    ) -> Dataset:
        """Map the vectorizer over the dataset, reading from vectorizer.input_col."""
        return dataset.map(
            lambda batch: vectorizer(
                batch[vectorizer.input_col], batchsize=vector_batch
            ),
            batched=True,
            batch_size=map_batch,  # Number of documents per batch
            features=features,
            remove_columns=remove_columns,
            new_fingerprint=cls._map_fingerprint(
                dataset, vectorizer, features, vector_batch, map_batch, remove_columns
            ),
        )

    @staticmethod
    def _map_fingerprint(
        dataset: Dataset,
        vectorizer,
        features: Features,
        vector_batch: int,
        map_batch: int,
        remove_columns: Optional[list[str]],
    ) -> str:
        """Fingerprint the map() above from what actually identifies its output.

        Without an explicit `new_fingerprint`, `datasets` derives one by
        pickle-hashing the mapped function -- which closes over the vectorizer,
        i.e. the whole torch model. That is a flat ~21s per create() call,
        independent of dataset size (measured: 21s for 250 images and for 2000).

        The fingerprint names the on-disk arrow cache file that `map` will reuse,
        so it must be BOTH deterministic (or nothing is ever reused) AND
        sensitive to every input that changes the output (or a stale result is
        served silently). The inputs are: the source data (`dataset._fingerprint`),
        the transformation (`vectorizer.fingerprint_fields()`, which subclasses
        extend with their fitted state), the output schema, and the map kwargs.
        Batch sizes are included because they can affect float accumulation.
        """
        from vectormesh import __version__

        dataset_fingerprint = getattr(dataset, "_fingerprint", None)
        if not dataset_fingerprint:
            # No identity for the source data means we cannot prove a cached
            # result belongs to *this* dataset. Fall back to a unique value:
            # recomputing is wasteful, serving another dataset's vectors is not
            # recoverable.
            dataset_fingerprint = f"unknown-{uuid.uuid4().hex}"
            logger.warning(
                "Dataset has no _fingerprint; using a random one, so this "
                "map() result cannot be reused by a later call."
            )

        payload = {
            "vectormesh_version": __version__,
            "dataset": dataset_fingerprint,
            "vectorizer": vectorizer.fingerprint_fields(),
            "features": features.to_dict(),
            "vector_batch": vector_batch,
            "map_batch": map_batch,
            "remove_columns": list(remove_columns or []),
        }
        # No `default=`: a non-serialisable field must fail loudly here rather
        # than silently hash to an object repr (which contains a memory address
        # and would make the fingerprint differ every run).
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:32]

    @classmethod
    def _build_metadata(
        cls,
        vectorizer,
        column_name: str,
        features: Features,
        num_observations: int,
        dataset: Optional[Dataset] = None,
    ) -> dict:
        """Assemble the metadata dict describing this vectorizer's output column.

        `chunk_sizes` is derived from `dataset[column_name]`'s own row lengths when a
        rank-2 (chunked) dataset is available, rather than read off
        `vectorizer.chunk_sizes`. The vectorizer's own counter is a side effect of the
        mapped function actually running -- `dataset.map()` may instead serve an
        on-disk result for an identical (dataset, vectorizer) fingerprint (the whole
        point of `new_fingerprint` above: skip re-embedding, not just re-hashing), and
        a skipped call means a skipped side effect. The rows themselves are correct
        either way, so counting them directly is correct either way too.
        """
        from vectormesh import __version__

        tensordtype = cls.get_dtensor(vectorizer)
        chunk_sizes = getattr(vectorizer, "chunk_sizes", None)
        if dataset is not None and tensordtype == 2:
            chunk_sizes = Counter(len(row) for row in dataset[column_name])

        return {
            column_name: {
                "vectormesh_version": __version__,
                "model_tag": vectorizer.model_name,
                "vectorizer_type": vectorizer.__class__.__name__,
                "tensordtype": tensordtype,
                "hidden_size": vectorizer.get_hidden_size,
                "context_size": vectorizer.get_context_size,
                "stride": getattr(vectorizer, "get_stride", None),
                "offsets_supported": getattr(vectorizer, "get_offsets_supported", None),
                "chunk_sizes": chunk_sizes,
            },
            "features": list(features.keys()),
            "created_at": datetime.now().isoformat(),
            "num_observations": num_observations,
        }

    @staticmethod
    def _write(new_dataset: Dataset, filepath: Path, metadata: dict) -> None:
        """Persist the vectorized dataset and its metadata.json to disk."""
        new_dataset.save_to_disk(filepath)
        with open(filepath / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.success("Vectorization complete.")

    @classmethod
    def load(cls, path: Path) -> "VectorCache[TVectorizer]":
        if not path.exists():
            raise VectorMeshError(f"Cache path {path} does not exist.")
        if not path.is_dir():
            raise VectorMeshError(f"Cache path {path} is expected to be a directory.")

        loaded_data = load_from_disk(path)
        if isinstance(loaded_data, DatasetDict):
            raise VectorMeshError(
                f"Expected Dataset but got DatasetDict at {path}. "
                "Please load a specific split instead."
            )
        dataset: Dataset = loaded_data
        dataset.set_format(type="torch")
        metadata_path = path / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        logger.success(f"Cache loaded from {path}")
        return cls(
            name=path.stem,
            cache_dir=path.parent.resolve(),
            dataset=dataset,
            metadata=metadata,
        )

    @staticmethod
    def update_metadata(path: Path, new_metadata: dict) -> dict:
        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            logger.info(f"No existing metadata found at {path}, creating new metadata.")
            return new_metadata
        logger.info(f"Updating existing metadata found at {path}.")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        metadata.update(new_metadata)
        return metadata

    @staticmethod
    def get_features(dataset: Dataset, tensord: int, embedding_column: str) -> Features:
        """Extract the embedding feature creation logic"""
        features = dataset.features.copy()
        if tensord == 2:
            embedding_feature = Sequence(Sequence(Value("float32")))  # (chunks, dim)
        elif tensord == 1:
            embedding_feature = Sequence(Value("float32"))  # (dim,)
        else:
            raise ValueError(f"Unsupported tensor dtype with {tensord} dimensions.")
        features[embedding_column] = embedding_feature
        return features

    @staticmethod
    def get_dtensor(vectorizer) -> int:
        hints = get_type_hints(vectorizer.__call__)
        return_type = hints[
            "return"
        ]  # dict[str, list[jaxtyping.Float[Tensor, '_ dim']]]
        key, values = get_args(
            return_type
        )  # (str, list[jaxtyping.Float[Tensor, '_ dim']])
        valua_args = get_args(values)  # (jaxtyping.Float[Tensor, '_ dim'],)
        tensor_type = valua_args[0]  # jaxtyping.Float[Tensor, '_ dim']
        return len(tensor_type.dim_str.split())  # '_ dim'.split() -> 2

    def _ensure_dataset_loaded(self) -> Dataset:
        """Ensure dataset is loaded, raise error if not."""
        if self.dataset is None:
            raise VectorMeshError("Dataset not loaded. Call create() or load() first.")
        return self.dataset

    def __len__(self) -> int:
        return len(self._ensure_dataset_loaded())

    def __getitem__(self, key):
        return self._ensure_dataset_loaded()[key]

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        attribute = getattr(self._ensure_dataset_loaded(), name)
        return attribute

    def __iter__(self):
        return iter(self._ensure_dataset_loaded())
