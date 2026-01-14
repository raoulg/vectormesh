import json
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
        vector_batch: Optional[int] = 32,
        map_batch: Optional[int] = 32,
        column_name: Optional[str] = None,
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

        Returns:
            VectorCache[TVectorizer]
        """
        from vectormesh import __version__

        vtype = vectorizer.__class__.__name__
        if column_name is None:
            if not vectorizer.col_name:
                raise VectorMeshError(
                    "column_name must be provided if vectorizer.col_name is not set."
                )
            column_name = vectorizer.col_name
        logger.info(f"Using embedding column: {column_name}")

        tensord = cls.get_dtensor(vectorizer)

        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created cache directory at {cache_dir}")

        if features is None:
            features = cls.get_features(dataset, tensord, embedding_column=column_name)

        now = datetime.now().strftime("%Y%m%d%H%M%S")
        cachetag = f"{now}_{dataset_tag}_{column_name}"
        filepath = cache_dir / cachetag
        metadata_path = filepath / "metadata.json"
        logger.info(f"Starting {cachetag}")

        try:
            new_dataset = dataset.map(
                lambda batch: vectorizer(batch["text"], batchsize=vector_batch),
                batched=True,
                batch_size=map_batch,  # Number of documents per batch
                features=features,
            )

            new_dataset.save_to_disk(filepath)
            metadata = {
                f"{column_name}": {
                    "vectormesh_version": __version__,
                    "model_tag": vectorizer.model_name,
                    "vectorizer_type": vtype,
                    "tensordtype": cls.get_dtensor(vectorizer),
                    "hidden_size": vectorizer.get_hidden_size,
                    "context_size": vectorizer.get_context_size,
                    "chunk_sizes": getattr(vectorizer, "chunk_sizes", None),
                },
                "features": list(features.keys()),
                "created_at": datetime.now().isoformat(),
                "num_observations": len(new_dataset),
            }
            # check for existing metadata to update
            metadata = cls.update_metadata(cache_dir / dataset_tag, metadata)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.success("Vectorization complete.")

        except Exception as e:
            if filepath.exists():
                filepath.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            raise VectorMeshError(f"Failed to create cache at {filepath}") from e

        new_dataset.set_format(type="torch")
        logger.success(f"Cache saved to {filepath}")
        return cls(
            name=cachetag,
            cache_dir=cache_dir,
            dataset=new_dataset,
            metadata=metadata,
        )

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
