import json
from datetime import datetime
from pathlib import Path
from typing import Generic, Optional, TypeVar, get_args, get_type_hints

from datasets import Dataset, Features, Sequence, Value, load_from_disk
from loguru import logger

from vectormesh.types import VectorMeshComponent, VectorMeshError

from .vectorizers import BaseVectorizer

TVectorizer = TypeVar("TVectorizer", bound=BaseVectorizer)


class VectorCache(VectorMeshComponent, Generic[TVectorizer]):
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
        dataset_tag: Optional[str] = "default",
        features: Optional[Features] = None,
        vector_batch: Optional[int] = 32,
        map_batch: Optional[int] = 32,
    ) -> "VectorCache[TVectorizer]":
        from vectormesh import __version__

        vtype = vectorizer.__class__.__name__

        tensord = cls.get_dtensor(vectorizer)
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created cache directory at {cache_dir}")
        if features is None:
            features = cls.get_features(vectorizer, tensord)
        cachetag = f"{dataset_tag}_{tensord}d_{vectorizer.get_hidden_size}"
        filepath = cache_dir / cachetag
        metadata_path = filepath / "metadata.json"
        logger.info(
            f"Starting cache with\n tag {cachetag}\n at {filepath}\n features {features}"
        )

        try:
            new_dataset = dataset.map(
                lambda batch: vectorizer(batch["text"], batchsize=vector_batch),
                batched=True,
                batch_size=map_batch,  # Number of documents per batch
                features=features,
            )
            logger.success("Vectorization complete.")

            new_dataset.save_to_disk(filepath)
            metadata = {
                "vectormesh_version": __version__,
                "model_tag": vectorizer.model_name,
                "vectorizer_type": vtype,
                "tensordtype": cls.get_dtensor(vectorizer),
                "hidden_size": vectorizer.get_hidden_size,
                "context_size": vectorizer.get_context_size,
                "chunk_sizes": dict(vectorizer.chunk_sizes),
                "created_at": datetime.now().isoformat(),
                "num_observations": len(new_dataset),
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

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
    def load(cls, path: Path) -> "Cache[TVectorizer]":
        metadata_path = path / "metadata.json"
        if not path.exists():
            raise VectorMeshError(f"Directory {path} does not exist.")
        if not metadata_path.exists():
            raise VectorMeshError(f"Metadata file {metadata_path} does not exist.")

        dataset = load_from_disk(path)
        dataset.set_format(type="torch")
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
    def get_features(vectorizer: TVectorizer, tensord: int) -> Features:
        if tensord == 2:
            embedding_feature = Sequence(Sequence(Value("float32")))  # (chunks, 768)
        elif tensord == 1:
            embedding_feature = Sequence(Value("float32"))  # (768,)
        else:
            raise ValueError(f"Unsupported tensor dtype with {tensord} dimensions.")

        features = Features(
            {
                "text": Value("string"),
                "target": Sequence(Value("int32")),
                "embedding": embedding_feature,
            }
        )
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

    def __iter__(self):
        return iter(self._ensure_dataset_loaded())

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(self._ensure_dataset_loaded(), name)
