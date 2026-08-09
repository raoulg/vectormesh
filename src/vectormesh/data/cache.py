import hashlib
import json
import shutil
import tempfile
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_type_hints,
)

import datasets
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk
from loguru import logger
from torch.utils.data import Dataset as TorchDataset

from vectormesh.types import Cachable, VectorMeshError

from .hub import (
    META_NON_COLUMN_KEYS,
    HubUpload,
    render_card,
    stage_payload,
    vector_columns,
)
from .vectorizers import BaseVectorizer

TVectorizer = TypeVar("TVectorizer", bound=BaseVectorizer)

__all__ = ["VectorCache", "META_NON_COLUMN_KEYS", "HubUpload"]


class VectorCache(Cachable, Generic[TVectorizer], TorchDataset[Any]):
    """A `dataset`-less or `metadata`-less instance is not a usable cache, so both
    are required rather than Optional -- `create()` and `load()` are the only two
    constructors and both always supply real values for both fields."""

    name: str
    cache_dir: Path
    dataset: Dataset
    metadata: dict

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

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        split: str = "train",
        revision: Optional[str] = None,
        token: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> "VectorCache[TVectorizer]":
        """Download a published cache from the HuggingFace hub and load one split.

        `load()` reads a local directory; this is the hub counterpart, so that consuming
        a cache someone else published is one call rather than a `snapshot_download`
        incantation copied between notebooks.

            train = VectorCache.from_hub("pttrn-io/eurosat-dinov2-small")
            test = VectorCache.from_hub("pttrn-io/eurosat-dinov2-small", split="test")

        Only the requested split is downloaded. These are `save_to_disk` layouts, not
        parquet, so `datasets.load_dataset()` will not read them and this will.

        Args:
            repo_id: hub dataset id, e.g. "pttrn-io/dtd-dinov2-small"
            split: which split folder to fetch. Defaults to "train".
            revision: pin a specific commit/tag. Recommended for anything reproducible.
            token: for private repos; falls back to the ambient HF login.
            cache_dir: override the local huggingface download directory.
        """
        from huggingface_hub import snapshot_download

        try:
            local = Path(
                snapshot_download(
                    repo_id,
                    repo_type="dataset",
                    revision=revision,
                    token=token,
                    cache_dir=str(cache_dir) if cache_dir is not None else None,
                    allow_patterns=f"{split}/*",
                )
            )
        except Exception as e:
            raise VectorMeshError(
                f"Could not download {repo_id!r} from the hub. If the repo is private, "
                "pass token= or log in with `huggingface-cli login`."
            ) from e

        path = local / split
        if not path.exists():
            raise VectorMeshError(
                f"{repo_id!r} has no split {split!r}. Available: "
                f"{cls._hub_splits(repo_id, revision=revision, token=token)}"
            )
        return cls.load(path)

    def hub_repo_id(self, org: str, dataset: str) -> str:
        """The repo id this cache belongs in: `{org}/{dataset}-{encoder}`.

        Derived rather than typed, and derived from `metadata.json`'s `model_tag` rather
        than from the folder name, so the repo id and the metadata cannot disagree about
        which checkpoint produced the vectors:

            cache.hub_repo_id("pttrn-io", "eurosat")   # pttrn-io/eurosat-dinov2-small

        The encoder tag is mirrored unabbreviated -- everything after the `/` in the
        model id. One repo per (dataset x encoder): a repo named for the dataset alone
        collides with itself the moment a second encoder is tried, and nobody downstream
        can tell which vectors they got.

        Raises:
            VectorMeshError: when this cache has no single encoder to name a repo after,
                i.e. after a `join()`. Pass the repo id to `push_to_hub` yourself.
        """
        columns = self.vector_columns
        if len(columns) != 1:
            raise VectorMeshError(
                f"cannot derive a repo id: this cache describes {columns}, so there is "
                "no single encoder to name it after. Pass repo_id to push_to_hub "
                "explicitly -- likely needed after a join()."
            )
        tag = self.metadata[columns[0]].get("model_tag")
        if not tag:
            raise VectorMeshError(
                f"metadata for {columns[0]!r} has no model_tag to derive a repo id from."
            )
        return f"{org}/{dataset}-{str(tag).split('/')[-1]}"

    def push_to_hub(
        self,
        repo_id: str,
        split: str = "train",
        *,
        source_dataset: Optional[str] = None,
        drop_columns: Optional[list[str]] = None,
        card: Optional[str] = None,
        write_card: bool = True,
        license: str = "other",
        private: bool = False,
        token: Optional[str] = None,
        dry_run: bool = False,
    ) -> HubUpload:
        """Publish this cache to the HuggingFace hub as one split of a dataset repo.

        The inverse of `from_hub`: what this uploads is what that downloads -- the vector
        columns, the labels, `source_idx` and `metadata.json`, in the `save_to_disk`
        layout, under a folder named for the split.

            cache.push_to_hub("pttrn-io/eurosat-dinov2-small", split="train")
            testcache.push_to_hub("pttrn-io/eurosat-dinov2-small", split="test")

        Call it once per split; splits are separate folders in one repo, and each call
        rewrites the dataset card to cover every split the repo has by then.

        What goes up is re-serialised from this cache's dataset into a temporary folder,
        not copied from `cache_dir`: `datasets` leaves its own `cache-<hash>.arrow` files
        next to a dataset it has mapped over, and uploading the folder as it sits means
        uploading those too. The returned `HubUpload` lists every file and its total
        size -- check it before a first publish, or pass `dry_run=True` to see it without
        touching the hub.

        This is not `Dataset.push_to_hub`, which writes parquet: browsable in the hub
        viewer, but not a layout `from_hub` can read, and it leaves `metadata.json`
        behind. Reach through to `cache.dataset.push_to_hub(...)` if that is what you
        want, and expect to upload the metadata yourself.

        Args:
            repo_id: hub dataset id. `hub_repo_id()` derives the conventional one.
            split: folder to publish into, and the name `from_hub(split=...)` will use.
            source_dataset: hub id of the dataset the vectors were built from. Recorded
                in the card, pinned to the revision it resolves to today.
            drop_columns: columns to leave out of the upload, e.g. `["image"]`. Raw
                pixels are ~100x their embeddings; text usually earns its place.
            card: dataset card to upload verbatim. Defaults to one generated from the
                metadata of every split in the repo.
            write_card: set False to leave the repo's README.md alone, e.g. when it has
                been edited by hand since the first push.
            license: the `license` field of the generated card's frontmatter. A cache
                inherits the terms of the data it was built from.
            private: create the repo private. Ignored if the repo already exists.
            token: write token; falls back to the ambient HF login.
            dry_run: render the card and stage the payload, upload nothing.

        Raises:
            VectorMeshError: on an empty cache, a `drop_columns` name that is not a
                column, dropping the last vector column, or a staged payload holding
                anything but arrow and json.
        """
        dataset = self._ensure_dataset_loaded().with_format(None)
        if len(dataset) == 0:
            raise VectorMeshError("refusing to publish an empty cache")

        for column in drop_columns or []:
            if column not in dataset.column_names:
                raise VectorMeshError(
                    f"cannot drop {column!r}: this cache has {dataset.column_names}"
                )
        if drop_columns:
            dataset = dataset.remove_columns(drop_columns)

        metadata = {
            key: value
            for key, value in self.metadata.items()
            if key in dataset.column_names or key in META_NON_COLUMN_KEYS
        }
        metadata["features"] = dataset.column_names
        if not vector_columns(metadata):
            raise VectorMeshError(
                "nothing left to publish: every column described by metadata.json was "
                f"dropped. Kept columns: {dataset.column_names}"
            )
        self._warn_about_payload(dataset)

        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp) / split
            files = stage_payload(dataset, metadata, staged)
            nbytes = sum(f.stat().st_size for f in files)
            logger.info(
                f"{repo_id} [{split}]: {len(dataset)} rows, {nbytes / 1e6:.1f} MB in "
                f"{[f.name for f in files]}"
            )

            if not dry_run:
                self._upload(repo_id, split, staged, private, token)

            # Rendering resolves revisions over the network, so only pay for it when the
            # card is going to be read: uploaded, or shown by a dry run.
            if card is None and (write_card or dry_run):
                card = render_card(
                    repo_id,
                    self._card_splits(repo_id, split, metadata, token=token),
                    source_dataset=source_dataset,
                    license=license,
                    token=token,
                )
            card = card or ""
            if write_card and not dry_run:
                self._upload_card(repo_id, card, token)

            upload = HubUpload(
                repo_id=repo_id,
                split=split,
                files=[f.name for f in files],
                nbytes=nbytes,
                card=card,
                url=f"https://huggingface.co/datasets/{repo_id}",
                uploaded=not dry_run,
            )

        if dry_run:
            logger.info(f"dry run: nothing uploaded to {upload.url}")
        else:
            logger.success(f"published {split} to {upload.url}")
        return upload

    def _warn_about_payload(self, dataset: Dataset) -> None:
        """Flag the two things that make a published cache hard to use."""
        heavy = [
            name
            for name, feature in dataset.features.items()
            if type(feature).__name__ in {"Image", "Audio", "Video"}
        ]
        if heavy:
            logger.warning(
                f"{heavy} carry raw media and are ~100x the size of their embeddings. "
                "Pass drop_columns= unless whoever downloads this needs them."
            )
        if "source_idx" not in dataset.column_names:
            logger.warning(
                "no 'source_idx' column: nobody downloading this can get from a vector "
                "back to the row it was built from. Add it before vectorizing, with "
                "dataset.add_column('source_idx', range(len(dataset))) *before* any "
                "shuffle or select."
            )

    @staticmethod
    def _upload(
        repo_id: str, split: str, staged: Path, private: bool, token: Optional[str]
    ) -> None:
        """Create the repo if needed and upload the staged split folder."""
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        try:
            api.create_repo(
                repo_id, repo_type="dataset", private=private, exist_ok=True
            )
            api.upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(staged),
                path_in_repo=split,
                allow_patterns=["*.arrow", "*.json"],
                commit_message=f"{split} split",
            )
        except Exception as e:
            raise VectorMeshError(
                f"Could not upload to {repo_id!r}. Publishing needs a write token: "
                "pass token= or log in with `hf auth login`."
            ) from e

    @staticmethod
    def _upload_card(repo_id: str, card: str, token: Optional[str]) -> None:
        """Write README.md. A failed card is not a failed publish."""
        from huggingface_hub import HfApi

        try:
            HfApi(token=token).upload_file(
                path_or_fileobj=card.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="dataset card",
            )
        except Exception as e:  # pragma: no cover - network path
            logger.warning(f"data uploaded, but the dataset card failed: {e}")

    @classmethod
    def _card_splits(
        cls, repo_id: str, split: str, metadata: dict, token: Optional[str]
    ) -> dict[str, dict]:
        """Metadata per split for the card: this one, plus any already in the repo.

        The card quotes row counts per split, so pushing `test` into a repo that already
        holds `train` must not rewrite the card as if `train` were gone. The other splits
        are read back from their own `metadata.json` -- a few KB each, and the only place
        that knows what they hold.
        """
        splits = {split: metadata}
        for other in cls._hub_splits(repo_id, token=token):
            if other == split:
                continue
            try:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(
                    repo_id,
                    f"{other}/metadata.json",
                    repo_type="dataset",
                    token=token,
                )
                with open(path) as f:
                    splits[other] = json.load(f)
            except Exception as e:  # pragma: no cover - network path
                logger.warning(
                    f"could not read {other}/metadata.json for the card: {e}"
                )

        order = ["train", "validation", "valid", "dev", "test"]
        return dict(
            sorted(
                splits.items(),
                key=lambda kv: (
                    order.index(kv[0]) if kv[0] in order else len(order),
                    kv[0],
                ),
            )
        )

    @classmethod
    def get_or_create(
        cls,
        cache_dir: Path,
        dataset_tag: str,
        vectorizer: TVectorizer,
        dataset: "Union[Dataset, Callable[[], Dataset]]",
        remove_columns: Optional[list[str]] = None,
    ) -> "VectorCache[TVectorizer]":
        """Load the cache for (dataset_tag, this vectorizer's encoder) if one is on
        disk; build it otherwise.

        A notebook that builds its own cache -- rather than downloading a published
        one via `from_hub` -- should not re-encode on every rerun, and should not
        silently serve a *different* encoder's vectors just because they happen to
        share a `dataset_tag`. This does both: the folder name embeds the encoder
        identity, and a hit whose stored `model_tag` disagrees with the vectorizer
        handed in raises rather than being served.

        `dataset` may be a ready `Dataset` or a zero-argument callable that builds one
        lazily -- a cache hit then costs nothing beyond a glob, which matters when
        building the dataset itself is not free (e.g. a fresh HF `load_dataset` plus a
        `.select()`).

        Also disables `datasets`' own `.map()` file-cache for the process (see
        `_vectorize`'s fingerprint): that cache is keyed independently of `cache_dir`
        or `dataset_tag`, so a second `create()` over a dataset+vectorizer pair this
        process has already mapped can silently skip the vectorizer's side effects
        (`chunk_sizes`) even under a fresh tag. `VectorCache`'s own folder is this
        course's artefact of record, so disabling the redundant layer costs nothing.

        Raises:
            VectorMeshError: on a cache hit whose stored `model_tag` disagrees with
                `vectorizer.model_name` -- a folder from before this convention, or one
                renamed by hand.
        """
        datasets.disable_caching()

        column = vectorizer.col_name
        tag = f"{dataset_tag}_{cls._model_slug(vectorizer)}"
        hits = sorted(cache_dir.glob(f"*_{tag}_{column}"))
        if hits:
            cache = cls.load(hits[-1])
            stored = (cache.metadata or {}).get(column, {}).get("model_tag")
            wanted = getattr(vectorizer, "model_name", None)
            if stored and wanted and stored != wanted:
                raise VectorMeshError(
                    f"{hits[-1]} was built with model_tag={stored!r}, but the "
                    f"vectorizer handed to get_or_create has model_name={wanted!r}. "
                    "Delete the stale cache or pick a new dataset_tag rather than "
                    "silently reusing someone else's encoding."
                )
            logger.info(f"found cache: {hits[-1].name}")
            return cache

        logger.info(f"no cache for {tag}_{column}, building")
        built = dataset if isinstance(dataset, Dataset) else dataset()
        return cls.create(
            cache_dir=cache_dir,
            vectorizer=vectorizer,
            dataset=built,
            dataset_tag=tag,
            remove_columns=remove_columns,
        )

    @staticmethod
    def _model_slug(vectorizer) -> str:
        """Filesystem-safe encoder identity, e.g. "granite-embedding-small-english-r2".

        `vectorizer.model_name` is the swappable HF id and is what must disambiguate a
        cache. `ChunkedRegexVectorizer.model_name` defaults to a fixed constant
        ("chunked_regex_vectorizer") rather than a swappable checkpoint -- correct
        here, since a regex feature *design* is not an interchangeable "model" the way
        a pretrained encoder is.
        """
        name = getattr(vectorizer, "model_name", None) or vectorizer.__class__.__name__
        return name.split("/")[-1]

    @staticmethod
    def _hub_splits(
        repo_id: str, revision: Optional[str] = None, token: Optional[str] = None
    ) -> list[str]:
        """Split folders present in a hub cache repo. Error paths only."""
        try:
            from huggingface_hub import HfApi

            files = HfApi().list_repo_files(
                repo_id, repo_type="dataset", revision=revision, token=token
            )
            return sorted({f.split("/")[0] for f in files if f.endswith("/state.json")})
        except Exception:  # pragma: no cover - best-effort hint
            return []

    @property
    def vector_columns(self) -> list[str]:
        """Columns described by metadata.json, i.e. the ones a vectorizer produced."""
        return vector_columns(self.metadata)

    @property
    def num_classes(self) -> int:
        """Shortcut for `cache.features["label"].num_classes` -- a head's output width.

        Every notebook that trains a classifier re-derives this by hand today. It is a
        property, not a method, because it costs nothing: `features` is already resident
        once a cache is loaded.
        """
        features = self._ensure_dataset_loaded().features
        if "label" not in features:
            raise VectorMeshError(
                f"no 'label' column in this cache (has: {list(features)}); "
                ".num_classes only makes sense for a labelled classification cache."
            )
        label_feature = features["label"]
        if not hasattr(label_feature, "num_classes"):
            raise VectorMeshError(
                f"'label' is a {type(label_feature).__name__}, not a ClassLabel, so it "
                "has no num_classes. Read cache.features['label'] directly instead."
            )
        return label_feature.num_classes

    @property
    def dim(self) -> int:
        """Shortcut for the hidden_size of this cache's one vector column.

        Raises the same way `join()`'s column resolution does once there is more than
        one vector column (i.e. after a `join()`) -- at that point "the" width is
        ambiguous and the caller must say which column they mean via
        `cache.metadata[<col>]["hidden_size"]`.
        """
        columns = self.vector_columns
        if len(columns) != 1:
            raise VectorMeshError(
                f"cannot guess which column's width you mean: this cache describes "
                f"{columns}. Read metadata[<col>]['hidden_size'] explicitly -- likely "
                "needed after a join()."
            )
        return self.metadata[columns[0]]["hidden_size"]

    def join(
        self,
        other: "VectorCache",
        on: str = "source_idx",
        column: Optional[str] = None,
        into: Optional[str] = None,
    ) -> "VectorCache[TVectorizer]":
        """Bring another cache's vector column onto this one, aligned row-for-row on `on`.

        Two caches built by different encoders over the same source rows are the raw
        material for a parallel/fusion pipeline, but they arrive as two separate
        downloads. This aligns them properly instead of trusting that the row orders
        happen to match:

            dino = VectorCache.from_hub("pttrn-io/dtd-dinov2-small")
            rn18 = VectorCache.from_hub("pttrn-io/dtd-resnet-18")
            both = dino.join(rn18)             # -> columns: label, source_idx, embed, embed_resnet-18

            CollateParallel(vec1_col="embed", vec2_col="embed_resnet-18", ...)

        Args:
            other: the cache to join in.
            on: key column present in both. "source_idx" is written by the course's
                build scripts precisely so this is possible.
            column: which column of `other` to take. Defaults to its single vector
                column, and raises if there is more than one to choose from.
            into: name for the joined column. Defaults to `column`, suffixed with
                `other`'s encoder slug when that would collide with an existing name.

        Raises:
            VectorMeshError: on a missing key column, duplicate keys, a key-set
                mismatch, or -- the one that catches a genuinely wrong join -- labels
                that disagree once the rows have been aligned.
        """
        left, right = self.dataset.with_format(None), other.dataset.with_format(None)

        for name, ds in (("this cache", left), ("the other cache", right)):
            if on not in ds.column_names:
                raise VectorMeshError(
                    f"{name} has no column {on!r} to join on (has: {ds.column_names}). "
                    "Caches built before source_idx existed cannot be aligned safely."
                )

        column = other._resolve_join_column(column)
        into = into or (
            column if column not in left.column_names else f"{column}_{other._slug()}"
        )
        if into in left.column_names:
            raise VectorMeshError(
                f"column {into!r} already exists in this cache; pass into= to name it."
            )

        lkeys, rkeys = list(left[on]), list(right[on])
        if len(set(rkeys)) != len(rkeys):
            raise VectorMeshError(
                f"the other cache has duplicate {on!r} values, so the join is ambiguous."
            )
        if set(lkeys) != set(rkeys):
            missing = len(set(lkeys) - set(rkeys))
            raise VectorMeshError(
                f"the two caches do not cover the same rows: {len(lkeys)} vs "
                f"{len(rkeys)} rows, {missing} of this cache's {on!r} values missing "
                "from the other. Were they built from different splits or subsamples?"
            )

        position = {key: i for i, key in enumerate(rkeys)}
        aligned = right.select([position[key] for key in lkeys])

        # The check worth having: if both sides carry labels and they disagree after
        # alignment, the join is wrong in a way that would otherwise train quietly.
        for shared in ("label", "labels"):
            if shared in left.column_names and shared in aligned.column_names:
                if list(left[shared]) != list(aligned[shared]):
                    raise VectorMeshError(
                        f"{shared!r} disagrees between the two caches after aligning on "
                        f"{on!r}. They are not describing the same rows -- check that "
                        "both were built from the same source dataset revision."
                    )

        joined = left.add_column(into, list(aligned[column]))
        joined.set_format(type="torch")

        metadata = dict(self.metadata)
        metadata[into] = dict(other.metadata.get(column, {}))
        metadata["features"] = joined.column_names
        logger.success(f"joined {into!r} ({len(joined)} rows) on {on!r}")

        return type(self)(
            name=f"{self.name}+{into}",
            cache_dir=self.cache_dir,
            dataset=joined,
            metadata=metadata,
        )

    def _resolve_join_column(self, column: Optional[str]) -> str:
        """Which column of this cache a join should take."""
        if column is not None:
            if column not in self.dataset.column_names:
                raise VectorMeshError(
                    f"column {column!r} not in {self.dataset.column_names}"
                )
            return column
        candidates = self.vector_columns
        if len(candidates) != 1:
            raise VectorMeshError(
                f"cannot guess which column to join: this cache describes {candidates}. "
                "Pass column= explicitly."
            )
        return candidates[0]

    def _slug(self) -> str:
        """Short encoder identity for naming a joined column, e.g. "resnet-18"."""
        for col in self.vector_columns:
            tag = self.metadata.get(col, {}).get("model_tag")
            if tag:
                return str(tag).split("/")[-1]
        return self.name

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

    def __getitem__(self, index):
        return self._ensure_dataset_loaded()[index]

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        attribute = getattr(self._ensure_dataset_loaded(), name)
        return attribute

    def __iter__(self):
        return iter(self._ensure_dataset_loaded())
