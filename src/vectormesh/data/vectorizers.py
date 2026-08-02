"""Text vectorization components using HuggingFace models."""

import json
import re
from math import isqrt
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Callable, ClassVar, Optional

import matplotlib.pyplot as plt
import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from loguru import logger
from pydantic import Field, PrivateAttr, model_validator
from torch import Tensor
from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoTokenizer

from vectormesh.types import Cachable


def _json_safe(value: Any) -> Any:
    """Return `value` if it can go into a cache fingerprint, else a type marker.

    A marker is a deliberate dead end: it identifies *that* the field exists
    without pretending to identify its content, so a subclass that owns such a
    field has to describe it itself (see `BaseVectorizer.fingerprint_fields`).
    Falling back to `repr` instead would embed a memory address and make the
    fingerprint differ on every run.
    """
    try:
        json.dumps(value)
    except TypeError:
        return f"<unhashable:{type(value).__name__}>"
    return value


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class BaseVectorizer(ABC, Cachable):
    """
    Base class for all vectorizers.

    All vectorizers must:
    - Have a model_name, device, and metadata
    - model_name is used in the VectorCache to identify the model used for a vectorizer
    - col_name is used to store the output of the vectorizer in the dataset
    - input_col is the dataset column the vectorizer reads from (e.g. "text", "image")
    - device is for hardware acceleration, if applicable
    - Implement __call__ that returns dict[str, list[Float[Tensor, "..."]]]
    - The exact tensor dimensionality can vary by implementation
    """

    model_name: str
    col_name: str
    input_col: str = "text"
    device: str = Field(default_factory=detect_device)

    # Fields excluded from the cache fingerprint because they change while
    # vectorizing rather than describing what the vectorizer does.
    FINGERPRINT_EXCLUDE: ClassVar[frozenset[str]] = frozenset()

    _metadata: Any = PrivateAttr()
    _effective_max_length: Optional[int] = PrivateAttr(default=None)

    @abstractmethod
    @model_validator(mode="after")
    def initialize_model(self):
        """
        Initialize the model/API connection.
        Must set self._metadata with at least:
        - hidden_size or dim: output dimension
        Must set self._effective_max_length to the actual context limit used,
        or None if the concept does not apply (e.g. RegexVectorizer).
        """
        pass

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def __call__(
        self, inputs: list, batchsize: int
    ) -> dict[str, list[Float[Tensor, "..."]]]:
        """
        Process a batch of inputs and return embeddings.

        Args:
            inputs: List of input items (texts, images, ...) read from input_col
            batchsize: Batch size for processing

        Returns:
            Dict with '{self.col_name : list[Tensor]}'.
            Tensor dimensionality varies by implementation
        """
        pass

    @property
    def get_metadata(self) -> dict:
        """
        Return metadata about the model.
        Subclasses can override to add more fields.
        """
        return {
            "model_name": self.model_name,
            "col_name": self.col_name,
            "hidden_size": getattr(self._metadata, "hidden_size"),
            "context_size": self._effective_max_length,
        }

    def fingerprint_fields(self) -> dict:
        """Everything that identifies the transformation this vectorizer applies.

        The VectorCache hashes this dict to fingerprint its `dataset.map` call
        instead of letting `datasets` pickle-hash the vectorizer itself (which,
        for a torch model, costs ~21s per call regardless of dataset size).

        Every public pydantic field is included by default, so a field added
        later cannot silently go unfingerprinted -- the failure mode of that
        default is an unnecessary recompute, never a stale cache. Two kinds of
        field need explicit handling:

        - Fields that are not JSON-serialisable (a `Callable`, say) collapse to a
          type marker. A subclass owning such a field must override this method
          and add something that describes it deterministically.
        - Fields mutated *while vectorizing* (counters, statistics) must be named
          in `FINGERPRINT_EXCLUDE`, or the fingerprint stops being reproducible.

        Values must also be deterministic across processes: no object reprs, no
        memory addresses, no unordered sets.

        Model *weights* are identified by `model_name` only; a locally mutated
        or fine-tuned model reusing the same tag is not distinguishable here.
        """
        fields = {
            name: _json_safe(getattr(self, name))
            for name in type(self).model_fields
            if name not in self.FINGERPRINT_EXCLUDE
        }
        fields.update(
            {
                "class": type(self).__name__,
                "hidden_size": self.get_hidden_size,
                "context_size": self.get_context_size,
            }
        )
        return fields

    @property
    def get_hidden_size(self) -> int:
        return getattr(self._metadata, "hidden_size")

    @property
    def get_context_size(self) -> Optional[int]:
        return self._effective_max_length


class Vectorizer(BaseVectorizer):
    model_name: str
    col_name: str
    device: str = Field(default_factory=detect_device)
    max_length: Optional[int] = None

    _metadata: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _stride: int = PrivateAttr()
    _effective_max_length: int = PrivateAttr()
    chunk_sizes: Counter = Counter()

    # Stride (chunk overlap) is a fixed fraction of the context window. Kept as a
    # named constant -- and persisted in the cache metadata via `get_stride` -- so a
    # ChunkedRegexVectorizer can reproduce the exact same chunk boundaries later.
    STRIDE_DIVISOR: ClassVar[int] = 10

    # chunk_sizes is a running tally filled in by extend() during vectorization,
    # so it says how often this vectorizer has been used, not what it computes.
    FINGERPRINT_EXCLUDE: ClassVar[frozenset[str]] = frozenset({"chunk_sizes"})

    @model_validator(mode="after")
    def initialize_model(self):
        self._metadata = AutoConfig.from_pretrained(self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

        max_pos = getattr(self._metadata, "max_position_embeddings")
        self._effective_max_length = (
            min(max_pos, self.max_length) if self.max_length else max_pos
        )
        self._stride = self._effective_max_length // self.STRIDE_DIVISOR

        logger.info(f"Using device: {self.device}")
        logger.info(
            f"using max_length: {self._effective_max_length} (model max: {max_pos}), stride: {self._stride}"
        )
        return self

    @jaxtyped(typechecker=beartype)
    def tokenize(
        self, text: list[str]
    ) -> tuple[
        Int[Tensor, "batch tokens"],
        Int[Tensor, "batch tokens"],
        Int[Tensor, "batch"],
    ]:
        """
        Receives a batch of texts Σ* , where Σ is an alphabet, and * represents that the strings
        can be concatenated in any order to create a sequence to create sentences.

        The output is:
        - input_ids: A 2D tensor of token ids (batch_size * chunks, max_length) ∈ ℕ
        - attention_mask: A 2D tensor of attention mask (batch_size * chunks, max_length) ∈ {0, 1}
        - overflow_to_sample_mapping: A 1D tensor of document indices (batch_size * chunks,) ∈ ℕ  (eg 0, 0 ,0, 1, 1, 2, ...)

        Because of the context window (eg 512 tokens), we "overflow" the tokens into
        a (batch * chunks, max_length) tensor.
        eg batch might be 32, but some documents are 3 * 512 tokens, others are 5 * 512 tokens, etc.
        So we end up with eg (115, 512) tokens from an input of (32) documents

        We will later reconstruct into
        (chunks, max_length) for each document with the help of the overflow indices
        """
        tokens = self._tokenizer(
            text,
            truncation=True,
            max_length=self._effective_max_length,
            stride=self._stride,
            return_overflowing_tokens=True,
            return_tensors="pt",
            padding="max_length",
        )
        input_ids = tokens["input_ids"]
        attention = tokens["attention_mask"]
        overflow = tokens["overflow_to_sample_mapping"]
        return input_ids, attention, overflow

    @jaxtyped(typechecker=beartype)
    def embed(
        self,
        input_ids: Int[Tensor, "batch tokens"],
        attention: Int[Tensor, "batch tokens"],
        batchsize: int,
    ) -> tuple[Float[Tensor, "batch tokens dim"], Int[Tensor, "batch tokens"]]:
        """
        This function turns a 2D tensor (batch * chunks, tokens) ∈ ℕ  into an embedding
        (batch * chunks, tokens, dim) ∈ ℝ

        The attention mask is used to mask out padding tokens.
        batchsize is the number of chunks to be processed at once
        """
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            attention_mask = attention.to(self.device)
            chunks = input_ids.shape[0]
            embs = []
            for i in range(0, chunks, batchsize):
                input_ids_batch = input_ids[i : i + batchsize]
                attention_mask_batch = attention_mask[i : i + batchsize]
                outputs = self._model(
                    input_ids_batch, attention_mask=attention_mask_batch
                )
                embs.append(outputs.last_hidden_state)
        embeddings = torch.cat(embs, dim=0)

        return embeddings, attention_mask

    @jaxtyped(typechecker=beartype)
    def aggregate(
        self,
        embeddings: Float[Tensor, "batch tokens dim"],
        attention: Int[Tensor, "batch tokens"],
    ) -> Float[Tensor, "batch dim"]:
        """
        This function turns a 3D tensor (batch, tokens, dim) ∈ ℝ
        into an embedding (batch, dim) ∈ ℝ by aggregating over the tokens dimension.

        We can do this because due to the attention mechanism, all
        tokens have been "mixed" like a hologram and
        sort-of contain the information of the full contextwindow.
        """
        mask_expand = attention.unsqueeze(-1)
        sum_emb = torch.sum(embeddings * mask_expand, dim=1)
        sum_mask = torch.sum(mask_expand, dim=1)
        return sum_emb / sum_mask

    @jaxtyped(typechecker=beartype)
    def extend(
        self,
        agg: Float[Tensor, "batch dim"],
        overflow: Int[Tensor, "batch"],
        num_docs: int,
    ) -> dict[str, list[Float[Tensor, "_ dim"]]]:
        """
        With the help of the overflow indices, we can regroup the embeddings back into
        a (chunks, dim) ∈ ℝ tensor per document where chunk varies per document.
        """
        regrouped = []
        for doc_idx in range(num_docs):
            idx = overflow == doc_idx
            embed = agg[idx]
            self.chunk_sizes[embed.shape[0]] += 1
            regrouped.append(embed)
        return {self.col_name: regrouped}

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, inputs: list[str], batchsize: int
    ) -> dict[str, list[Float[Tensor, "_ dim"]]]:
        input_ids, attention, overflow = self.tokenize(inputs)
        embedded, attention = self.embed(input_ids, attention, batchsize=batchsize)
        agg = self.aggregate(embedded, attention)
        return self.extend(agg, overflow, num_docs=len(inputs))

    def fingerprint_fields(self) -> dict:
        # Chunking (stride) changes which text ends up in which chunk, so it
        # changes the output even when the model and context size are identical.
        fields = super().fingerprint_fields()
        fields["stride"] = self._stride
        return fields

    @property
    def get_model(self):
        return self._model

    @property
    def get_tokenizer(self):
        return self._tokenizer

    @property
    def get_stride(self) -> int:
        return self._stride


class ImageVectorizer(BaseVectorizer):
    """
    Image vectorizer using HuggingFace vision models.

    Mirrors :class:`Vectorizer`, but for images: it uses an ``AutoImageProcessor``
    plus an ``AutoModel`` (e.g. a small CNN like ``microsoft/resnet-18`` or a
    distilled ViT like ``facebook/dinov2-small``) and produces a single embedding
    vector per image, shape ``(dim,)``.

    Because each image collapses to one vector (``tensordtype == 1``), the cached
    output plugs directly into the same downstream pipeline as the regex path:
    ``Collate(..., padder=torch.stack)`` + ``Serial([NeuralNet(...)])``.

    The model is only ever run to fill the VectorCache, so the (one-time) cost
    scales with model size while training the head on the cached vectors stays
    cheap on CPU. Swap ``model_name`` to trade embedding quality for speed.
    """

    model_name: str
    col_name: str = "embed"
    input_col: str = "image"
    device: str = Field(default_factory=detect_device)

    _metadata: Any = PrivateAttr()
    _processor: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _hidden_size: int = PrivateAttr()
    _effective_max_length: Optional[int] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def initialize_model(self):
        self._metadata = AutoConfig.from_pretrained(self.model_name)
        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        self._effective_max_length = None

        # Probe the embedding dimension with a dummy forward. This is robust across
        # architectures (ResNet config exposes `hidden_sizes`, ViT/DINOv2 exposes
        # `hidden_size`) and matches whatever `_pool` actually returns.
        self._hidden_size = self._probe_dim()

        logger.info(f"Using device: {self.device}")
        logger.info(f"Image embedding dim: {self._hidden_size}")
        return self

    @jaxtyped(typechecker=beartype)
    def _pool(self, outputs) -> Float[Tensor, "batch dim"]:
        """
        Reduce a vision model's output to one vector per image.

        - If a ``pooler_output`` is present we use it, flattening any trailing
          spatial dims (ResNet: ``(b, dim, 1, 1)`` -> ``(b, dim)``; ViT: ``(b, dim)``).
        - Otherwise we fall back on ``last_hidden_state``: average over the token
          dimension for 3D ``(b, seq, dim)`` outputs, or over the spatial dims for
          4D ``(b, dim, h, w)`` outputs.
        """
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            return pooled.reshape(pooled.shape[0], -1)

        hidden = outputs.last_hidden_state
        if hidden.dim() == 3:  # (batch, seq, dim) -> ViT-like
            return hidden.mean(dim=1)
        if hidden.dim() == 4:  # (batch, dim, h, w) -> CNN-like
            return hidden.mean(dim=(-2, -1))
        raise ValueError(
            f"Cannot pool last_hidden_state with {hidden.dim()} dimensions."
        )

    def _probe_dim(self) -> int:
        from PIL import Image

        dummy = Image.new("RGB", (224, 224))
        inputs = self._processor([dummy], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        return int(self._pool(outputs).shape[-1])

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, inputs: list, batchsize: int
    ) -> dict[str, list[Float[Tensor, "dim"]]]:
        vectors: list[Tensor] = []
        for i in range(0, len(inputs), batchsize):
            batch = [img.convert("RGB") for img in inputs[i : i + batchsize]]
            processed = self._processor(batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self._model(**processed)
            pooled = self._pool(outputs).cpu()
            vectors.extend([vec for vec in pooled])
        return {self.col_name: vectors}

    @property
    def get_metadata(self) -> dict:
        return {
            "model_name": self.model_name,
            "col_name": self.col_name,
            "hidden_size": self._hidden_size,
            "context_size": self._effective_max_length,
        }

    @property
    def get_hidden_size(self) -> int:
        return self._hidden_size

    @property
    def get_model(self):
        return self._model

    @property
    def get_processor(self):
        return self._processor


class PatchImageVectorizer(ImageVectorizer):
    """
    Image vectorizer that **keeps** the patch axis: ``(C, D)`` per image.

    :class:`ImageVectorizer` pools every image down to one vector, which is the
    right call for classification and is why those caches are small. But pooling
    is also the step that destroys *where* -- and a per-patch task (segmentation,
    localisation, dense probing) needs the axis that pooling removes.

    The output shape is the same rung of the tensor ladder as the chunked text
    vectorizer's, so an image cache built here plugs into ``FixedPadding``, the
    aggregators and ``Collate`` unchanged. Aggregating ``C`` away afterwards
    reproduces :class:`ImageVectorizer`; not aggregating is the point.

    Two backends, both handled:

    - ViT-like ``(batch, seq, dim)`` -- the sequence *is* the patch grid, minus
      any leading summary/CLS token (see ``drop_prefix_tokens``).
    - CNN-like ``(batch, dim, h, w)`` -- the spatial grid flattened to
      ``(batch, h*w, dim)``. A CNN carries a grid all the way to its final
      pooling layer; ``AdaptiveAvgPool2d(1)`` is where it gets thrown away.

    Note this is a *separate class* rather than a ``pool="none"`` flag on
    :class:`ImageVectorizer`. ``VectorCache`` reads the rank of the cached tensor
    off the return annotation of ``__call__``, so the declared shape has to match
    what is actually returned -- a runtime flag would leave the annotation lying
    and write the wrong ``tensordtype`` into the metadata.
    """

    col_name: str = "patches"
    drop_prefix_tokens: Optional[int] = None
    """How many leading non-patch tokens (CLS, registers) to drop from a ViT's
    sequence. ``None`` infers it by checking which count leaves a square grid."""

    _grid: Any = PrivateAttr(default=None)

    @jaxtyped(typechecker=beartype)
    def _patches(self, outputs) -> Float[Tensor, "batch chunks dim"]:
        hidden = outputs.last_hidden_state
        if hidden.dim() == 4:  # (b, dim, h, w) -> CNN
            return hidden.flatten(2).transpose(1, 2)
        if hidden.dim() == 3:  # (b, seq, dim) -> ViT
            return hidden[:, self._prefix(hidden.shape[1]) :, :]
        raise ValueError(
            f"Cannot take patches from last_hidden_state with {hidden.dim()} dimensions."
        )

    def _prefix(self, seq_len: int) -> int:
        """Number of leading tokens that are not patches.

        Inferred rather than assumed: a ViT's patches tile a square grid, so the
        right answer is the small offset that leaves a perfect square. DINOv2 has
        one CLS token; a model with register tokens has more; a model with none
        needs zero, and assuming 1 would silently drop a real patch.
        """
        if self.drop_prefix_tokens is not None:
            return self.drop_prefix_tokens
        for prefix in range(0, 9):
            n = seq_len - prefix
            if n > 0 and isqrt(n) ** 2 == n:
                return prefix
        return 0

    def _probe_dim(self) -> int:
        from PIL import Image

        dummy = Image.new("RGB", (224, 224))
        inputs = self._processor([dummy], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        patches = self._patches(outputs)
        n = patches.shape[1]
        side = isqrt(n)
        self._grid = (side, side) if side * side == n else (1, n)
        logger.info(f"Patch grid: {self._grid[0]}x{self._grid[1]} ({n} patches)")
        return int(patches.shape[-1])

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, inputs: list, batchsize: int
    ) -> dict[str, list[Float[Tensor, "chunks dim"]]]:
        vectors: list[Tensor] = []
        for i in range(0, len(inputs), batchsize):
            batch = [img.convert("RGB") for img in inputs[i : i + batchsize]]
            processed = self._processor(batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self._model(**processed)
            patches = self._patches(outputs).cpu()
            vectors.extend([p for p in patches])
        return {self.col_name: vectors}

    @property
    def get_metadata(self) -> dict:
        return {
            "model_name": self.model_name,
            "col_name": self.col_name,
            "hidden_size": self._hidden_size,
            "context_size": self._effective_max_length,
            "patch_grid": list(self._grid) if self._grid else None,
        }

    @property
    def get_grid(self) -> Optional[tuple[int, int]]:
        """(rows, cols) of the patch grid -- needed to fold ``(C, D)`` back into a map."""
        return self._grid


class RegexVectorizer(BaseVectorizer):
    """
    Vectorizer that creates binary feature vectors based on regex pattern matches.
    """

    model_name: str = "regex_vectorizer"
    col_name: str = "regex_features"
    training_texts: Optional[list[str]] = Field(
        default=None, description="Texts to fit on during initialization"
    )
    min_doc_frequency: int = Field(
        default=50, description="Minimum documents a pattern must appear in"
    )
    max_features: int = Field(
        default=1000, description="Maximum number of features (top-k patterns)"
    )
    pattern_builder: Callable[[], re.Pattern] = Field(
        description="Function that returns compiled regex pattern"
    )
    harmonizer: Callable[[Any], str] = Field(
        description="Function that harmonizes match groups into canonical form. "
        "Receives whatever re.findall() produces for the paired pattern_builder: "
        "a bare str when the pattern has 0-1 groups (e.g. harmonize_imdb_match), "
        "a tuple when it has 2+ (e.g. harmonize_legal_reference). A Callable's "
        "parameter type is contravariant, so a Union here would reject *every* "
        "concrete harmonizer -- each only handles the one shape its own pattern "
        "produces, never both -- which is exactly what Any is for: this field's "
        "true input type is a runtime property of the paired pattern, not "
        "something a static signature can pin down without dependent typing."
    )

    _pattern_to_idx: dict[str, int] = PrivateAttr()
    _compiled_pattern: re.Pattern = PrivateAttr()
    _match_counts: Optional[Counter] = PrivateAttr(default=None)
    _doc_frequencies: Optional[Counter] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def initialize_model(self):
        """
        Initialize with compiled pattern.
        """
        self._init_regex()
        return self

    def _init_regex(self):
        """Compile the pattern, set up metadata, and fit if training_texts given.

        Kept separate from the validator so subclasses (e.g. ChunkedRegexVectorizer)
        can reuse it without invoking a pydantic validator through super().
        """
        self._pattern_to_idx = {}
        self._compiled_pattern = self.pattern_builder()
        self._match_counts = None
        self._doc_frequencies = None

        class RegexMetadata:
            def __init__(self, max_features: int):
                self.hidden_size = max_features

        self._metadata = RegexMetadata(self.max_features)
        self._effective_max_length = None

        if self.training_texts is not None:
            self.fit(self.training_texts)

    def _compute_match_counts(self, texts: list[str]) -> tuple[Counter, Counter]:
        match_counts = Counter()
        doc_frequencies = Counter()

        for text in texts:
            matches = self._compiled_pattern.findall(text)
            doc_patterns = set()

            for match in matches:
                harmonized = self.harmonizer(match)
                match_counts[harmonized] += 1
                doc_patterns.add(harmonized)

            for pattern in doc_patterns:
                doc_frequencies[pattern] += 1

        return match_counts, doc_frequencies

    def fit(self, texts: list[str]) -> "RegexVectorizer":
        # Compute or reuse cached counts
        if self._match_counts is None or self._doc_frequencies is None:
            self._match_counts, self._doc_frequencies = self._compute_match_counts(
                texts
            )

        # Filter by minimum document frequency
        filtered_patterns = [
            pattern
            for pattern, doc_freq in self._doc_frequencies.items()
            if doc_freq >= self.min_doc_frequency
        ]

        # Take top-k by total frequency
        top_patterns = [
            pattern
            for pattern, _ in self._match_counts.most_common(self.max_features)
            if pattern in filtered_patterns
        ]

        # Store pattern lookup
        self._pattern_to_idx = {
            pattern: i for i, pattern in enumerate(top_patterns[: self.max_features])
        }

        # Update metadata with actual feature count
        self._metadata.hidden_size = len(self._pattern_to_idx)

        logger.info(f"Fitted {len(self._pattern_to_idx)} patterns")

        return self

    def _vectorize_text(self, text: str) -> Tensor:
        """Run the regexes on a single string and return its binary feature vector.

        Operates on whatever string it is given, so it works equally on a whole
        document (RegexVectorizer) or a single chunk (ChunkedRegexVectorizer).
        Returns a CPU tensor; callers move it to the device.
        """
        vector = torch.zeros(len(self._pattern_to_idx), dtype=torch.float32)
        doc_patterns = {
            self.harmonizer(match) for match in self._compiled_pattern.findall(text)
        }
        for pattern in doc_patterns:
            idx = self._pattern_to_idx.get(pattern)
            if idx is not None:
                vector[idx] = 1.0
        return vector

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, inputs: list[str], batchsize: int = 32
    ) -> dict[str, list[Float[Tensor, "hidden_size"]]]:
        if not self._pattern_to_idx:
            raise RuntimeError(
                "Vectorizer must be fitted before calling. Run .fit(texts) first."
            )

        vectors = [self._vectorize_text(text).to(self.device) for text in inputs]
        return {self.col_name: vectors}

    def print_stats(
        self, texts: Optional[list[str]] = None, top_k: int = 20, plot: bool = True
    ):
        if texts is None:
            if self._match_counts is None:
                raise RuntimeError(
                    "No cached match counts. Either call fit() first or provide texts."
                )
            match_counts = self._match_counts
        else:
            match_counts, _ = self._compute_match_counts(texts)

        logger.info(f"Total unique patterns: {len(match_counts)}")
        logger.info(f"most commonn {match_counts.most_common(top_k)}")

        if plot and len(match_counts) > 0:
            plt.figure(figsize=(12, 6))
            top_matches = match_counts.most_common(min(50, len(match_counts)))
            labels = [ref for ref, _ in top_matches]
            counts = [count for _, count in top_matches]

            plt.bar(range(len(counts)), counts)
            plt.xticks(range(len(labels)), labels, rotation=90, ha="right")
            plt.xlabel("Pattern")
            plt.ylabel("Frequency")
            plt.title(f"Top {len(counts)} Pattern Matches")
            plt.tight_layout()
            plt.show()

    @property
    def get_metadata(self) -> dict:
        """Return metadata about the vectorizer"""
        base_metadata = super().get_metadata
        base_metadata.update(
            {
                "min_doc_frequency": self.min_doc_frequency,
                "max_features": self.max_features,
            }
        )
        return base_metadata

    def fingerprint_fields(self) -> dict:
        """Add the regex and the *fitted* vocabulary to the fingerprint.

        `model_name` is a constant here and `pattern_builder`/`harmonizer` are
        callables the base class cannot describe, so two RegexVectorizers fitted
        on different corpora would otherwise look identical to the cache -- and a
        different fit means different feature columns for the same input text.
        `__call__` uses exactly the compiled pattern and the pattern->index
        mapping, so those two pin the output down; `training_texts` is dropped
        because any corpus producing the same fit produces the same vectors.
        """
        fields = super().fingerprint_fields()
        fields.pop("training_texts", None)
        fields.update(
            {
                "pattern": self._compiled_pattern.pattern,
                "pattern_flags": int(self._compiled_pattern.flags),
                # Name only: the qualname is stable across processes where a
                # function's repr (memory address) is not. The harmonizer's
                # actual effect on the fitted corpus shows up in fitted_patterns.
                "harmonizer": getattr(self.harmonizer, "__qualname__", ""),
                # Sorted by index: this is the exact column layout of the output.
                "fitted_patterns": [
                    pattern
                    for pattern, _ in sorted(
                        self._pattern_to_idx.items(), key=lambda kv: kv[1]
                    )
                ],
            }
        )
        return fields


class ChunkedRegexVectorizer(RegexVectorizer):
    """Regex vectorizer that aligns its binary features with an embedder's chunks.

    Where ``RegexVectorizer`` emits one ``(hidden_size,)`` vector per document, this
    emits a ``(chunks, hidden_size)`` matrix whose rows line up with the chunks an
    embedding ``Vectorizer`` produces. Chunk ``i`` of the regex matrix then
    corresponds to chunk ``i`` of the embedding, so the two can be concatenated
    per chunk downstream.

    Alignment contract: chunking is fully determined by the triple
    ``(tokenizer_name, context_size, stride)``. Pass the values stored in an
    embedding cache's metadata (``model_tag``, ``context_size``, ``stride``) and the
    chunk counts/boundaries are guaranteed identical -- without re-deriving any
    stride logic. Only the *tokenizer* is loaded (no model weights): chunking
    happens in token space and is mapped back to character spans via
    ``offset_mapping`` so the regexes can run on each chunk's raw substring.

    Fast (Rust) tokenizers are required for ``offset_mapping``. If the tokenizer is
    not fast, we do not crash: we warn, fall back to a single whole-document chunk
    per text (a ``(1, hidden_size)`` matrix), and record ``offsets_supported=False``
    in the cache metadata.
    """

    model_name: str = "chunked_regex_vectorizer"
    col_name: str = "chunked_regex"
    tokenizer_name: str = Field(
        description="HF model tag of the embedder whose chunking we align to"
    )
    context_size: int = Field(
        description="Effective max_length (tokens) used by the embedder"
    )
    stride: int = Field(description="Token overlap between consecutive chunks")

    _tokenizer: Any = PrivateAttr()
    _supports_offsets: bool = PrivateAttr(default=True)

    @model_validator(mode="after")
    def initialize_model(self):
        self._init_regex()  # compile pattern + fit if training_texts given
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        # Report the aligned context window through the BaseVectorizer API.
        self._effective_max_length = self.context_size
        # offset_mapping (token chunk -> char span) only exists on fast tokenizers.
        self._supports_offsets = bool(getattr(self._tokenizer, "is_fast", False))
        if not self._supports_offsets:
            logger.warning(
                f"Tokenizer for '{self.tokenizer_name}' is not a fast tokenizer and "
                "does not support offset_mapping. Falling back to a single "
                "whole-document chunk per text: regex features will NOT be "
                "chunk-aligned (metadata records offsets_supported=False)."
            )
        return self

    @property
    def get_stride(self) -> int:
        return self.stride

    @property
    def get_offsets_supported(self) -> bool:
        return self._supports_offsets

    def fingerprint_fields(self) -> dict:
        # tokenizer_name/context_size/stride are public fields and come along
        # automatically; the offsets fallback is private state and emits a
        # different chunking (one whole-document chunk) entirely.
        fields = super().fingerprint_fields()
        fields["offsets_supported"] = self._supports_offsets
        return fields

    def _chunk_texts(self, texts: list[str]) -> tuple[list[str], list[int]]:
        """Recover, for every chunk, the raw character substring of its document.

        Tokenizes with the embedder's exact settings, then uses offset_mapping to
        slice each chunk back out of the original string. Returns
        ``(chunk_texts, overflow)`` where ``overflow[i]`` is the source document
        index of chunk ``i`` (same semantics as ``Vectorizer``'s overflow mapping).
        """
        enc = self._tokenizer(
            texts,
            truncation=True,
            max_length=self.context_size,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )
        offsets = enc["offset_mapping"]
        overflow = list(enc["overflow_to_sample_mapping"])

        chunk_texts = []
        for row, doc_idx in zip(offsets, overflow):
            # Special/pad tokens carry the offset (0, 0); real tokens carry the
            # char span they cover in the original document.
            real = [(s, e) for (s, e) in row if not (s == 0 and e == 0)]
            if real:
                start, end = real[0][0], real[-1][1]
                chunk_texts.append(texts[doc_idx][start:end])
            else:
                chunk_texts.append("")
        return chunk_texts, overflow

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, inputs: list[str], batchsize: int = 32
    ) -> dict[str, list[Float[Tensor, "chunks hidden_size"]]]:
        if not self._pattern_to_idx:
            raise RuntimeError(
                "Vectorizer must be fitted before calling. Run .fit(texts) first."
            )

        if not self._supports_offsets:
            # No alignment possible: one whole-document chunk per text, still a 2D
            # (1, hidden_size) tensor so the cache schema matches the aligned case.
            out = [
                self._vectorize_text(text).unsqueeze(0).to(self.device)
                for text in inputs
            ]
            return {self.col_name: out}

        chunk_texts, overflow = self._chunk_texts(inputs)
        chunk_vecs = [self._vectorize_text(t) for t in chunk_texts]

        out = []
        for doc_idx in range(len(inputs)):
            rows = [vec for vec, d in zip(chunk_vecs, overflow) if d == doc_idx]
            out.append(torch.stack(rows).to(self.device))
        return {self.col_name: out}


def build_legal_reference_pattern() -> re.Pattern:
    """Build the regex pattern for Dutch legal references
    eg:
        "artikel 265 Boek 3 van het Burgerlijk Wetboek",
        "artikel 7:2 Burgerlijk Wetboek",
        "artikel 7:26 lid 3 van het Burgerlijk Wetboek",
        "artikelen 6:251 en 6:252 Burgerlijk Wetboek",
        "artikel 55 Wet Bodembescherming"
    """
    article_prefix = r"artikel(?:en)?"
    article_number = r"\d+(?::\d+)?"
    article_modifier = r"(?:\s+(?:en\s+\d+(?::\d+)?|lid\s+\d+))?"
    book_reference = r"(?:\s+[Bb]oek\s+\d+)?"
    connector = r"(?:\s+van\s+het\s+)?"

    law_name = (
        r"(?:[Bb]urgerlijk\s+[Ww]etboek|\bBW\b|[Ww]et\s+[Bb]odembescherming|\bWbb\b)"
    )

    full_pattern = (
        rf"\b{article_prefix}\s+"
        rf"({article_number}{article_modifier}{book_reference})\s+"
        rf"{connector}({law_name})"
    )
    return re.compile(full_pattern)


def harmonize_legal_reference(match: tuple) -> str:
    """Convert legal reference match to harmonized format"""
    article_ref, law_name = match
    law_lower = law_name.lower()

    if "burgerlijk wetboek" in law_lower or law_lower == "bw":
        law_abbr = "BW"
    elif "bodembescherming" in law_lower or law_lower == "wbb":
        law_abbr = "Bodem"
    else:
        law_abbr = law_name

    return f"{article_ref} {law_abbr}"


def build_imdb_review_pattern() -> re.Pattern:
    """Match film-related vocabulary in IMDB reviews.

    Each match is a single word/phrase, e.g. "great", "terrible", "acting",
    "horror". After fitting, each unique matched term becomes one binary
    feature dimension (1 if the word appears anywhere in the text, 0 if not).

    Examples:
        "The acting was great but the plot was terrible"
        → matches: ["acting", "great", "plot", "terrible"]
    """
    sentiment = (
        "great|good|bad|poor|excellent|terrible|awful|brilliant|boring|"
        "amazing|weak|superb|dreadful|outstanding|mediocre|impressive|"
        "masterpiece|hilarious|touching|predictable|formulaic"
    )
    genre = (
        "horror|comedy|thriller|drama|romance|action|western|"
        "documentary|musical|mystery|fantasy|animation"
    )
    craft = (
        "acting|performance|screenplay|script|direction|plot|"
        "cinematography|dialogue|soundtrack|editing"
    )
    return re.compile(rf"\b(?:{sentiment}|{genre}|{craft})\b", re.IGNORECASE)


def harmonize_imdb_match(match: str) -> str:
    """Normalize a matched film vocabulary word to lowercase."""
    return match.lower()
