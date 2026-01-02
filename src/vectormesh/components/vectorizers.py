"""Text vectorization components using HuggingFace models."""

from typing import Optional, Union, Literal, Any
from pathlib import Path

import torch
from beartype.typing import List
from jaxtyping import jaxtyped
from beartype import beartype as typechecker
from pydantic import ConfigDict, Field
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

from vectormesh.base import VectorMeshComponent
from vectormesh.errors import VectorMeshError
from vectormesh.types import TwoDTensor, ThreeDTensor
from vectormesh.utils.model_info import get_model_metadata, ModelMetadata


class BaseVectorizer(VectorMeshComponent):
    """Base class for text vectorizers with shared functionality."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    model_name: str
    device: Optional[str] = None

    # Internal state
    _metadata: Optional[ModelMetadata] = None
    _model: Optional[Any] = None
    _tokenizer: Optional[Any] = None

    def model_post_init(self, __context):
        """Initialize model and introspect metadata."""
        # Get model metadata via AutoConfig (fast, no full model download)
        self._metadata = get_model_metadata(self.model_name)

        # Cache the metadata for properties
        object.__setattr__(self, "_metadata", self._metadata)

    @property
    def output_mode(self) -> Literal["2d", "3d"]:
        """Dimension of output tensors (2D for pooled, 3D for chunked)."""
        return self._metadata.output_mode

    @property
    def embedding_dim(self) -> int:
        """Dimension of embedding vectors."""
        return self._metadata.hidden_size

    @property
    def context_window(self) -> int:
        """Maximum tokens per chunk."""
        return self._metadata.max_position_embeddings

    def _detect_device(self) -> str:
        """Auto-detect the best available device.

        Returns:
            Device string: 'cuda', 'mps', or 'cpu'
        """
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"


class TwoDVectorizer(BaseVectorizer):
    """Sentence-transformer vectorizer producing 2D output (pooled embeddings).

    Uses sentence-transformers library for models with built-in pooling.

    Args:
        model_name: HuggingFace sentence-transformer model ID
        device: Target device ('cuda', 'mps', 'cpu', or None for auto-detection)

    Example:
        ```python
        vectorizer = TwoDVectorizer(model_name="sentence-transformers/all-MiniLM-L6-v2")
        embeddings = vectorizer(["Hello world"])  # Returns: TwoDTensor [1, 384]
        ```

    Shapes:
        Input: List[str] with N strings
        Output: TwoDTensor [batch, dim] where batch=N texts, dim=embedding_dimension
    """

    def model_post_init(self, __context):
        """Initialize and validate 2D model."""
        super().model_post_init(__context)

        if self._metadata.output_mode != "2d":
            raise VectorMeshError(
                message=f"TwoDVectorizer requires 2D model, got {self._metadata.output_mode} for {self.model_name}",
                hint="Use ThreeDVectorizer for raw transformers or TextVectorizer for automatic detection.",
                fix=f"Try: TwoDVectorizer(model_name='sentence-transformers/all-MiniLM-L6-v2')"
            )

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load and cache the sentence transformer model."""
        if self._model is not None:
            return self._model

        # Determine device
        target_device = self.device if self.device is not None else self._detect_device()

        try:
            # Load sentence-transformer model
            model = SentenceTransformer(self.model_name, device=target_device)
            # Cache the model (use object.__setattr__ to bypass frozen config)
            object.__setattr__(self, "_model", model)
            return model

        except Exception as e:
            raise VectorMeshError(
                message=f"Failed to load 2D model '{self.model_name}': {str(e)}",
                hint="Check that the model ID is correct and supports sentence-transformers.",
                fix=f"Try: `huggingface-cli download {self.model_name}` to test manually."
            ) from e

    @jaxtyped(typechecker=typechecker)
    def __call__(self, texts: List[str]) -> TwoDTensor:
        """Vectorize texts using sentence-transformers (pooled output).

        Args:
            texts: List of text strings to vectorize

        Returns:
            TwoDTensor[batch, dim] with pooled embeddings

        Shapes:
            Input: List[str] with N strings
            Output: [N, embedding_dim]
        """
        try:
            model = self._get_model()

            # Encode texts (returns tensor directly)
            embeddings = model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=self.device
            )
            return embeddings  # Shape: [batch, dim]

        except VectorMeshError:
            # Re-raise our custom errors as-is
            raise

        except Exception as e:
            raise VectorMeshError(
                message=f"Failed to encode texts with 2D model: {str(e)}",
                hint="Text encoding failed - check that inputs are valid strings",
                fix="Ensure all texts are non-empty strings and the model loaded correctly",
            ) from e


class ThreeDVectorizer(BaseVectorizer):
    """Raw transformer vectorizer producing 3D output (chunked embeddings).

    Uses raw transformers with manual chunking for models without built-in pooling.

    Args:
        model_name: HuggingFace raw transformer model ID
        auto_chunk: Whether to chunk texts longer than context window
        chunk_size: Maximum tokens per chunk (overrides model's max_position_embeddings)
        device: Target device ('cuda', 'mps', 'cpu', or None for auto-detection)

    Example:
        ```python
        vectorizer = ThreeDVectorizer(model_name="bert-base-uncased")
        chunks = vectorizer(["Long document..."])  # Returns: ThreeDTensor [1, chunks, 768]
        ```

    Shapes:
        Input: List[str] with N strings
        Output: ThreeDTensor [batch, chunks, dim] where chunks=max_chunks across batch
    """

    auto_chunk: bool = True
    chunk_size: Optional[int] = None

    def model_post_init(self, __context):
        """Initialize and validate 3D model."""
        super().model_post_init(__context)

        if self._metadata.output_mode != "3d":
            raise VectorMeshError(
                message=f"ThreeDVectorizer requires 3D model, got {self._metadata.output_mode} for {self.model_name}",
                hint="Use TwoDVectorizer for sentence-transformers or TextVectorizer for automatic detection.",
                fix=f"Try: ThreeDVectorizer(model_name='bert-base-uncased')"
            )

    @property
    def context_window(self) -> int:
        """Maximum tokens per chunk."""
        return self.chunk_size or self._metadata.max_position_embeddings

    def _get_model_and_tokenizer(self):
        """Lazy-load and cache the model and tokenizer."""
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        # Determine device
        target_device = self.device if self.device is not None else self._detect_device()

        try:
            # Raw transformer (requires manual pooling/chunking)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            model = model.to(target_device)

            # Cache both (use object.__setattr__ to bypass frozen config)
            object.__setattr__(self, "_model", model)
            object.__setattr__(self, "_tokenizer", tokenizer)

            return model, tokenizer

        except Exception as e:
            raise VectorMeshError(
                message=f"Failed to load 3D model '{self.model_name}': {str(e)}",
                hint="Check that the model ID is correct and you have internet connectivity.",
                fix=f"Try: `huggingface-cli download {self.model_name}` to test manually."
            ) from e

    @jaxtyped(typechecker=typechecker)
    def __call__(self, texts: List[str]) -> ThreeDTensor:
        """Vectorize texts using raw transformers with chunking.

        Args:
            texts: List of text strings to vectorize

        Returns:
            ThreeDTensor[batch, chunks, dim] with chunked embeddings

        Shapes:
            Input: List[str] with N strings
            Output: [N, max_chunks, embedding_dim]
        """
        try:
            model, tokenizer = self._get_model_and_tokenizer()

            all_chunks = []
            max_chunks = 0

            for text in texts:
                # Tokenize and chunk
                tokens = tokenizer(
                    text,
                    truncation=False,
                    return_tensors="pt",
                    padding=False
                )["input_ids"][0]

                # Split into chunks
                chunks = self._split_into_chunks(tokens, self.context_window)

                # Embed each chunk
                chunk_embeddings = []
                for chunk in chunks:
                    outputs = model(chunk.unsqueeze(0).to(model.device))
                    # Use mean pooling over tokens (not CLS)
                    embedding = outputs.last_hidden_state.mean(dim=1)  # [1, dim]
                    chunk_embeddings.append(embedding)

                # Stack chunks for this document
                doc_chunks = torch.cat(chunk_embeddings, dim=0)  # [num_chunks, dim]
                all_chunks.append(doc_chunks)
                max_chunks = max(max_chunks, len(chunks))

            # Pad all documents to max_chunks
            padded_chunks = []
            for doc_chunks in all_chunks:
                if len(doc_chunks) < max_chunks:
                    padding = torch.zeros(
                        max_chunks - len(doc_chunks),
                        self.embedding_dim,
                        device=doc_chunks.device
                    )
                    doc_chunks = torch.cat([doc_chunks, padding], dim=0)
                padded_chunks.append(doc_chunks)

            result = torch.stack(padded_chunks, dim=0)  # [batch, max_chunks, dim]
            return result

        except VectorMeshError:
            # Re-raise our custom errors as-is
            raise

        except Exception as e:
            raise VectorMeshError(
                message=f"Failed to encode texts with 3D model: {str(e)}",
                hint="Text chunking and encoding failed - check inputs and model compatibility",
                fix="Ensure texts are valid and model supports the chunking strategy",
            ) from e

    def _split_into_chunks(self, tokens: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
        """Split token sequence into fixed-size chunks."""
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i + chunk_size]
            chunks.append(chunk)
        return chunks


class TextVectorizer(BaseVectorizer):
    """Automatic text vectorizer that routes to TwoDVectorizer or ThreeDVectorizer.

    Automatically detects whether a model produces 2D (sentence-transformers)
    or 3D (raw transformers) output and delegates to the appropriate vectorizer.

    Args:
        model_name: HuggingFace model identifier
        auto_chunk: Whether to chunk texts longer than context window (3D only)
        chunk_size: Maximum tokens per chunk (3D only, overrides model's max_position_embeddings)
        device: Target device ('cuda', 'mps', 'cpu', or None for auto-detection)

    Example:
        ```python
        # Automatically routes to TwoDVectorizer
        vectorizer = TextVectorizer(model_name="sentence-transformers/all-MiniLM-L6-v2")
        embeddings = vectorizer(["Hello world"])  # Returns: TwoDTensor [1, 384]

        # Automatically routes to ThreeDVectorizer
        vectorizer = TextVectorizer(model_name="bert-base-uncased")
        chunks = vectorizer(["Long document..."])  # Returns: ThreeDTensor [1, chunks, 768]
        ```

    Shapes:
        2D output: TwoDTensor [batch, dim]
        3D output: ThreeDTensor [batch, chunks, dim]
    """

    auto_chunk: bool = True
    chunk_size: Optional[int] = None

    # Internal vectorizers
    _twod_vectorizer: Optional[TwoDVectorizer] = None
    _threed_vectorizer: Optional[ThreeDVectorizer] = None

    @property
    def context_window(self) -> int:
        """Maximum tokens per chunk."""
        return self.chunk_size or self._metadata.max_position_embeddings

    def _get_vectorizer(self) -> Union[TwoDVectorizer, ThreeDVectorizer]:
        """Get the appropriate vectorizer based on model type."""
        if self._metadata.output_mode == "2d":
            if self._twod_vectorizer is None:
                vectorizer = TwoDVectorizer(model_name=self.model_name, device=self.device)
                object.__setattr__(self, "_twod_vectorizer", vectorizer)
            return self._twod_vectorizer
        else:
            if self._threed_vectorizer is None:
                vectorizer = ThreeDVectorizer(
                    model_name=self.model_name,
                    auto_chunk=self.auto_chunk,
                    chunk_size=self.chunk_size,
                    device=self.device
                )
                object.__setattr__(self, "_threed_vectorizer", vectorizer)
            return self._threed_vectorizer

    @jaxtyped(typechecker=typechecker)
    def __call__(self, texts: List[str]) -> Union[TwoDTensor, ThreeDTensor]:
        """Vectorize texts with automatic dimension detection.

        Args:
            texts: List of text strings to vectorize

        Returns:
            TwoDTensor[batch, dim] if output_mode=="2d"
            ThreeDTensor[batch, chunks, dim] if output_mode=="3d"

        Shapes:
            2D output: [B, D] where B=batch_size, D=embedding_dim
            3D output: [B, C, D] where C=max_chunks across batch
        """
        vectorizer = self._get_vectorizer()
        return vectorizer(texts)