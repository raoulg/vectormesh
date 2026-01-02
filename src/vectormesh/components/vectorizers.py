"""Text vectorization components using HuggingFace models."""

from typing import Optional, Literal, Any
import torch
from beartype.typing import List
from jaxtyping import jaxtyped
from beartype import beartype as typechecker
from pydantic import ConfigDict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

from vectormesh.types import VectorMeshComponent, VectorMeshError, TwoDTensor, ThreeDTensor, NDTensor
from vectormesh.utils.model_info import get_model_metadata, ModelMetadata


class BaseVectorizer(VectorMeshComponent):
    """Base class for text vectorizers with shared functionality."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    model_name: str
    device: Optional[str] = None

    # Internal state
    _metadata: Optional[ModelMetadata] = None
    _model: Any = None
    _tokenizer: Any = None

    def __call__(self, texts: List[str]) -> NDTensor:
        """Vectorize texts. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement __call__")

    def model_post_init(self, __context):
        """Initialize model - NO autodetection here (optimistic loading)."""
        pass

    @property
    def output_mode(self) -> Literal["2d", "3d"]:
        """Dimension of output tensors (2D for pooled, 3D for chunked)."""
        # We only introspect if needed or if model is loaded
        if self._metadata is None:
             self._metadata = get_model_metadata(self.model_name)
        return self._metadata.output_mode

    @property
    def embedding_dim(self) -> int:
        """Dimension of embedding vectors."""
        if self._metadata is None:
             self._metadata = get_model_metadata(self.model_name)
        return self._metadata.hidden_size

    @property
    def context_window(self) -> int:
        """Maximum tokens per chunk."""
        if self._metadata is None:
             self._metadata = get_model_metadata(self.model_name)
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

    def _check_compatibility(self, required_mode: Literal["2d", "3d"]) -> None:
        """Check if model compatibility matches required mode.

        Args:
            required_mode: The output mode required by the vectorizer ("2d" or "3d")

        Raises:
            VectorMeshError: If model compatibility does not match.
        """
        try:
             metadata = get_model_metadata(self.model_name)
             # Cache metadata while we have it
             object.__setattr__(self, "_metadata", metadata)

             if metadata.output_mode != required_mode:
                 if required_mode == "2d":
                     raise VectorMeshError(
                         message=f"TwoDVectorizer requires a 2D model (sentence-transformer), but '{self.model_name}' is a 3D model.",
                         hint="This model produces 3D output (chunks). Use `ThreeDVectorizer` instead.",
                         fix=f"vectorizer = ThreeDVectorizer(model_name='{self.model_name}')"
                     )
                 else:
                     raise VectorMeshError(
                         message=f"ThreeDVectorizer requires a 3D model (raw transformer), but '{self.model_name}' is a 2D model.",
                         hint="This model produces 2D output (pooled). Use `TwoDVectorizer` instead.",
                         fix=f"vectorizer = TwoDVectorizer(model_name='{self.model_name}')"
                     )

        except VectorMeshError:
            raise

        except Exception:
            # If we can't get metadata, we can't give a specific compatibility hint.
            # We'll just let the original error bubble up or be handled by the caller.
            pass


class TwoDVectorizer(BaseVectorizer):
    """Sentence-transformer vectorizer producing 2D output (pooled embeddings).

    Optimistically attempts to load the model. If loading fails, checks if the
    model is actually a 3D model and provides a helpful error.

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

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load and cache the sentence transformer model."""
        if self._model is not None:
            return self._model

        # Determine device
        target_device = self.device if self.device is not None else self._detect_device()

        try:
            # Check compatibility BEFORE loading (efficiency)
            self._check_compatibility(required_mode="2d")

            # Optimistic load
            model = SentenceTransformer(self.model_name, device=target_device)
            # Cache the model
            object.__setattr__(self, "_model", model)
            
            # Enforce strict compatibility is done at start.
            
            return model

        except VectorMeshError:
            raise

        except Exception as e:
            # Check if it failed because verification was skipped (e.g. metadata fetch failed)?
            # Or if it's a genuine loading error.
            # We can try check again, or just handle error.
            # safe to just handle error.

            # If check passed (or failed to check), raise the original error
            # wrapped in VectorMeshError
            raise VectorMeshError(
                message=f"Failed to load user-specified 2D model '{self.model_name}': {str(e)}",
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

    Optimistically attempts to load the model. If loading fails, checks if the
    model is actually a 2D model and provides a helpful error.

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

    @property
    def context_window(self) -> int:
        """Maximum tokens per chunk."""
        # Use cached metadata if available, otherwise fetch it
        if self.chunk_size:
            return self.chunk_size
        if self._metadata is None:
             self._metadata = get_model_metadata(self.model_name)
        return self._metadata.max_position_embeddings

    def _get_model_and_tokenizer(self):
        """Lazy-load and cache the model and tokenizer."""
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        # Determine device
        target_device = self.device if self.device is not None else self._detect_device()

        try:
            # Check compatibility BEFORE loading (efficiency)
            self._check_compatibility(required_mode="3d")

            # Optimistic load
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            model = model.to(target_device)

            # Cache both
            object.__setattr__(self, "_model", model)
            object.__setattr__(self, "_tokenizer", tokenizer)

            # Enforce strict compatibility is done at start.
            
            return model, tokenizer

        except VectorMeshError:
            raise

        except Exception as e:
            # Check compatibility is done at start.
            
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