"""Text vectorization components using HuggingFace models."""

from typing import Optional

import torch
from beartype.typing import List
from jaxtyping import jaxtyped
from beartype import beartype as typechecker
from pydantic import ConfigDict
from sentence_transformers import SentenceTransformer

from vectormesh.base import VectorMeshComponent
from vectormesh.errors import VectorMeshError
from vectormesh.types import TwoDTensor


class TextVectorizer(VectorMeshComponent):
    """Vectorize text using HuggingFace sentence transformer models.

    This component wraps HuggingFace sentence-transformers for easy text embedding
    generation with automatic device management and educational error handling.

    Args:
        model_name: HuggingFace model identifier (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
        device: Target device ('cuda', 'mps', 'cpu', or None for auto-detection)

    Example:
        ```python
        vectorizer = TextVectorizer(model_name="sentence-transformers/all-MiniLM-L6-v2")
        embeddings = vectorizer(["Hello world", "AI is amazing"])
        # Returns: TwoDTensor with shape (2, 384)
        ```

    Shapes:
        Input: List[str] with N strings
        Output: TwoDTensor (N, embedding_dim)
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    model_name: str
    device: Optional[str] = None

    # Private cached model instance (not part of Pydantic fields)
    _model: Optional[SentenceTransformer] = None

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

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load and cache the sentence transformer model.

        Returns:
            Loaded SentenceTransformer instance

        Raises:
            VectorMeshError: If model loading fails with educational hints
        """
        # Use cached model if available
        if self._model is not None:
            return self._model

        # Determine device
        target_device = self.device if self.device is not None else self._detect_device()

        try:
            # Load model with device placement
            model = SentenceTransformer(self.model_name, device=target_device)
            # Cache the model (use object.__setattr__ to bypass frozen config)
            object.__setattr__(self, "_model", model)
            return model

        except OSError as e:
            # Model not found on HuggingFace Hub
            raise VectorMeshError(
                message=f"Failed to load model '{self.model_name}': {str(e)}",
                hint="Invalid HuggingFace model identifier or network issue",
                fix="Check model name at https://huggingface.co/models?library=sentence-transformers or try 'sentence-transformers/all-MiniLM-L6-v2'",
            ) from e

        except Exception as e:
            # Generic error
            raise VectorMeshError(
                message=f"Unexpected error loading model '{self.model_name}': {str(e)}",
                hint="Model loading failed unexpectedly",
                fix="Check your internet connection and ensure the model is compatible with sentence-transformers",
            ) from e

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, texts: List[str]
    ) -> TwoDTensor:
        """Encode a list of text strings into embeddings.

        Args:
            texts: List of text strings to vectorize

        Returns:
            Tensor of shape (batch, embedding_dim) containing the embeddings

        Raises:
            VectorMeshError: If encoding fails with educational hints

        Shapes:
            Input: List[str] with N strings
            Output: (N, embedding_dim)
        """
        try:
            # Get or load model
            model = self._get_model()

            # Encode texts (returns NumPy array)
            embeddings_np = model.encode(texts)

            # Convert to PyTorch tensor
            embeddings_tensor = torch.from_numpy(embeddings_np).float()

            return embeddings_tensor

        except VectorMeshError:
            # Re-raise our custom errors as-is
            raise

        except Exception as e:
            # Catch any encoding errors
            raise VectorMeshError(
                message=f"Failed to encode texts: {str(e)}",
                hint="Text encoding failed - check that inputs are valid strings",
                fix="Ensure all texts are non-empty strings and the model is loaded correctly",
            ) from e
