"""Model introspection utilities for VectorMesh."""
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict
from transformers import AutoConfig
from ..types import VectorMeshError


class ModelMetadata(BaseModel):
    """Metadata extracted from HuggingFace AutoConfig.

    Attributes:
        model_id: HuggingFace model ID
        max_position_embeddings: Maximum context window (tokens)
        hidden_size: Embedding dimension
        output_mode: Whether model produces 2D or 3D output
        pooling_strategy: Pooling method for sentence-transformers
    """
    model_id: str
    max_position_embeddings: int
    hidden_size: int
    output_mode: Literal["2d", "3d"]
    pooling_strategy: Optional[Literal["mean", "cls", "max"]] = None

    model_config = ConfigDict(frozen=True)


def get_model_metadata(model_id: str, cache_dir: Optional[Path] = None) -> ModelMetadata:
    """Query HuggingFace AutoConfig for model metadata.

    This function downloads ONLY the config.json (< 10KB) without loading
    the full model weights. Results are cached locally to avoid repeated
    network calls.

    Args:
        model_id: HuggingFace model ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")
        cache_dir: Optional local cache directory for config files

    Returns:
        ModelMetadata with introspected model properties

    Raises:
        VectorMeshError: If model not found or config invalid

    Examples:
        >>> metadata = get_model_metadata("sentence-transformers/all-MiniLM-L6-v2")
        >>> assert metadata.output_mode == "2d"
        >>> assert metadata.max_position_embeddings == 512
    """
    try:
        config = AutoConfig.from_pretrained(
            model_id,
            cache_dir=str(cache_dir) if cache_dir else None
        )
    except Exception as e:
        raise VectorMeshError(
            message=f"Failed to load config for model: {model_id}",
            hint="Check that the model ID is correct and you have internet connectivity.",
            fix=f"Try: `huggingface-cli download {model_id} config.json` to test manually."
        ) from e

    # Detect sentence-transformer vs raw transformer
    is_sentence_transformer = (
        "sentence-transformers" in model_id or
        hasattr(config, "pooling_mode_mean_tokens")  # ST config marker
    )

    output_mode = "2d" if is_sentence_transformer else "3d"
    pooling = _detect_pooling_strategy(config) if output_mode == "2d" else None

    return ModelMetadata(
        model_id=model_id,
        max_position_embeddings=config.max_position_embeddings,
        hidden_size=getattr(config, "hidden_size", getattr(config, "dim", None)),  # Handle variations
        output_mode=output_mode,
        pooling_strategy=pooling
    )


def _detect_pooling_strategy(config) -> Optional[Literal["mean", "cls", "max"]]:
    """Detect pooling strategy from sentence-transformer config."""
    if hasattr(config, "pooling_mode_mean_tokens") and config.pooling_mode_mean_tokens:
        return "mean"
    elif hasattr(config, "pooling_mode_cls_token") and config.pooling_mode_cls_token:
        return "cls"
    elif hasattr(config, "pooling_mode_max_tokens") and config.pooling_mode_max_tokens:
        return "max"
    return "mean"  # Default fallback