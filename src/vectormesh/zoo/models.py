"""Curated model registry with validated metadata.

All 10 models have been tested via HuggingFace MCP for:
- Context window size (max_position_embeddings)
- Embedding dimension (hidden_size)
- Pooling strategy (for sentence-transformers)
- Dutch language support (for course requirements)
"""

from dataclasses import dataclass
from typing import Literal
from enum import Enum


@dataclass(frozen=True)
class ZooModel:
    """Metadata for a validated HuggingFace model.

    Attributes:
        model_id: HuggingFace model ID
        context_window: Maximum context window (tokens)
        embedding_dim: Embedding dimension
        output_mode: Whether model produces 2D or 3D output
        description: Human-readable description
    """
    model_id: str
    context_window: int
    embedding_dim: int
    output_mode: Literal["2d", "3d"]
    description: str


# MVP Models (4)
MPNET = ZooModel(
    model_id="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    context_window=514,
    embedding_dim=768,
    output_mode="2d",
    description="Dutch support, 249M downloads, best general performance"
)

QWEN_0_6B = ZooModel(
    model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    context_window=32768,
    embedding_dim=896,
    output_mode="3d",  # No built-in pooling
    description="32k context, CPU-friendly (0.5B params), multilingual"
)

LABSE = ZooModel(
    model_id="sentence-transformers/LaBSE",
    context_window=512,
    embedding_dim=768,
    output_mode="2d",
    description="109 languages, 470M params"
)

MINILM = ZooModel(
    model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    context_window=512,
    embedding_dim=384,
    output_mode="2d",
    description="Fastest (117M params), good baseline"
)

# Growth Phase Models (6 more)
BGE_GEMMA2 = ZooModel(
    model_id="BAAI/bge-multilingual-gemma2",
    context_window=8192,
    embedding_dim=3584,
    output_mode="3d",
    description="8k context, multilingual, 9.2B params"
)

E5_MISTRAL = ZooModel(
    model_id="intfloat/e5-mistral-7b-instruct",
    context_window=32768,
    embedding_dim=4096,
    output_mode="3d",
    description="32k context, English, 7B params"
)

QWEN_8B = ZooModel(
    model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    context_window=32768,
    embedding_dim=4096,
    output_mode="3d",
    description="32k context, multilingual, 7B params"
)

DISTILUSE = ZooModel(
    model_id="sentence-transformers/distiluse-base-multilingual-cased-v2",
    context_window=512,
    embedding_dim=512,
    output_mode="2d",
    description="Distilled, 134M params"
)

BERT_MULTILINGUAL = ZooModel(
    model_id="bert-base-multilingual-cased",
    context_window=512,
    embedding_dim=768,
    output_mode="3d",
    description="Classic BERT, 104 languages, good for Dutch"
)

XLMR_BASE = ZooModel(
    model_id="xlm-roberta-base",
    context_window=512,
    embedding_dim=768,
    output_mode="3d",
    description="XLM-RoBERTa, 100 languages, 278M params"
)

# Model groups by use case
ESSENTIAL_MODELS = [MPNET, QWEN_0_6B, LABSE, MINILM]  # Core 4 for immediate use
EXTENDED_MODELS = [BGE_GEMMA2, E5_MISTRAL, QWEN_8B, DISTILUSE, BERT_MULTILINGUAL, XLMR_BASE]  # Additional 6 for specialized use
ALL_MODELS = ESSENTIAL_MODELS + EXTENDED_MODELS

# Legacy aliases for backward compatibility
MVP_MODELS = ESSENTIAL_MODELS
GROWTH_MODELS = EXTENDED_MODELS


class Models(Enum):
    """Enum of all curated models for autocomplete and IDE support.

    Usage:
        from vectormesh.zoo.models import Models

        # IDE autocomplete works!
        model = Models.MPNET.value
        vectorizer = TwoDVectorizer(model_name=model.model_id)
    """

    # MVP Models
    MPNET = MPNET
    QWEN_0_6B = QWEN_0_6B
    LABSE = LABSE
    MINILM = MINILM

    # Growth Models
    BGE_GEMMA2 = BGE_GEMMA2
    E5_MISTRAL = E5_MISTRAL
    QWEN_8B = QWEN_8B
    DISTILUSE = DISTILUSE
    BERT_MULTILINGUAL = BERT_MULTILINGUAL
    XLMR_BASE = XLMR_BASE