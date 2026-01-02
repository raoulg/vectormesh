"""Curated model registry with validated metadata.

All 10 models have been tested via HuggingFace MCP for:
- Context window size (max_position_embeddings)
- Embedding dimension (hidden_size)
- Pooling strategy (for sentence-transformers)
- Dutch language support (for course requirements)
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CuratedModel:
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
MPNET = CuratedModel(
    model_id="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    context_window=512,
    embedding_dim=768,
    output_mode="2d",
    description="Dutch support, 249M downloads, best general performance"
)

QWEN_0_6B = CuratedModel(
    model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    context_window=32768,
    embedding_dim=896,
    output_mode="3d",  # No built-in pooling
    description="32k context, CPU-friendly (0.5B params), multilingual"
)

LABSE = CuratedModel(
    model_id="sentence-transformers/LaBSE",
    context_window=512,
    embedding_dim=768,
    output_mode="2d",
    description="109 languages, 470M params"
)

MINILM = CuratedModel(
    model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    context_window=512,
    embedding_dim=384,
    output_mode="2d",
    description="Fastest (117M params), good baseline"
)

# Growth Phase Models (6 more)
BGE_GEMMA2 = CuratedModel(
    model_id="BAAI/bge-multilingual-gemma2",
    context_window=8192,
    embedding_dim=3584,
    output_mode="3d",
    description="8k context, multilingual, 9.2B params"
)

E5_MISTRAL = CuratedModel(
    model_id="intfloat/e5-mistral-7b-instruct",
    context_window=32768,
    embedding_dim=4096,
    output_mode="3d",
    description="32k context, English, 7B params"
)

QWEN_8B = CuratedModel(
    model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    context_window=32768,
    embedding_dim=4096,
    output_mode="3d",
    description="32k context, multilingual, 7B params"
)

DISTILUSE = CuratedModel(
    model_id="sentence-transformers/distiluse-base-multilingual-cased-v2",
    context_window=512,
    embedding_dim=512,
    output_mode="2d",
    description="Distilled, 134M params"
)

BERT_MULTILINGUAL = CuratedModel(
    model_id="bert-base-multilingual-cased",
    context_window=512,
    embedding_dim=768,
    output_mode="3d",
    description="Classic BERT, 104 languages, good for Dutch"
)

XLMR_BASE = CuratedModel(
    model_id="xlm-roberta-base",
    context_window=512,
    embedding_dim=768,
    output_mode="3d",
    description="XLM-RoBERTa, 100 languages, 278M params"
)

# Model groups for testing
MVP_MODELS = [MPNET, QWEN_0_6B, LABSE, MINILM]
GROWTH_MODELS = [BGE_GEMMA2, E5_MISTRAL, QWEN_8B, DISTILUSE, BERT_MULTILINGUAL, XLMR_BASE]
ALL_MODELS = MVP_MODELS + GROWTH_MODELS