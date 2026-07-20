"""Add a chunk-aligned regex column to an existing embedding cache.

Loads an embedding cache (like train_moe.py does), reads the chunking contract
(model_tag, context_size, stride) straight from that cache's metadata, fits a
ChunkedRegexVectorizer with the exact same chunking, and writes a new cache that
contains both the embeddings and the chunk-aligned regex matrices.

Run:  uv run python scripts/add_chunked_regex.py
"""

from pathlib import Path

from loguru import logger

from vectormesh.data.cache import VectorCache
from vectormesh.data.vectorizers import (
    ChunkedRegexVectorizer,
    build_legal_reference_pattern,
    harmonize_legal_reference,
)

EMB_COLUMN = "legal_dutch"  # the embedding column whose chunks we align to
REGEX_COLUMN = "chunked_regex"
# Fallback only: older caches were written before 'stride' was stored in metadata.
# New caches store it, so this is just a safety net matching Vectorizer.STRIDE_DIVISOR.
STRIDE_DIVISOR = 10


def find_embedding_cache(artefacts: Path, split: str) -> Path:
    """Pick the embedding cache for a split, skipping any regex-extended caches."""
    candidates = sorted(
        p
        for p in artefacts.glob(f"*bert*{split}*")
        if p.is_dir() and "regex" not in p.name
    )
    if not candidates:
        raise FileNotFoundError(f"No *bert*{split}* embedding cache in {artefacts}")
    return candidates[0]


def read_alignment(metadata: dict, column: str) -> tuple[str, int, int]:
    """Extract the (model_tag, context_size, stride) chunking contract."""
    col_meta = metadata[column]
    model_tag = col_meta["model_tag"]
    context_size = col_meta["context_size"]
    stride = col_meta.get("stride")
    if stride is None:
        stride = context_size // STRIDE_DIVISOR
        logger.warning(
            f"No 'stride' in metadata for '{column}' (cache predates stride storage); "
            f"falling back to context_size // {STRIDE_DIVISOR} = {stride}."
        )
    logger.info(
        f"Aligning chunked regex to model='{model_tag}', "
        f"context_size={context_size}, stride={stride}"
    )
    return model_tag, context_size, stride


def main():
    artefacts = Path("artefacts")
    trainpath = find_embedding_cache(artefacts, "train")
    validpath = find_embedding_cache(artefacts, "valid")
    logger.info(f"Train embedding cache: {trainpath}")
    logger.info(f"Valid embedding cache: {validpath}")

    traincache = VectorCache.load(path=trainpath)
    validcache = VectorCache.load(path=validpath)
    assert traincache.metadata is not None

    model_tag, context_size, stride = read_alignment(traincache.metadata, EMB_COLUMN)

    # Fit on all training texts (corpus-level feature selection), then apply per chunk.
    vectorizer = ChunkedRegexVectorizer(
        col_name=REGEX_COLUMN,
        tokenizer_name=model_tag,
        context_size=context_size,
        stride=stride,
        pattern_builder=build_legal_reference_pattern,
        harmonizer=harmonize_legal_reference,
        min_doc_frequency=15,
        max_features=200,
        device="cpu",
        training_texts=traincache["text"],
    )

    for cache, source in [(traincache, trainpath), (validcache, validpath)]:
        assert cache.dataset is not None
        VectorCache.create(
            cache_dir=artefacts,
            vectorizer=vectorizer,
            dataset=cache.dataset,  # already holds the embedding column
            dataset_tag=source.name,  # merges the source cache's metadata
        )

    logger.success(
        f"Added '{REGEX_COLUMN}' column. Train with scripts/train_moe_parallel.py"
    )


if __name__ == "__main__":
    main()
