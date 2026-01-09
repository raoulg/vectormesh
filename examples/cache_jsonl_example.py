"""Example: Creating a VectorCache from JSONL data.

This example demonstrates how to:
1. Load JSONL data (like assets/train.jsonl)
2. Extract text fields
3. Create a VectorCache
4. Load and use the cache
"""

import json
from pathlib import Path

from vectormesh import TwoDVectorizer, VectorCache


def load_jsonl_texts(file_path: Path, limit: int = None) -> list[str]:
    """Load text fields from JSONL file.

    Args:
        file_path: Path to JSONL file
        limit: Optional limit on number of records

    Returns:
        List of text strings
    """
    texts = []
    with open(file_path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            record = json.loads(line)
            texts.append(record["text"])
    return texts


def main():
    """Main example workflow."""
    # 1. Load JSONL data
    jsonl_path = Path("assets/train.jsonl")
    print(f"Loading texts from {jsonl_path}...")

    # Load first 100 texts for demo (remove limit for full dataset)
    texts = load_jsonl_texts(jsonl_path, limit=100)
    print(f"Loaded {len(texts)} texts")

    # 2. Create vectorizer
    print("\nInitializing vectorizer...")
    vectorizer = TwoDVectorizer(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Create cache
    print("\nCreating vector cache...")
    cache = VectorCache.create(
        texts=texts,
        vectorizer=vectorizer,
        name="train_cache",
        batch_size=32,  # Process 32 texts at a time
    )
    print(f"Cache created: {cache.name}")

    # 4. Check metadata
    metadata = cache.get_metadata()
    print("\nCache metadata:")
    print(f"  Model: {metadata['model_name']}")
    print(f"  Samples: {metadata['num_samples']}")
    print(f"  Embedding dim: {metadata['embedding_dim']}")
    print(f"  Created: {metadata['created_at']}")

    # 5. Get embeddings
    embeddings = cache.get_embeddings()
    print(f"\nEmbeddings shape: {embeddings.shape}")

    # 6. Get specific embeddings
    first_10 = cache.get_embeddings(indices=list(range(10)))
    print(f"First 10 embeddings shape: {first_10.shape}")

    # 7. Load cache later (from disk)
    print("\n--- Loading cache from disk ---")
    loaded_cache = VectorCache.load(name="train_cache")
    loaded_embeddings = loaded_cache.get_embeddings()
    print(f"Loaded embeddings shape: {loaded_embeddings.shape}")

    print("\nDone! Cache is stored in .vmcache/train_cache")


if __name__ == "__main__":
    main()
