"""Example: Using TwoDVectorizer and ThreeDVectorizer explicitly.

This example demonstrates:
1. Using TwoDVectorizer for sentence-transformers (2D output)
2. Using ThreeDVectorizer for raw transformers (3D output)
3. Handling compatibility errors when mixing models
"""

import torch
from vectormesh import TwoDVectorizer, ThreeDVectorizer, VectorMeshError


def main():
    print("=== VectorMesh Vectorization Demo ===\n")

    texts = ["This is a short sentence.", "Here is another one for the batch."]

    # 1. 2D Vectorization (Sentence Transformers)
    print("--- 1. TwoDVectorizer (SentenceTransformer) ---")
    model_2d = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading {model_2d}...")

    vec_2d = TwoDVectorizer(model_name=model_2d)
    embeddings_2d = vec_2d(texts)

    print(f"Output type: {type(embeddings_2d)}")
    print(f"Shape: {embeddings_2d.shape} (batch, dim)")
    print("Success!\n")

    # 2. 3D Vectorization (Raw Transformers)
    print("--- 2. ThreeDVectorizer (Raw Transformer) ---")
    model_3d = "bert-base-uncased"
    print(f"Loading {model_3d}...")

    vec_3d = ThreeDVectorizer(model_name=model_3d)
    embeddings_3d = vec_3d(texts)

    print(f"Output type: {type(embeddings_3d)}")
    print(f"Shape: {embeddings_3d.shape} (batch, chunks, dim)")
    print("Success!\n")

    # 3. Error Handling Demo
    print("--- 3. Compatibility Checks (Error Handling) ---")

    # Case A: Using TwoDVectorizer with a 3D model
    print("Attempting to load a 3D model (BERT) into TwoDVectorizer...")
    try:
        TwoDVectorizer(model_name="bert-base-uncased")(["test"])  # Trigger lazy load
    except VectorMeshError as e:
        print(f"\n[Caught Expected Error]: {e}")
        print(f"Hint: {e.hint}")
        print(f"Fix:  {e.fix}\n")

    # Case B: Using ThreeDVectorizer with a 2D model
    print("Attempting to load a 2D model (MiniLM) into ThreeDVectorizer...")
    try:
        ThreeDVectorizer(model_name="sentence-transformers/all-MiniLM-L6-v2")(["test"])
    except VectorMeshError as e:
        print(f"\n[Caught Expected Error]: {e}")
        print(f"Hint: {e.hint}")
        print(f"Fix:  {e.fix}\n")

    print("=== Demo Complete ===")


if __name__ == "__main__":
    main()
