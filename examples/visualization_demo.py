"""Visualization Demo: Showcasing VectorMesh Pipeline Architectures.

This demo shows how the visualize() function renders different pipeline architectures
with professional mathematical notation.
"""

import torch
from vectormesh import (
    Serial,
    Parallel,
    GlobalConcat,
    GlobalStack,
    visualize,
)
from vectormesh.components.gating import Skip, Gate
from vectormesh.types import NDTensor


# =============================================================================
# Mock Components for Demo
# =============================================================================


class MockVectorizer:
    """Mock text vectorizer for demo."""

    def __init__(self, model_name: str, embedding_dim: int = 768):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.output_mode = "2d"

    def __call__(self, texts):
        batch_size = len(texts) if isinstance(texts, list) else 1
        return torch.randn(batch_size, self.embedding_dim)


class Mock3DVectorizer:
    """Mock 3D vectorizer for demo."""

    def __init__(self, model_name: str, embedding_dim: int = 768, chunks: int = 3):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.chunks = chunks
        self.output_mode = "3d"

    def __call__(self, texts):
        batch_size = len(texts) if isinstance(texts, list) else 1
        return torch.randn(batch_size, self.chunks, self.embedding_dim)


class MockAggregator:
    """Mock aggregator for demo."""

    def __init__(self, method: str = "mean"):
        self.method = method
        self.embedding_dim = None  # Will match input

    def __call__(self, tensor_3d):
        if tensor_3d.dim() == 3:
            if self.method == "mean":
                return torch.mean(tensor_3d, dim=1)
            else:
                return torch.max(tensor_3d, dim=1)[0]
        return tensor_3d


class MockProcessor:
    """Mock processing layer for demo."""

    def __init__(self, output_dim: int):
        self.output_dim = output_dim

    def __call__(self, tensor):
        batch_size = tensor.shape[0]
        return torch.randn(batch_size, self.output_dim)


# =============================================================================
# Register mock components in morphism system
# =============================================================================

from vectormesh.validation import register_morphism, Morphism, TensorDimensionality

register_morphism(
    MockVectorizer,
    Morphism(
        source=TensorDimensionality.TEXT,
        target=TensorDimensionality.TWO_D,
        component_name="MockVectorizer",
        description="Mock 2D vectorizer (TEXT→2D)",
    ),
)

register_morphism(
    Mock3DVectorizer,
    Morphism(
        source=TensorDimensionality.TEXT,
        target=TensorDimensionality.THREE_D,
        component_name="Mock3DVectorizer",
        description="Mock 3D vectorizer (TEXT→3D)",
    ),
)

register_morphism(
    MockAggregator,
    Morphism(
        source=TensorDimensionality.THREE_D,
        target=TensorDimensionality.TWO_D,
        component_name="MockAggregator",
        description="Mock aggregator (3D→2D)",
    ),
)

register_morphism(
    MockProcessor,
    Morphism(
        source=TensorDimensionality.TWO_D,
        target=TensorDimensionality.TWO_D,
        component_name="MockProcessor",
        description="Mock processor (2D→2D)",
    ),
)

register_morphism(
    Skip,
    Morphism(
        source=TensorDimensionality.TWO_D,
        target=TensorDimensionality.TWO_D,
        component_name="Skip",
        description="Residual skip connection (2D→2D)",
    ),
)

register_morphism(
    Gate,
    Morphism(
        source=TensorDimensionality.TWO_D,
        target=TensorDimensionality.TWO_D,
        component_name="Gate",
        description="Gating mechanism (2D→2D)",
    ),
)


# =============================================================================
# Demo Architectures
# =============================================================================


def print_separator(title: str):
    """Print a section separator."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_simple_serial():
    """Demo 1: Simple Serial Pipeline."""
    print_separator("Demo 1: Simple Serial Pipeline")

    pipeline = Serial(
        components=[
            MockVectorizer("sentence-transformers/all-MiniLM-L6-v2", 384),
            MockProcessor(128),
            MockProcessor(64),
        ]
    )

    print("Architecture: Single embedding → dimension reduction")
    print("\nVisualization:")
    print(visualize(pipeline))


def demo_parallel_ensemble():
    """Demo 2: Parallel Ensemble (Multiple Models)."""
    print_separator("Demo 2: Parallel Ensemble - Multi-Model Encoding")

    pipeline = Parallel(
        branches=[
            MockVectorizer("bert-base-uncased", 768),
            MockVectorizer("roberta-base", 768),
            MockVectorizer("sentence-transformers/all-MiniLM-L6-v2", 384),
        ]
    )

    print("Architecture: Multiple models encode text in parallel")
    print("\nVisualization:")
    print(visualize(pipeline))


def demo_parallel_with_concat():
    """Demo 3: Parallel + GlobalConcat (Feature Fusion)."""
    print_separator("Demo 3: Parallel + GlobalConcat - Feature Fusion")

    pipeline = Serial(
        components=[
            Parallel(
                branches=[
                    MockVectorizer("bert-base-uncased", 768),
                    MockVectorizer("sentence-transformers/all-MiniLM-L6-v2", 384),
                ]
            ),
            GlobalConcat(dim=1),
            MockProcessor(256),
        ]
    )

    print("Architecture: Multi-model → concatenate features → process")
    print("\nVisualization:")
    print(visualize(pipeline))


def demo_3d_aggregation():
    """Demo 4: 3D Vectorization + Aggregation."""
    print_separator("Demo 4: 3D Vectorization + Aggregation")

    pipeline = Serial(
        components=[
            Mock3DVectorizer("mock-3d-model", 768, chunks=5),
            MockAggregator("mean"),
            MockProcessor(256),
        ]
    )

    print("Architecture: 3D chunked embedding → aggregate → process")
    print("\nVisualization:")
    print(visualize(pipeline))


def demo_skip_connection():
    """Demo 5: Skip Connection (Residual Network)."""
    print_separator("Demo 5: Skip Connection - Residual Pattern")

    pipeline = Serial(
        components=[
            MockVectorizer("bert-base-uncased", 768),
            Skip(main=MockProcessor(768)),  # Keep same dimension for residual
            MockProcessor(256),
        ]
    )

    print("Architecture: Encode → skip connection → final processing")
    print("\nVisualization:")
    print(visualize(pipeline))


def demo_gated_pathway():
    """Demo 6: Gated Pathway."""
    print_separator("Demo 6: Gated Pathway - Conditional Processing")

    def adaptive_gate(x: NDTensor) -> float:
        """Router that computes gate value from input."""
        return 0.7  # In practice, this would be learned or computed

    pipeline = Serial(
        components=[
            MockVectorizer("roberta-base", 768),
            Gate(component=MockProcessor(768), router=adaptive_gate),
            MockProcessor(256),
        ]
    )

    print("Architecture: Encode → gated processing → final layer")
    print("\nVisualization:")
    print(visualize(pipeline))


def demo_complex_nested():
    """Demo 7: Complex Nested Architecture."""
    print_separator("Demo 7: Complex Nested - Multi-Level Pipeline")

    # Branch 1: 3D → aggregated
    branch1 = Serial(
        components=[
            Mock3DVectorizer("longformer-base", 768, chunks=8),
            MockAggregator("mean"),
        ]
    )

    # Branch 2: Simple 2D
    branch2 = MockVectorizer("sentence-transformers/all-mpnet-base-v2", 768)

    # Main pipeline: Parallel branches → concat → process
    pipeline = Serial(
        components=[
            Parallel(branches=[branch1, branch2]),
            GlobalConcat(dim=1),
            Skip(main=MockProcessor(1536)),  # 768 + 768 = 1536
            MockProcessor(512),
            MockProcessor(128),
        ]
    )

    print("Architecture: Multi-branch (3D + 2D) → fusion → residual → reduction")
    print("\nVisualization:")
    print(visualize(pipeline))


def demo_hierarchical_ensemble():
    """Demo 8: Hierarchical Ensemble with Stacking."""
    print_separator("Demo 8: Hierarchical Ensemble - Model Stacking")

    # Create parallel ensemble
    ensemble = Parallel(
        branches=[
            MockVectorizer("bert-base-uncased", 768),
            MockVectorizer("roberta-base", 768),
            MockVectorizer("distilbert-base-uncased", 768),
        ]
    )

    # Stack embeddings and process
    pipeline = Serial(
        components=[
            ensemble,
            GlobalStack(dim=1),
            MockAggregator("mean"),  # Aggregate across stacked models
            MockProcessor(256),
        ]
    )

    print("Architecture: Multi-model → stack → aggregate → process")
    print("\nVisualization:")
    print(visualize(pipeline))


def demo_dual_path_fusion():
    """Demo 9: Dual-Path Feature Fusion."""
    print_separator("Demo 9: Dual-Path Feature Fusion")

    # Semantic path
    semantic_path = Serial(
        components=[
            MockVectorizer("sentence-transformers/all-mpnet-base-v2", 768),
            MockProcessor(512),
        ]
    )

    # Syntactic path (3D with aggregation)
    syntactic_path = Serial(
        components=[
            Mock3DVectorizer("mock-syntactic-model", 384, chunks=10),
            MockAggregator("max"),
            MockProcessor(512),
        ]
    )

    # Fusion pipeline
    pipeline = Serial(
        components=[
            Parallel(branches=[semantic_path, syntactic_path]),
            GlobalConcat(dim=1),
            Skip(main=MockProcessor(1024)),
            MockProcessor(256),
        ]
    )

    print("Architecture: Dual processing paths → fusion → residual → output")
    print("\nVisualization:")
    print(visualize(pipeline))


# =============================================================================
# Main Demo Runner
# =============================================================================


def main():
    """Run all visualization demos."""
    print("\n" + "╔" + "=" * 78 + "╗")
    print(
        "║" + " " * 15 + "VectorMesh Architecture Visualization Demo" + " " * 20 + "║"
    )
    print("║" + " " * 20 + "Professional Mathematical Notation" + " " * 24 + "║")
    print("╚" + "=" * 78 + "╝")

    demos = [
        demo_simple_serial,
        demo_parallel_ensemble,
        demo_parallel_with_concat,
        demo_3d_aggregation,
        demo_skip_connection,
        demo_gated_pathway,
        demo_complex_nested,
        demo_hierarchical_ensemble,
        demo_dual_path_fusion,
    ]

    for demo in demos:
        demo()

    print("\n" + "=" * 80)
    print("  Demo Complete! 🎉")
    print("  All architectures rendered with mathematical notation (ℝ, Σ*, ×)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
