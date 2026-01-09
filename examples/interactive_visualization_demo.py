"""Interactive Category Theory Visualization Demo.

This demo showcases the new interactive visualization system with:
- Category theory diagrams (nodes = domains, arrows = morphisms)
- MathJax LaTeX rendering (Σ*, ℝ^{B×D})
- Hover tooltips with implementation details
- Multiple interaction modes (VIEW, DEBUG, LEARN)
- Professional color scheme
"""

import torch
from vectormesh.visualization import (
    visualize,
    VisualizationFormat,
    VisualizationMode,
    InteractionMode,
)
from vectormesh import Serial, Parallel, GlobalConcat
from vectormesh.validation import register_morphism, Morphism, TensorDimensionality


# Mock components for demo
class MockVectorizer:
    """Mock text vectorizer."""

    def __init__(self, model_name: str, embedding_dim: int = 768):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.output_mode = "2d"

    def __call__(self, texts):
        batch_size = len(texts) if isinstance(texts, list) else 1
        return torch.randn(batch_size, self.embedding_dim)


class MockProcessor:
    """Mock processing layer."""

    def __init__(self, output_dim: int):
        self.output_dim = output_dim

    def __call__(self, tensor):
        batch_size = tensor.shape[0]
        return torch.randn(batch_size, self.output_dim)


# Register morphisms
register_morphism(
    MockVectorizer,
    Morphism(
        source=TensorDimensionality.TEXT,
        target=TensorDimensionality.TWO_D,
        component_name="MockVectorizer",
        description="Mock vectorizer (TEXT→2D)",
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


def demo_simple_serial():
    """Demo 1: Simple Serial Pipeline - Interactive."""
    print("\n" + "=" * 80)
    print("Demo 1: Simple Serial Pipeline (Interactive)")
    print("=" * 80)

    pipeline = Serial(
        components=[
            MockVectorizer("bert-base-uncased", 768),
            MockProcessor(256),
            MockProcessor(128),
        ]
    )

    # Generate interactive diagram
    diagram = visualize(
        pipeline,
        format=VisualizationFormat.INTERACTIVE,
        mode=VisualizationMode.HYBRID,
        interaction=InteractionMode.VIEW,
    )

    # Save to file
    diagram.save("demo1_serial.html")
    print("\n✓ Saved to: demo1_serial.html")
    print("  Features:")
    print("  - Nodes: Σ*, ℝ^{B×768}, ℝ^{B×256}, ℝ^{B×128}")
    print("  - Arrows: f, g, h (morphisms)")
    print("  - Hover on arrows to see component details!")


def demo_parallel_ensemble():
    """Demo 2: Parallel Ensemble with Multiple Models."""
    print("\n" + "=" * 80)
    print("Demo 2: Parallel Ensemble (Fan-out Structure)")
    print("=" * 80)

    pipeline = Parallel(
        branches=[
            MockVectorizer("bert-base-uncased", 768),
            MockVectorizer("roberta-base", 768),
            MockVectorizer("distilbert-base-uncased", 768),
        ]
    )

    diagram = visualize(
        pipeline,
        format=VisualizationFormat.INTERACTIVE,
        mode=VisualizationMode.HYBRID,
        interaction=InteractionMode.VIEW,
    )

    diagram.save("demo2_parallel.html")
    print("\n✓ Saved to: demo2_parallel.html")
    print("  Features:")
    print("  - Vertical fan-out from Σ* source")
    print("  - Three parallel morphisms (f₁, f₂, f₃)")
    print("  - Product structure in category theory!")


def demo_feature_fusion():
    """Demo 3: Parallel + GlobalConcat (Feature Fusion)."""
    print("\n" + "=" * 80)
    print("Demo 3: Feature Fusion (Coproduct Morphism)")
    print("=" * 80)

    pipeline = Serial(
        components=[
            Parallel(
                branches=[
                    MockVectorizer("bert-base-uncased", 768),
                    MockVectorizer("sentence-transformers/all-MiniLM-L6-v2", 384),
                ]
            ),
            GlobalConcat(dim=1),
            MockProcessor(512),
        ]
    )

    diagram = visualize(
        pipeline,
        format=VisualizationFormat.INTERACTIVE,
        mode=VisualizationMode.HYBRID,
        interaction=InteractionMode.VIEW,
    )

    diagram.save("demo3_fusion.html")
    print("\n✓ Saved to: demo3_fusion.html")
    print("  Features:")
    print("  - Parallel fan-out + concatenation")
    print("  - Coproduct → morphism composition")
    print("  - Shows full pipeline structure!")


def demo_learn_mode():
    """Demo 4: LEARN Mode with Category Theory Explanations."""
    print("\n" + "=" * 80)
    print("Demo 4: LEARN Mode (Category Theory Guide)")
    print("=" * 80)

    pipeline = Serial(
        components=[
            MockVectorizer("bert-base-uncased", 768),
            MockProcessor(256),
        ]
    )

    diagram = visualize(
        pipeline,
        format=VisualizationFormat.INTERACTIVE,
        mode=VisualizationMode.HYBRID,
        interaction=InteractionMode.LEARN,  # <-- LEARN mode!
    )

    diagram.save("demo4_learn.html")
    print("\n✓ Saved to: demo4_learn.html")
    print("  Features:")
    print("  - CT legend in top-right corner")
    print("  - Hover tooltips explain morphisms, composition")
    print("  - Perfect for teaching category theory!")


def demo_debug_mode():
    """Demo 5: DEBUG Mode with Code Snippets."""
    print("\n" + "=" * 80)
    print("Demo 5: DEBUG Mode (Code Location & Snippets)")
    print("=" * 80)

    pipeline = Serial(
        components=[
            MockVectorizer("roberta-base", 768),
            MockProcessor(128),
        ]
    )

    diagram = visualize(
        pipeline,
        format=VisualizationFormat.INTERACTIVE,
        mode=VisualizationMode.HYBRID,
        interaction=InteractionMode.DEBUG,  # <-- DEBUG mode!
    )

    diagram.save("demo5_debug.html")
    print("\n✓ Saved to: demo5_debug.html")
    print("  Features:")
    print("  - Shows source file & line number")
    print("  - Code snippets in tooltips")
    print("  - 'Jump to code' if $EDITOR is set!")


def main():
    """Run all interactive visualization demos."""
    print("\n" + "╔" + "=" * 78 + "╗")
    print(
        "║"
        + " " * 10
        + "VectorMesh Interactive Category Theory Diagrams"
        + " " * 20
        + "║"
    )
    print("║" + " " * 15 + "Professional Mathematical Visualization" + " " * 23 + "║")
    print("╚" + "=" * 78 + "╝")

    print("\n🎨 Generating interactive diagrams with:")
    print("   • Category theory layout (nodes = domains, arrows = morphisms)")
    print("   • MathJax LaTeX rendering (Σ*, ℝ^{B×D})")
    print("   • Curved edges for elegant morphism display")
    print("   • Hover tooltips with implementation details")
    print("   • Professional color scheme")

    demo_simple_serial()
    demo_parallel_ensemble()
    demo_feature_fusion()
    demo_learn_mode()
    demo_debug_mode()

    print("\n" + "=" * 80)
    print("✓ All demos complete!")
    print("=" * 80)
    print("\n📂 Output files:")
    print("   • demo1_serial.html      - Simple serial pipeline")
    print("   • demo2_parallel.html    - Parallel ensemble")
    print("   • demo3_fusion.html      - Feature fusion architecture")
    print("   • demo4_learn.html       - LEARN mode with CT guide")
    print("   • demo5_debug.html       - DEBUG mode with code links")
    print("\n💡 Open any HTML file in your browser to explore!")
    print("   Hover over arrows to see morphism details")
    print("   Hover over nodes to see domain information")
    print("   In LEARN mode, see the CT legend in top-right\n")


if __name__ == "__main__":
    main()
