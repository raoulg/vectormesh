"""Visualization system for VectorMesh component graphs.

Supports multiple output formats:
- Markdown (text with Unicode symbols)
- Interactive (HTML with category theory diagrams)
- SVG (static vector graphics)
- JSON (structured data export)
"""

from typing import Any, Optional

from .types import VisualizationFormat, VisualizationMode, InteractionMode
from .graph_builder import CategoryGraphBuilder
from .layout_engine import CTLayoutEngine
from .renderers.interactive import InteractiveRenderer

# Import legacy markdown visualizer
from .markdown import visualize as visualize_markdown


class InteractiveDiagram:
    """Interactive diagram result that can be displayed or saved."""

    def __init__(self, html: str):
        """Initialize with HTML content."""
        self.html = html

    def save(self, filename: str) -> None:
        """Save diagram to HTML file."""
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.html)
        print(f"Saved interactive diagram to: {filename}")

    def show(self) -> None:
        """Open diagram in default browser."""
        import tempfile
        import webbrowser

        # Save to temporary file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".html",
            delete=False,
            encoding="utf-8"
        ) as f:
            f.write(self.html)
            temp_path = f.name

        # Open in browser
        webbrowser.open(f"file://{temp_path}")
        print(f"Opened diagram in browser: {temp_path}")

    def _repr_html_(self) -> str:
        """Jupyter notebook integration."""
        return self.html

    def __str__(self) -> str:
        """String representation."""
        return f"<InteractiveDiagram: {len(self.html)} bytes>"


def visualize(
    component: Any,
    format: VisualizationFormat = VisualizationFormat.MARKDOWN,
    mode: VisualizationMode = VisualizationMode.HYBRID,
    interaction: InteractionMode = InteractionMode.VIEW,
    output: Optional[str] = None,
):
    """Visualize VectorMesh component with category theory diagrams.

    Args:
        component: VectorMesh component or combinator to visualize
        format: Output format (MARKDOWN, INTERACTIVE, SVG, JSON)
        mode: Detail level (PRACTICAL, THEORETICAL, HYBRID)
        interaction: Interactive mode (VIEW, DEBUG, LEARN, COMPARE)
        output: Optional output file path

    Returns:
        - str: For MARKDOWN, SVG, JSON formats
        - InteractiveDiagram: For INTERACTIVE format

    Examples:
        >>> # Simple markdown (backward compatible)
        >>> print(visualize(pipeline))

        >>> # Interactive diagram
        >>> diagram = visualize(pipeline, format=VisualizationFormat.INTERACTIVE)
        >>> diagram.show()  # Opens in browser
        >>> diagram.save("pipeline.html")

        >>> # Learning mode with CT explanations
        >>> diagram = visualize(
        ...     pipeline,
        ...     format=VisualizationFormat.INTERACTIVE,
        ...     interaction=InteractionMode.LEARN
        ... )

        >>> # Debug mode with code snippets
        >>> diagram = visualize(
        ...     pipeline,
        ...     format=VisualizationFormat.INTERACTIVE,
        ...     interaction=InteractionMode.DEBUG
        ... )
    """
    if format == VisualizationFormat.MARKDOWN:
        # Use legacy markdown visualizer
        result = visualize_markdown(component)
        if output:
            with open(output, "w") as f:
                f.write(result)
        return result

    elif format == VisualizationFormat.INTERACTIVE:
        # Build category graph
        builder = CategoryGraphBuilder()
        graph = builder.build(component)

        # Compute layout
        layout_engine = CTLayoutEngine()
        positions = layout_engine.compute_layout(graph)

        # Render to interactive HTML
        renderer = InteractiveRenderer(mode=mode, interaction=interaction)
        html = renderer.render(graph, positions, output_file=output)

        return InteractiveDiagram(html)

    elif format == VisualizationFormat.SVG:
        raise NotImplementedError("SVG format not yet implemented")

    elif format == VisualizationFormat.JSON:
        raise NotImplementedError("JSON format not yet implemented")

    else:
        raise ValueError(f"Unknown format: {format}")


# Export all public API
__all__ = [
    "visualize",
    "InteractiveDiagram",
    "VisualizationFormat",
    "VisualizationMode",
    "InteractionMode",
]
