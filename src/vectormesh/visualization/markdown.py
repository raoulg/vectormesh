"""Visualization utilities for VectorMesh component graphs.

This module provides functions to visualize VectorMesh component pipelines with
professional mathematical notation, showing component topology and tensor shapes.
"""

from typing import Any


def visualize(component: Any) -> str:
    """Visualize component graph with mathematical notation.

    Renders a text representation of the component pipeline showing:
    - Component names and parameters
    - Tensor flow with mathematical notation (ℝ, Σ*)
    - Dimension transformations
    - Hierarchical structure for nested combinators

    Args:
        component: VectorMeshComponent or combinator to visualize
                   (Serial, Parallel, or single component)

    Returns:
        String representation with Unicode mathematical symbols

    Example:
        >>> pipeline = Serial([TwoDVectorizer("bert"), MeanAggregator()])
        >>> print(visualize(pipeline))
        TwoDVectorizer("bert") → ℝ^{B×768}
        MeanAggregator() → ℝ^{B×768}

    Shapes:
        N/A - utility function, no tensor processing
    """
    # Import here to avoid circular imports
    from vectormesh.components.combinators import Serial, Parallel

    # Detect component type and route to appropriate visualizer
    if isinstance(component, Serial):
        return _visualize_serial(component)
    elif isinstance(component, Parallel):
        return _visualize_parallel(component)
    else:
        return _visualize_single(component)


def _visualize_serial(serial: Any) -> str:
    """Render Serial combinator as linear chain.

    Args:
        serial: Serial combinator instance

    Returns:
        String showing sequential flow with arrows
    """
    from vectormesh.components.combinators import Serial, Parallel

    components = serial.components
    if not components:
        return "Serial[]  # Empty pipeline"

    lines = []
    for i, comp in enumerate(components):
        # Check if component is nested combinator
        if isinstance(comp, (Serial, Parallel)):
            # Recursively visualize nested combinator
            nested = visualize(comp)
            # Indent nested visualization
            nested_lines = nested.split("\n")
            for line in nested_lines:
                lines.append(f"  {line}")
        else:
            # Single component visualization
            comp_str = _format_component(comp)
            shape_str = _infer_shape(comp)
            lines.append(f"{comp_str} → {shape_str}")

    return "\n".join(lines)


def _visualize_parallel(parallel: Any) -> str:
    """Render Parallel combinator as tree structure.

    Args:
        parallel: Parallel combinator instance

    Returns:
        String showing branching topology with tree characters
    """
    from vectormesh.components.combinators import Serial, Parallel

    branches = parallel.branches
    if not branches:
        return "Parallel[]  # No branches"

    lines = []
    lines.append("Input: Σ*")  # Text input to parallel branches

    for i, branch in enumerate(branches):
        # Determine tree character
        if i < len(branches) - 1:
            prefix = "├── "
        else:
            prefix = "└── "

        # Check if branch is nested combinator
        if isinstance(branch, (Serial, Parallel)):
            # Nested combinator - show type and recurse
            branch_type = type(branch).__name__
            lines.append(f"{prefix}{branch_type}[")
            nested = visualize(branch)
            nested_lines = nested.split("\n")
            for line in nested_lines:
                lines.append(f"    {line}")
            lines.append("  ]")
        else:
            # Single component
            comp_str = _format_component(branch)
            shape_str = _infer_shape(branch)
            lines.append(f"{prefix}{comp_str} → {shape_str}")

    # Show tuple output
    output_shapes = [_infer_shape(b) for b in branches]
    lines.append(f"Output: ({', '.join(output_shapes)})")

    return "\n".join(lines)


def _visualize_single(component: Any) -> str:
    """Render single component with shape inference.

    Args:
        component: Single component instance

    Returns:
        String showing component and output shape
    """
    comp_str = _format_component(component)
    shape_str = _infer_shape(component)
    return f"{comp_str} → {shape_str}"


def _format_component(component: Any) -> str:
    """Extract component name with parameters.

    Args:
        component: Component instance

    Returns:
        Formatted string like "TwoDVectorizer(\"bert-base\")" or "GlobalConcat(dim=1)"
    """
    comp_type = type(component).__name__

    # Try to extract common parameters
    params = []

    # Check for model_name (vectorizers)
    if hasattr(component, 'model_name'):
        params.append(f'"{component.model_name}"')

    # Check for dim (connectors)
    if hasattr(component, 'dim'):
        params.append(f"dim={component.dim}")

    # Format with parameters
    if params:
        return f"{comp_type}({', '.join(params)})"
    else:
        return f"{comp_type}()"


def _infer_shape(component: Any) -> str:
    """Infer mathematical notation for output shape.

    Attempts to determine the output tensor shape from component metadata
    and returns a LaTeX-style mathematical notation.

    Args:
        component: Component instance

    Returns:
        String like "ℝ^{B×768}" or "ℝ^{B×C×384}"
    """
    from vectormesh.components.combinators import Serial, Parallel

    # Handle combinators
    if isinstance(component, Serial):
        # Serial output is output of last component
        if component.components:
            return _infer_shape(component.components[-1])
        return "ℝ^{B×E}"

    if isinstance(component, Parallel):
        # Parallel output is tuple - handled by caller
        # Return generic for single branch
        if component.branches:
            shapes = [_infer_shape(b) for b in component.branches]
            return f"({', '.join(shapes)})"
        return "()"

    # Try to get embedding dimension
    embedding_dim = None
    if hasattr(component, 'embedding_dim'):
        embedding_dim = component.embedding_dim
    elif hasattr(component, 'output_dim'):
        embedding_dim = component.output_dim

    # Try to get output mode (2d vs 3d)
    output_mode = None
    if hasattr(component, 'output_mode'):
        output_mode = component.output_mode

    # Construct mathematical notation
    # ℝ (U+211D) for real tensors
    # × (U+00D7) for dimension separator

    if output_mode == "3d":
        # Three-dimensional: ℝ^{B×C×E}
        if embedding_dim:
            return f"ℝ^{{B×C×{embedding_dim}}}"
        else:
            return "ℝ^{B×C×E}"
    elif output_mode == "2d" or embedding_dim:
        # Two-dimensional: ℝ^{B×E}
        if embedding_dim:
            return f"ℝ^{{B×{embedding_dim}}}"
        else:
            return "ℝ^{B×E}"
    else:
        # Unknown - generic notation
        return "ℝ^{B×E}"
