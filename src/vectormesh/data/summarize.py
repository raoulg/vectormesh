"""Pipeline visualization using jaxtyping type hints to show tensor shape flow."""

from typing import Optional, get_args, get_type_hints

import networkx as nx
import torch.nn as nn
from loguru import logger
from pyvis.network import Network

from vectormesh.components.pipelines import Parallel, Serial


def extract_shape_from_hint(hint) -> Optional[str]:
    """Extract shape string from jaxtyping Float[Tensor, "batch dim"] annotation.

    Args:
        hint: Type hint that may contain jaxtyping annotation

    Returns:
        Shape string like "batch dim" or None if not extractable
    """
    try:
        # Check if this is a jaxtyping annotation
        if hasattr(hint, "__origin__"):
            # For Float[Tensor, "batch dim"], get_args returns the Tensor and shape
            args = get_args(hint)
            if args and hasattr(args[0], "dim_str"):
                return args[0].dim_str
            # Try getting dim_str directly
            if hasattr(hint, "dim_str"):
                return hint.dim_str
    except Exception:
        pass
    return None


def get_component_shapes(component: nn.Module) -> tuple[Optional[str], Optional[str]]:
    """Extract input and output shapes from component's forward() type hints.

    Args:
        component: PyTorch module with jaxtyped forward method

    Returns:
        Tuple of (input_shape, output_shape) as strings, or None if not found
    """
    try:
        # Get type hints from forward method
        hints = get_type_hints(component.forward)

        # Extract input shape - could be from first parameter (after self)
        input_shape = None
        output_shape = None

        # Get parameter hints (skip 'self' and 'return')
        param_names = [name for name in hints.keys() if name not in ("self", "return")]
        if param_names:
            input_hint = hints[param_names[0]]
            input_shape = extract_shape_from_hint(input_hint)

        # Extract output shape from return type
        if "return" in hints:
            output_hint = hints["return"]
            output_shape = extract_shape_from_hint(output_hint)

        return input_shape, output_shape
    except Exception as e:
        logger.debug(
            f"Could not extract shapes from {component.__class__.__name__}: {e}"
        )
        return None, None


def build_serial_graph(
    components: list, graph: nx.DiGraph, parent_id: Optional[str] = None
) -> tuple[str, str]:
    """Build graph nodes for Serial pipeline (left to right flow).

    Args:
        components: List of components in serial
        graph: NetworkX directed graph to add nodes to
        parent_id: ID of parent node to connect from

    Returns:
        Tuple of (first_node_id, last_node_id) for this serial chain
    """
    prev_node = parent_id
    first_node = None

    for idx, component in enumerate(components):
        component_name = component.__class__.__name__
        node_id = f"{id(component)}_{idx}"

        # Get shapes for this component
        input_shape, output_shape = get_component_shapes(component)

        # Create label with shapes
        if input_shape and output_shape:
            label = f"{component_name}\n({input_shape})\n→\n({output_shape})"
        elif output_shape:
            label = f"{component_name}\n→ ({output_shape})"
        else:
            label = component_name

        # Add node
        graph.add_node(node_id, label=label, component=component_name, level=idx)

        if first_node is None:
            first_node = node_id

        # Connect to previous node
        if prev_node is not None:
            # Add edge with shape annotation
            edge_label = f"({input_shape})" if input_shape else ""
            graph.add_edge(prev_node, node_id, label=edge_label)

        prev_node = node_id

    return first_node, prev_node


def build_parallel_graph(
    branches: list, graph: nx.DiGraph, parent_id: Optional[str] = None
) -> tuple[str, str]:
    """Build graph nodes for Parallel pipeline (multiple rows).

    Args:
        branches: List of branches in parallel
        graph: NetworkX directed graph to add nodes to
        parent_id: ID of parent node to connect from

    Returns:
        Tuple of (split_node_id, merge_node_id) for this parallel section
    """
    # Create split node
    split_id = f"split_{id(branches)}"
    graph.add_node(split_id, label="Split", component="Split", shape="diamond")

    if parent_id is not None:
        graph.add_edge(parent_id, split_id)

    # Create merge node
    merge_id = f"merge_{id(branches)}"
    graph.add_node(merge_id, label="Merge", component="Merge", shape="diamond")

    # Process each branch
    branch_outputs = []
    for branch_idx, branch in enumerate(branches):
        if isinstance(branch, Serial):
            # Handle nested Serial
            first_node, last_node = build_serial_graph(
                branch._all_components, graph, split_id
            )
            branch_outputs.append(last_node)
        elif isinstance(branch, Parallel):
            # Handle nested Parallel
            first_node, last_node = build_parallel_graph(
                branch._all_branches, graph, split_id
            )
            branch_outputs.append(last_node)
        else:
            # Single component in branch
            component_name = branch.__class__.__name__
            node_id = f"{id(branch)}_{branch_idx}"
            input_shape, output_shape = get_component_shapes(branch)

            if input_shape and output_shape:
                label = f"{component_name}\n({input_shape})\n→\n({output_shape})"
            elif output_shape:
                label = f"{component_name}\n→ ({output_shape})"
            else:
                label = component_name

            graph.add_node(node_id, label=label, component=component_name)
            graph.add_edge(split_id, node_id)
            branch_outputs.append(node_id)

    # Connect all branch outputs to merge node
    for output_node in branch_outputs:
        graph.add_edge(output_node, merge_id)

    return split_id, merge_id


def summarize(pipeline: nn.Module, output_file: str = "pipeline_viz.html") -> Network:
    """Visualize pipeline structure with tensor shapes from jaxtyping annotations.

    Creates an interactive HTML visualization showing:
    - Serial pipelines: left-to-right flow with shape transformations
    - Parallel pipelines: multiple branches showing parallel processing
    - Tensor shapes extracted from jaxtyping type hints

    Args:
        pipeline: Serial or Parallel pipeline to visualize
        output_file: Path to save HTML visualization (default: "pipeline_viz.html")

    Returns:
        PyVis Network object (also saves to output_file)

    Example:
        >>> from vectormesh.components import Serial, MeanAggregator, NeuralNet
        >>> pipeline = Serial([
        ...     MeanAggregator(),
        ...     NeuralNet(hidden_size=768, out_size=32)
        ... ])
        >>> summarize(pipeline, "my_model.html")
    """
    # Create directed graph
    graph = nx.DiGraph()

    # Build graph based on pipeline type
    if isinstance(pipeline, Serial):
        logger.info("Building visualization for Serial pipeline")
        build_serial_graph(pipeline._all_components, graph)
    elif isinstance(pipeline, Parallel):
        logger.info("Building visualization for Parallel pipeline")
        build_parallel_graph(pipeline._all_branches, graph)
    else:
        raise ValueError(
            f"Pipeline must be Serial or Parallel, got {type(pipeline).__name__}"
        )

    # Create PyVis network
    net = Network(
        height="600px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#ffffff",
        font_color="#000000",
    )

    # Configure physics for left-to-right layout
    net.set_options(
        """
        {
            "layout": {
                "hierarchical": {
                    "enabled": true,
                    "direction": "LR",
                    "sortMethod": "directed",
                    "nodeSpacing": 200,
                    "levelSeparation": 300
                }
            },
            "physics": {
                "hierarchicalRepulsion": {
                    "nodeDistance": 200
                }
            },
            "nodes": {
                "shape": "box",
                "font": {
                    "size": 14,
                    "face": "monospace"
                },
                "borderWidth": 2,
                "color": {
                    "border": "#2B7CE9",
                    "background": "#D2E5FF"
                }
            },
            "edges": {
                "arrows": "to",
                "smooth": {
                    "type": "cubicBezier"
                },
                "font": {
                    "size": 12,
                    "face": "monospace",
                    "align": "middle"
                }
            }
        }
        """
    )

    # Convert NetworkX graph to PyVis
    net.from_nx(graph)

    # Save and return
    net.save_graph(output_file)
    logger.success(f"Pipeline visualization saved to {output_file}")

    return net
