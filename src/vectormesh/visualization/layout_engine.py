"""Layout engine for category theory diagrams with grid-based positioning."""

from typing import Dict, Tuple

import networkx as nx


class CTLayoutEngine:
    """Grid-based layout engine for category theory diagrams.

    Uses straight-line positioning suitable for CT diagrams:
    - Serial: Horizontal left-to-right chain
    - Parallel: Vertical fan-out from source
    - Skip: Dual horizontal paths (identity + morphism)
    """

    def __init__(
        self,
        node_spacing_x: int = 300,
        node_spacing_y: int = 200,
    ):
        """Initialize layout engine.

        Args:
            node_spacing_x: Horizontal spacing between nodes (pixels)
            node_spacing_y: Vertical spacing for parallel branches (pixels)
        """
        self.spacing_x = node_spacing_x
        self.spacing_y = node_spacing_y

    def compute_layout(
        self,
        G: nx.DiGraph,
        layout_type: str = "auto"
    ) -> Dict[str, Tuple[int, int]]:
        """Compute (x, y) positions for nodes.

        Args:
            G: NetworkX directed graph
            layout_type: Type of layout ("auto", "serial", "parallel", "skip")

        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        if layout_type == "serial":
            return self._layout_serial(G)
        elif layout_type == "parallel":
            return self._layout_parallel(G)
        elif layout_type == "auto":
            # Auto-detect layout type
            if self._is_parallel_graph(G):
                return self._layout_parallel(G)
            else:
                return self._layout_serial(G)
        else:
            # Default to serial
            return self._layout_serial(G)

    def _is_serial_graph(self, G: nx.DiGraph) -> bool:
        """Check if graph is a serial chain."""
        # Serial: Linear chain with no branching
        nodes = list(G.nodes())
        if len(nodes) < 2:
            return True

        # Check if it's a simple path (no branching)
        for node in nodes:
            if G.out_degree(node) > 1:
                return False
        return True

    def _is_parallel_graph(self, G: nx.DiGraph) -> bool:
        """Check if graph is a parallel structure."""
        # Parallel: One source node with multiple outgoing edges
        nodes = list(G.nodes())
        if len(nodes) < 2:
            return False

        # Find nodes with out-degree > 1 (source nodes)
        source_nodes = [n for n in nodes if G.out_degree(n) > 1]
        return len(source_nodes) > 0

    def _layout_serial(self, G: nx.DiGraph) -> Dict[str, Tuple[int, int]]:
        """Horizontal left-to-right chain layout."""
        positions = {}

        # Get nodes in topological order (source to target)
        try:
            nodes = list(nx.topological_sort(G))
        except nx.NetworkXError:
            # Graph has cycles, just use node list
            nodes = list(G.nodes())

        # Position nodes horizontally
        for i, node in enumerate(nodes):
            x = i * self.spacing_x
            y = 0  # All nodes at same vertical level
            positions[node] = (x, y)

        return positions

    def _layout_parallel(self, G: nx.DiGraph) -> Dict[str, Tuple[int, int]]:
        """Vertical fan-out from source node layout."""
        positions = {}

        # Find source node (node with max out-degree)
        nodes = list(G.nodes())
        source_node = max(nodes, key=lambda n: G.out_degree(n))

        # Position source at left
        positions[source_node] = (0, 0)

        # Get target nodes (nodes that source connects to)
        targets = list(G.successors(source_node))

        # Position targets vertically, centered around source
        n_targets = len(targets)
        if n_targets == 0:
            return positions

        # Calculate vertical spread
        total_height = (n_targets - 1) * self.spacing_y
        start_y = -total_height / 2

        for i, target in enumerate(targets):
            y = start_y + i * self.spacing_y
            x = self.spacing_x
            positions[target] = (x, y)

        # Position any remaining nodes (e.g., nested structures)
        positioned = {source_node, *targets}
        remaining = [n for n in nodes if n not in positioned]

        for i, node in enumerate(remaining):
            # Place to the right of targets
            x = self.spacing_x * 2
            y = start_y + (i % n_targets) * self.spacing_y
            positions[node] = (x, y)

        return positions
