"""Build NetworkX graphs from VectorMesh components for category theory diagrams."""

import inspect
import os
from typing import Any, Dict, Optional, Tuple

import networkx as nx


class CategoryGraphBuilder:
    """Builds NetworkX DiGraph from VectorMesh components.

    In the resulting graph:
    - Nodes represent domains/codomains (Σ*, ℝ^{B×D}, etc.)
    - Edges represent morphisms (components like vectorizers, aggregators)
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_counter = 0
        self.morphism_counter = 0

    def build(self, component: Any) -> nx.DiGraph:
        """Build graph from VectorMesh component."""
        from vectormesh.components.combinators import Serial, Parallel

        if isinstance(component, Serial):
            return self._build_serial(component)
        elif isinstance(component, Parallel):
            return self._build_parallel(component)
        else:
            return self._build_single(component)

    def _build_serial(self, serial: Any) -> nx.DiGraph:
        """Build graph for Serial combinator (horizontal chain)."""
        G = nx.DiGraph()
        components = serial.components

        if not components:
            return G

        # Create chain of nodes (domains) connected by morphisms (edges)
        for i, comp in enumerate(components):
            source_label, source_type = self._infer_source_domain(comp, i == 0)
            target_label, target_type = self._infer_target_domain(comp)

            source_node = f"domain_{i}"
            target_node = f"domain_{i + 1}"

            # Add nodes with labels and domain type metadata
            if source_node not in G:
                G.add_node(source_node, label=source_label, node_type="domain", domain_type=source_type)

            G.add_node(target_node, label=target_label, node_type="domain", domain_type=target_type)

            # Add morphism edge with metadata
            morphism_label = chr(102 + i)  # f, g, h, ...
            metadata = self._extract_component_metadata(comp, morphism_label)

            G.add_edge(
                source_node,
                target_node,
                morphism=morphism_label,
                **metadata
            )

        return G

    def _build_parallel(self, parallel: Any) -> nx.DiGraph:
        """Build graph for Parallel combinator (fan-out structure)."""
        G = nx.DiGraph()
        branches = parallel.branches

        if not branches:
            return G

        # Source node (common input domain)
        source_node = "source"
        G.add_node(source_node, label="Σ*", node_type="domain", domain_type="text")

        # Create fan-out to multiple target domains
        for i, branch in enumerate(branches):
            target_label, target_type = self._infer_target_domain(branch)
            target_node = f"branch_{i}"

            G.add_node(target_node, label=target_label, node_type="domain", domain_type=target_type)

            # Add morphism edge
            morphism_label = f"f_{i + 1}"
            metadata = self._extract_component_metadata(branch, morphism_label)

            G.add_edge(
                source_node,
                target_node,
                morphism=morphism_label,
                **metadata
            )

        return G

    def _build_single(self, component: Any) -> nx.DiGraph:
        """Build graph for single component."""
        G = nx.DiGraph()

        source_label, source_type = self._infer_source_domain(component, is_first=True)
        target_label, target_type = self._infer_target_domain(component)

        G.add_node("source", label=source_label, node_type="domain", domain_type=source_type)
        G.add_node("target", label=target_label, node_type="domain", domain_type=target_type)

        metadata = self._extract_component_metadata(component, "f")
        G.add_edge("source", "target", morphism="f", **metadata)

        return G

    def _infer_source_domain(self, component: Any, is_first: bool) -> Tuple[str, str]:
        """Infer source domain (Unicode notation for labels).

        Returns:
            Tuple of (label, domain_type) where domain_type is "text", "2d", "3d", or "4d"
        """
        if is_first:
            # First component typically takes text input
            return ("Σ*", "text")

        # Try to infer from output_mode
        if hasattr(component, "output_mode"):
            if component.output_mode == "2d":
                return (self._format_tensor_domain(component, "2d"), "2d")
            elif component.output_mode == "3d":
                return (self._format_tensor_domain(component, "3d"), "3d")

        return ("ℝᴮˣᴱ", "2d")

    def _infer_target_domain(self, component: Any) -> Tuple[str, str]:
        """Infer target domain (Unicode notation for labels).

        Returns:
            Tuple of (label, domain_type) where domain_type is "text", "2d", "3d", or "4d"
        """
        # Check for nested combinators
        from vectormesh.components.combinators import Serial, Parallel

        if isinstance(component, Serial):
            if component.components:
                return self._infer_target_domain(component.components[-1])
        elif isinstance(component, Parallel):
            # Parallel produces tuple - for now, just show first branch
            if component.branches:
                return self._infer_target_domain(component.branches[0])

        # Try to get dimension info from component
        if hasattr(component, "output_mode"):
            if component.output_mode == "3d":
                return (self._format_tensor_domain(component, "3d"), "3d")
            elif component.output_mode == "2d":
                return (self._format_tensor_domain(component, "2d"), "2d")

        # Try to get from embedding_dim or output_dim
        if hasattr(component, "embedding_dim"):
            dim = component.embedding_dim
            return (f"ℝᴮˣ{self._to_superscript(str(dim))}", "2d")
        elif hasattr(component, "output_dim"):
            dim = component.output_dim
            return (f"ℝᴮˣ{self._to_superscript(str(dim))}", "2d")

        return ("ℝᴮˣᴱ", "2d")

    def _format_tensor_domain(self, component: Any, mode: str) -> str:
        """Format tensor domain with dimensions (Unicode notation)."""
        if mode == "3d":
            if hasattr(component, "embedding_dim"):
                dim = component.embedding_dim
                return f"ℝᴮˣᶜˣ{self._to_superscript(str(dim))}"
            return "ℝᴮˣᶜˣᴱ"
        else:  # 2d
            if hasattr(component, "embedding_dim"):
                dim = component.embedding_dim
                return f"ℝᴮˣ{self._to_superscript(str(dim))}"
            elif hasattr(component, "output_dim"):
                dim = component.output_dim
                return f"ℝᴮˣ{self._to_superscript(str(dim))}"
            return "ℝᴮˣᴱ"

    def _to_superscript(self, text: str) -> str:
        """Convert digits to Unicode superscripts."""
        superscript_map = {
            '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
            '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'
        }
        return ''.join(superscript_map.get(c, c) for c in text)

    def _extract_component_metadata(
        self,
        component: Any,
        morphism_label: str
    ) -> Dict[str, Any]:
        """Extract metadata for morphism edge."""
        metadata = {
            "component_name": component.__class__.__name__,
            "morphism_label": morphism_label,
        }

        # Extract component parameters
        params = {}
        if hasattr(component, "model_name"):
            params["model_name"] = component.model_name
        if hasattr(component, "dim"):
            params["dim"] = component.dim
        if hasattr(component, "embedding_dim"):
            params["embedding_dim"] = component.embedding_dim
        if hasattr(component, "output_dim"):
            params["output_dim"] = component.output_dim

        if params:
            metadata["params"] = params

        # Try to get source code location
        location = self._get_source_location(component)
        if location:
            metadata.update(location)

        # Build signature for tooltip (use LaTeX in tooltips)
        source_latex = self._infer_source_domain_latex(component, is_first=True)
        target_latex = self._infer_target_domain_latex(component)
        metadata["signature"] = f"${morphism_label}: {source_latex} \\to {target_latex}$"

        return metadata

    def _infer_source_domain_latex(self, component: Any, is_first: bool) -> str:
        """Infer source domain (LaTeX notation for tooltips)."""
        if is_first:
            return "\\Sigma^*"

        if hasattr(component, "output_mode"):
            if component.output_mode == "2d":
                return self._format_tensor_domain_latex(component, "2d")
            elif component.output_mode == "3d":
                return self._format_tensor_domain_latex(component, "3d")

        return "\\mathbb{R}^{B \\times E}"

    def _infer_target_domain_latex(self, component: Any) -> str:
        """Infer target domain (LaTeX notation for tooltips)."""
        from vectormesh.components.combinators import Serial, Parallel

        if isinstance(component, Serial):
            if component.components:
                return self._infer_target_domain_latex(component.components[-1])
        elif isinstance(component, Parallel):
            if component.branches:
                return self._infer_target_domain_latex(component.branches[0])

        if hasattr(component, "output_mode"):
            if component.output_mode == "3d":
                return self._format_tensor_domain_latex(component, "3d")
            elif component.output_mode == "2d":
                return self._format_tensor_domain_latex(component, "2d")

        if hasattr(component, "embedding_dim"):
            dim = component.embedding_dim
            return f"\\mathbb{{R}}^{{{{B \\times {dim}}}}}"
        elif hasattr(component, "output_dim"):
            dim = component.output_dim
            return f"\\mathbb{{R}}^{{{{B \\times {dim}}}}}"

        return "\\mathbb{R}^{B \\times E}"

    def _format_tensor_domain_latex(self, component: Any, mode: str) -> str:
        """Format tensor domain with dimensions (LaTeX notation for tooltips)."""
        if mode == "3d":
            if hasattr(component, "embedding_dim"):
                dim = component.embedding_dim
                return f"\\mathbb{{R}}^{{{{B \\times C \\times {dim}}}}}"
            return "\\mathbb{R}^{B \\times C \\times E}"
        else:  # 2d
            if hasattr(component, "embedding_dim"):
                dim = component.embedding_dim
                return f"\\mathbb{{R}}^{{{{B \\times {dim}}}}}"
            elif hasattr(component, "output_dim"):
                dim = component.output_dim
                return f"\\mathbb{{R}}^{{{{B \\times {dim}}}}}"
            return "\\mathbb{R}^{B \\times E}"

    def _get_source_location(self, component: Any) -> Optional[Dict[str, Any]]:
        """Get source file and line number for component."""
        try:
            cls = component.__class__
            source_file = inspect.getfile(cls)

            # Get relative path from cwd
            try:
                rel_path = os.path.relpath(source_file)
            except ValueError:
                rel_path = source_file

            # Get line number
            source_lines, line_num = inspect.getsourcelines(cls)

            # Get code snippet (first 10 lines)
            snippet = "".join(source_lines[:10])

            return {
                "source_file": rel_path,
                "source_line": line_num,
                "code_snippet": snippet,
            }
        except (TypeError, OSError):
            # Component is built-in or defined interactively
            return None
