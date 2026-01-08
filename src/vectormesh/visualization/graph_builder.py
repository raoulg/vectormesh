"""Build NetworkX graphs from VectorMesh components for category theory diagrams."""

import inspect
import os
from typing import Any, Dict, Optional, Tuple

import networkx as nx


# Mathematical notation constants - single source of truth
UNICODE_REAL = "ℝ"  # U+211D
UNICODE_TEXT = "Σ*"  # Sigma star
UNICODE_DIM_SEP = "×"  # U+00D7

LATEX_REAL = r"\mathbb{R}"
LATEX_TEXT = r"\Sigma^*"
LATEX_DIM_SEP = r"\times"

# Superscript map for Unicode rendering
SUPERSCRIPT_MAP = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
    'B': 'ᴮ', 'C': 'ᶜ', 'E': 'ᴱ', 'x': 'ˣ',
}

# Domain type constants
DOMAIN_TEXT = "text"
DOMAIN_2D = "2d"
DOMAIN_3D = "3d"


def to_superscript(text: str) -> str:
    """Convert text to Unicode superscripts."""
    return ''.join(SUPERSCRIPT_MAP.get(c, c) for c in text)


def format_tensor_unicode(dimensions: list[str]) -> str:
    """Format tensor with Unicode notation.

    Args:
        dimensions: List of dimension labels (e.g., ["B", "768"])

    Returns:
        Unicode formatted tensor (e.g., "ℝᴮˣ⁷⁶⁸")
    """
    if not dimensions:
        return UNICODE_TEXT

    dim_parts = [to_superscript(d) for d in dimensions]
    sep = to_superscript('x')
    return f"{UNICODE_REAL}{sep.join(dim_parts)}"


def format_tensor_latex(dimensions: list[str]) -> str:
    """Format tensor with LaTeX notation.

    Args:
        dimensions: List of dimension labels (e.g., ["B", "768"])

    Returns:
        LaTeX formatted tensor (e.g., "\\mathbb{R}^{B \\times 768}")
    """
    if not dimensions:
        return LATEX_TEXT

    dim_str = f" {LATEX_DIM_SEP} ".join(dimensions)
    return f"{LATEX_REAL}^{{{{{dim_str}}}}}"


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
            # For serial chains, infer source from previous component's output
            if i == 0:
                source_label, source_type = self._infer_domain(comp, is_output=False)
            else:
                source_label, source_type = self._infer_domain(components[i-1], is_output=True)

            target_label, target_type = self._infer_domain(comp, is_output=True)

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

        # Infer source from first branch input
        source_label, source_type = self._infer_domain(branches[0], is_output=False)
        source_node = "source"
        G.add_node(source_node, label=source_label, node_type="domain", domain_type=source_type)

        # Create fan-out to multiple target domains
        for i, branch in enumerate(branches):
            target_label, target_type = self._infer_domain(branch, is_output=True)
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

        source_label, source_type = self._infer_domain(component, is_output=False)
        target_label, target_type = self._infer_domain(component, is_output=True)

        G.add_node("source", label=source_label, node_type="domain", domain_type=source_type)
        G.add_node("target", label=target_label, node_type="domain", domain_type=target_type)

        metadata = self._extract_component_metadata(component, "f")
        G.add_edge("source", "target", morphism="f", **metadata)

        return G

    def _infer_domain(self, component: Any, is_output: bool) -> Tuple[str, str]:
        """Infer domain from component using morphism system.

        Args:
            component: Component to analyze
            is_output: If True, infer output domain; else input domain

        Returns:
            Tuple of (label, domain_type)
        """
        # Check for nested combinators
        from vectormesh.components.combinators import Serial, Parallel

        if isinstance(component, Serial) and component.components:
            if is_output:
                return self._infer_domain(component.components[-1], is_output=True)
            else:
                return self._infer_domain(component.components[0], is_output=False)

        elif isinstance(component, Parallel) and component.branches:
            # For parallel, use first branch
            return self._infer_domain(component.branches[0], is_output)

        # Get tensor type from component's morphism metadata
        from vectormesh.validation import get_morphism

        try:
            morphism = get_morphism(component.__class__)
            if morphism:
                tensor_type = morphism.target if is_output else morphism.source

                # Map TensorDimensionality to our domain types
                from vectormesh.validation import TensorDimensionality

                if tensor_type == TensorDimensionality.TEXT:
                    return (UNICODE_TEXT, DOMAIN_TEXT)
                elif tensor_type == TensorDimensionality.TWO_D:
                    dims = self._extract_dimensions(component, "2d")
                    return (format_tensor_unicode(dims), DOMAIN_2D)
                elif tensor_type == TensorDimensionality.THREE_D:
                    dims = self._extract_dimensions(component, "3d")
                    return (format_tensor_unicode(dims), DOMAIN_3D)
        except (AttributeError, KeyError):
            pass

        # Fallback: infer from component attributes
        if hasattr(component, "output_mode"):
            if component.output_mode == "3d":
                dims = self._extract_dimensions(component, "3d")
                return (format_tensor_unicode(dims), DOMAIN_3D)
            elif component.output_mode == "2d":
                dims = self._extract_dimensions(component, "2d")
                return (format_tensor_unicode(dims), DOMAIN_2D)

        # Final fallback
        if hasattr(component, "embedding_dim") or hasattr(component, "output_dim"):
            dims = self._extract_dimensions(component, "2d")
            return (format_tensor_unicode(dims), DOMAIN_2D)

        # Unknown - use generic notation
        return (format_tensor_unicode(["B", "E"]), DOMAIN_2D)

    def _extract_dimensions(self, component: Any, mode: str) -> list[str]:
        """Extract dimension labels from component.

        Args:
            component: Component to analyze
            mode: "2d" or "3d"

        Returns:
            List of dimension labels (e.g., ["B", "768"] or ["B", "C", "384"])
        """
        if mode == "3d":
            base_dims = ["B", "C"]
        else:
            base_dims = ["B"]

        # Try to get concrete dimension value
        if hasattr(component, "embedding_dim"):
            base_dims.append(str(component.embedding_dim))
        elif hasattr(component, "output_dim"):
            base_dims.append(str(component.output_dim))
        else:
            base_dims.append("E")  # Generic embedding dimension

        return base_dims

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

        # Build signature for tooltip (use LaTeX)
        source_dims = self._extract_dimensions(component, "2d")  # Simplified
        target_dims = self._extract_dimensions(component, "2d")

        source_latex = format_tensor_latex(source_dims)
        target_latex = format_tensor_latex(target_dims)
        metadata["signature"] = f"${morphism_label}: {source_latex} \\to {target_latex}$"

        return metadata

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
