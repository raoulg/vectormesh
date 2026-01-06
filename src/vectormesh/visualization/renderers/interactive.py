"""Interactive HTML renderer using PyVis with MathJax support."""

import os
import subprocess
from typing import Any, Dict, Optional

import networkx as nx
from pyvis.network import Network

from ..types import InteractionMode, VisualizationMode


# Professional color scheme
COLOR_SCHEME = {
    "text_domain": "#FF6B6B",      # Σ* - Warm coral
    "tensor_2d": "#4ECDC4",        # ℝ^2D - Teal
    "tensor_3d": "#95E1D3",        # ℝ^3D - Mint green
    "tensor_4d": "#A78BFA",        # ℝ^4D - Soft purple
    "morphism_arrow": "#2C3E50",   # Arrows - Dark slate
    "hover_highlight": "#FFD93D",  # On hover - Gold
    "cached_component": "#6BCF7F", # Cached - Fresh green
    "background": "#F8F9FA",       # Canvas - Light gray
    "node_border": "#34495E",      # Node outlines - Charcoal
}


class InteractiveRenderer:
    """Render NetworkX graph to interactive HTML with PyVis and MathJax."""

    def __init__(
        self,
        mode: VisualizationMode = VisualizationMode.HYBRID,
        interaction: InteractionMode = InteractionMode.VIEW,
        color_scheme: Optional[Dict[str, str]] = None,
    ):
        """Initialize renderer.

        Args:
            mode: Visualization detail level
            interaction: Interactive mode
            color_scheme: Custom color scheme (uses default if None)
        """
        self.mode = mode
        self.interaction = interaction
        self.colors = color_scheme or COLOR_SCHEME

    def render(
        self,
        G: nx.DiGraph,
        positions: Dict[str, tuple],
        output_file: Optional[str] = None,
    ) -> str:
        """Render graph to interactive HTML.

        Args:
            G: NetworkX directed graph
            positions: Node positions (x, y) coordinates
            output_file: Optional output file path

        Returns:
            HTML string
        """
        # Create PyVis network
        net = Network(
            height="800px",
            width="100%",
            directed=True,
            notebook=False,
            bgcolor=self.colors["background"],
            select_menu=False,
            filter_menu=False,
        )

        # Disable physics (use fixed positions)
        net.set_options("""
        {
          "physics": {
            "enabled": false
          },
          "edges": {
            "smooth": {
              "type": "curvedCW",
              "roundness": 0.2
            },
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 1.2
              }
            }
          },
          "nodes": {
            "shape": "box",
            "margin": 10,
            "font": {
              "multi": "html"
            },
            "widthConstraint": {
              "minimum": 100,
              "maximum": 200
            }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 0,
            "hideEdgesOnDrag": false,
            "hideEdgesOnZoom": false
          }
        }
        """)

        # Add nodes with styling
        for node, (x, y) in positions.items():
            node_data = G.nodes[node]
            label = node_data.get("label", node)

            # Determine node color based on domain type metadata
            domain_type = node_data.get("domain_type", "2d")
            color = self._get_node_color_from_type(domain_type)

            # Build tooltip
            tooltip = self._build_node_tooltip(node, node_data)

            net.add_node(
                node,
                label=label,
                x=x,
                y=y,
                color=color,
                borderWidth=2,
                borderWidthSelected=4,
                title=tooltip,
                font={"size": 16, "face": "Arial, sans-serif"},
            )

        # Add edges with styling
        for source, target, edge_data in G.edges(data=True):
            morphism = edge_data.get("morphism", "")

            # Build edge tooltip
            tooltip = self._build_edge_tooltip(edge_data)

            # Build edge label
            label = morphism

            net.add_edge(
                source,
                target,
                label=label,
                title=tooltip,
                color=self.colors["morphism_arrow"],
                width=2,
                font={"size": 14, "align": "middle"},
            )

        # Generate HTML
        html = net.generate_html()

        # Inject MathJax and fix tooltip rendering
        html = self._inject_mathjax_and_fix_tooltips(html)

        # Add legend in LEARN mode
        if self.interaction == InteractionMode.LEARN:
            html = self._add_legend(html)

        # Save to file if specified
        if output_file:
            with open(output_file, "w") as f:
                f.write(html)

        return html

    def _get_node_color_from_type(self, domain_type: str) -> str:
        """Get node color based on domain type metadata.

        Args:
            domain_type: One of "text", "2d", "3d", "4d"

        Returns:
            Hex color code
        """
        color_map = {
            "text": self.colors["text_domain"],
            "2d": self.colors["tensor_2d"],
            "3d": self.colors["tensor_3d"],
            "4d": self.colors["tensor_4d"],
        }
        return color_map.get(domain_type, self.colors["tensor_2d"])

    def _build_node_tooltip(self, node_id: str, node_data: Dict[str, Any]) -> str:
        """Build tooltip HTML for node (domain)."""
        label = node_data.get("label", node_id)

        tooltip = f"<b>Domain:</b> {label}<br>"
        tooltip += "<br><i>Object in VectorMesh category</i>"

        return tooltip

    def _build_edge_tooltip(self, edge_data: Dict[str, Any]) -> str:
        """Build tooltip HTML for edge (morphism)."""
        morphism = edge_data.get("morphism", "f")
        component = edge_data.get("component_name", "Component")
        signature = edge_data.get("signature", "")
        params = edge_data.get("params", {})

        # Mathematical signature
        tooltip = f"<b>Morphism:</b> {morphism}<br>"
        if signature:
            tooltip += f"<b>Signature:</b> {signature}<br>"

        tooltip += f"<br><b>Implementation:</b> {component}"

        # Show parameters
        if params:
            tooltip += "<br><b>Parameters:</b><br>"
            for key, value in params.items():
                tooltip += f"  • {key}: {value}<br>"

        # Code location (DEBUG mode)
        if self.interaction == InteractionMode.DEBUG:
            source_file = edge_data.get("source_file")
            source_line = edge_data.get("source_line")

            if source_file:
                tooltip += f"<br><b>Source:</b> {source_file}:{source_line}<br>"

                # Add "open in editor" hint
                if self._can_open_in_editor():
                    tooltip += f"<br><i>Click to open in editor</i>"

                # Code snippet
                snippet = edge_data.get("code_snippet")
                if snippet:
                    # Truncate snippet for tooltip
                    lines = snippet.split("\n")[:5]
                    snippet_short = "\n".join(lines)
                    tooltip += f"<br><pre style='font-size:10px'>{snippet_short}</pre>"

        return tooltip

    def _can_open_in_editor(self) -> bool:
        """Check if we can open files in user's editor."""
        return os.environ.get("EDITOR") is not None

    def _inject_mathjax_and_fix_tooltips(self, html: str) -> str:
        """Inject MathJax CDN for LaTeX rendering and configure HTML tooltips."""
        mathjax_script = """
<!-- MathJax for LaTeX rendering -->
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
  },
  startup: {
    pageReady: () => {
      return MathJax.startup.defaultPageReady().then(() => {
        // Initial typeset only
        MathJax.typesetPromise();
      });
    }
  }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<!-- Hide default vis.js tooltips (we use custom HTML tooltips) -->
<style>
.vis-tooltip {
  display: none !important;
}

.vis-network canvas {
  outline: none;
}

#custom-tooltip {
  line-height: 1.5;
}

#custom-tooltip b {
  color: #2C3E50;
}

#custom-tooltip pre {
  background: #f5f5f5;
  padding: 5px;
  border-radius: 3px;
  margin: 5px 0;
  overflow-x: auto;
}
</style>

<!-- Custom tooltip system (vis.js doesn't support HTML tooltips natively) -->
<script>
let customTooltip = null;
let tooltipTimeout = null;

// Create custom tooltip element
function createCustomTooltip() {
  const tooltip = document.createElement('div');
  tooltip.id = 'custom-tooltip';
  tooltip.style.cssText = `
    position: absolute;
    background: white;
    border: 2px solid #34495E;
    border-radius: 5px;
    padding: 10px;
    font-family: sans-serif;
    font-size: 14px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    max-width: 400px;
    z-index: 10000;
    pointer-events: none;
    display: none;
  `;
  document.body.appendChild(tooltip);
  return tooltip;
}

// Show custom tooltip
function showCustomTooltip(x, y, htmlContent) {
  if (!customTooltip) {
    customTooltip = createCustomTooltip();
  }

  // Clear any pending hide
  if (tooltipTimeout) {
    clearTimeout(tooltipTimeout);
    tooltipTimeout = null;
  }

  // Set HTML content
  customTooltip.innerHTML = htmlContent;

  // Position tooltip
  customTooltip.style.left = (x + 10) + 'px';
  customTooltip.style.top = (y + 10) + 'px';
  customTooltip.style.display = 'block';

  // Typeset with MathJax
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise([customTooltip]).catch(err => {});
  }
}

// Hide custom tooltip
function hideCustomTooltip() {
  tooltipTimeout = setTimeout(() => {
    if (customTooltip) {
      customTooltip.style.display = 'none';
    }
  }, 100);
}

// Wait for vis.js network to be ready
window.addEventListener('load', function() {
  setTimeout(function() {
    // Access the global network variable created by PyVis
    if (typeof network === 'undefined') {
      console.error('Network not found');
      return;
    }

    // Handle hover on nodes
    network.on('hoverNode', function(params) {
      const nodeId = params.node;
      const nodeData = network.body.data.nodes.get(nodeId);

      if (nodeData && nodeData.title) {
        const pointer = params.pointer.DOM;
        showCustomTooltip(pointer.x, pointer.y, nodeData.title);
      }
    });

    // Handle hover on edges
    network.on('hoverEdge', function(params) {
      const edgeId = params.edge;
      const edgeData = network.body.data.edges.get(edgeId);

      if (edgeData && edgeData.title) {
        const pointer = params.pointer.DOM;
        showCustomTooltip(pointer.x, pointer.y, edgeData.title);
      }
    });

    // Handle blur (mouse leaves)
    network.on('blurNode', hideCustomTooltip);
    network.on('blurEdge', hideCustomTooltip);

    // Also hide on canvas click
    network.on('click', hideCustomTooltip);

  }, 500);
});
</script>
"""

        # Insert before closing </head> tag
        html = html.replace("</head>", f"{mathjax_script}</head>")

        # Configure vis.js to render HTML in tooltips (not escape it)
        # This is done by modifying the generated HTML to unescape HTML entities in title attributes
        # Vis.js will render the title as-is, which includes HTML tags
        # The CSS and MathJax will then properly format it

        return html

    def _add_legend(self, html: str) -> str:
        """Add CT legend in LEARN mode."""
        legend = """
<div style="position: fixed; top: 10px; right: 10px; background: white;
            border: 2px solid #34495E; border-radius: 8px; padding: 15px;
            max-width: 250px; font-family: sans-serif; font-size: 13px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
  <h3 style="margin-top: 0; color: #2C3E50;">VectorMesh</h3>

  <div style="margin: 10px 0;">
    <b>Color Code:</b><br>
    <span style="color: #FF6B6B;">● Σ* (Text)</span><br>
    <span style="color: #4ECDC4;">● ℝ^2D (2D Tensors)</span><br>
    <span style="color: #95E1D3;">● ℝ^3D (3D Tensors)</span>
  </div>

  <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #ccc; font-size: 11px; color: #666;">
    <i>Hover over arrows for details</i>
  </div>
</div>
"""

        # Insert before closing </body> tag
        html = html.replace("</body>", f"{legend}</body>")

        return html
