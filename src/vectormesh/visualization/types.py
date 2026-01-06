"""Type definitions and enums for visualization system."""

from enum import Enum


class VisualizationFormat(Enum):
    """Output format for visualization."""

    MARKDOWN = "markdown"
    INTERACTIVE = "interactive"
    SVG = "svg"
    JSON = "json"


class VisualizationMode(Enum):
    """Visualization detail level."""

    PRACTICAL = "practical"      # Component names only
    THEORETICAL = "theoretical"  # CT structure only
    HYBRID = "hybrid"            # Both (default)


class InteractionMode(Enum):
    """Interactive diagram modes."""

    VIEW = "view"          # Read-only exploration
    DEBUG = "debug"        # Code snippets and source links
    LEARN = "learn"        # CT education tooltips
    COMPARE = "compare"    # Side-by-side diff
