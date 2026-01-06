"""Mathematical notation constants for visualization.

Single source of truth for all mathematical symbols used in visualizations.
"""


class MathNotation:
    """Central configuration for mathematical notation symbols.

    Change these constants to customize notation across all visualizations.
    """

    # Unicode mathematical symbols
    REAL_TENSOR = "ℝ"  # U+211D - Double-struck capital R
    TEXT_DOMAIN = "Σ*"  # Sigma star (Kleene star notation)
    DIMENSION_SEP = "×"  # U+00D7 - Multiplication sign
    ARROW = "→"  # U+2192 - Rightwards arrow

    # LaTeX equivalents for tooltips/documents
    REAL_TENSOR_LATEX = r"\mathbb{R}"
    TEXT_DOMAIN_LATEX = r"\Sigma^*"
    DIMENSION_SEP_LATEX = r"\times"

    # Superscript characters for Unicode rendering
    SUPERSCRIPT_MAP = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        'B': 'ᴮ',  # U+1D2E - Modifier letter capital B
        'C': 'ᶜ',  # U+1D9C - Modifier letter small c
        'E': 'ᴱ',  # U+1D31 - Modifier letter capital E
        'x': 'ˣ',  # U+02E3 - Modifier letter small x (for ×)
    }

    @classmethod
    def to_superscript(cls, text: str) -> str:
        """Convert text to Unicode superscripts.

        Args:
            text: String to convert (e.g., "BxE", "768")

        Returns:
            Superscripted version using Unicode characters
        """
        return ''.join(cls.SUPERSCRIPT_MAP.get(c, c) for c in text)

    @classmethod
    def format_tensor_shape(cls, dimensions: list[str], use_latex: bool = False) -> str:
        """Format tensor shape with proper notation.

        Args:
            dimensions: List of dimension labels (e.g., ["B", "768"])
            use_latex: If True, return LaTeX notation; else Unicode

        Returns:
            Formatted tensor shape (e.g., "ℝᴮˣ⁷⁶⁸" or "\\mathbb{R}^{B \\times 768}")

        Examples:
            >>> MathNotation.format_tensor_shape(["B", "768"])
            'ℝᴮˣ⁷⁶⁸'
            >>> MathNotation.format_tensor_shape(["B", "C", "384"])
            'ℝᴮˣᶜˣ³⁸⁴'
            >>> MathNotation.format_tensor_shape(["B", "768"], use_latex=True)
            '\\\\mathbb{R}^{B \\\\times 768}'
        """
        if not dimensions:
            return cls.TEXT_DOMAIN if not use_latex else cls.TEXT_DOMAIN_LATEX

        if use_latex:
            dim_str = f" {cls.DIMENSION_SEP_LATEX} ".join(dimensions)
            return f"{cls.REAL_TENSOR_LATEX}^{{{{{dim_str}}}}}"
        else:
            # Unicode superscript version
            dim_parts = []
            for dim in dimensions:
                dim_parts.append(cls.to_superscript(dim))

            sep_super = cls.to_superscript('x')  # × becomes ˣ
            dim_str = sep_super.join(dim_parts)
            return f"{cls.REAL_TENSOR}{dim_str}"
