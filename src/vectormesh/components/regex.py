import re
from typing import Dict, List, Optional
from pydantic import ConfigDict, field_validator
import torch
from jaxtyping import Float, jaxtyped
from beartype import beartype

from vectormesh.types import VectorMeshComponent, VectorMeshError, TwoDTensor

class RegexVectorizer(VectorMeshComponent):
    """
    Vectorizes text based on regex pattern matches.
    
    Produces a binary vector where each dimension corresponds to a regex pattern,
    indicating presence (1.0) or absence (0.0).
    """
    model_config = ConfigDict(frozen=True)
    
    patterns: Dict[str, str]

    @field_validator("patterns")
    @classmethod
    def validate_patterns(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate that all patterns are valid python regexes."""
        for label, pattern in v.items():
            try:
                re.compile(pattern)
            except re.error as e:
                raise VectorMeshError(
                    msg=f"Invalid regex pattern for '{label}': {pattern}",
                    hint=f"Check regex syntax at position {e.pos}",
                    fix="Use a regex tester (e.g. regex101) to verify your pattern."
                ) from e
        return v

    @jaxtyped(typechecker=beartype)
    def __call__(self, texts: List[str]) -> Float[torch.Tensor, "batch num_patterns"]:
        """
        Vectorize a list of texts into binary pattern indicators.
        
        Args:
            texts: List of input strings.
            
        Returns:
            TwoDTensor of shape (batch, num_patterns) with 1.0 for match, 0.0 for none.
        """
        batch_size = len(texts)
        # Ensure consistent ordering of patterns
        pattern_items = list(self.patterns.items())
        num_patterns = len(pattern_items)
        
        if num_patterns == 0:
            return torch.zeros((batch_size, 0), dtype=torch.float32)

        # Pre-compile patterns (though they are compiled in validation, we need the objects)
        # For efficiency, we can cache compiled objects on the instance or compile here.
        # Given frozen=True and validation pass, we can safely compile.
        compiled_patterns = [re.compile(p) for _, p in pattern_items]
        
        # Initialize output tensor
        # Shape: (batch, num_patterns)
        vectors = torch.zeros((batch_size, num_patterns), dtype=torch.float32)
        
        for i, text in enumerate(texts):
            for j, pattern in enumerate(compiled_patterns):
                if pattern.search(text):
                    vectors[i, j] = 1.0
                    
        return vectors
