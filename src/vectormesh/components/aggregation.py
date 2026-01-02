"""Parameter-free aggregation strategies for vector embeddings.

This module provides simple pooling operations to reduce 3D embeddings
(batch, chunks, dim) to 2D vectors (batch, dim) for downstream tasks.

The module follows the Open-Closed Principle: users can extend with custom
aggregation strategies by subclassing BaseAggregator and implementing the
simple _aggregate() method.
"""

from abc import abstractmethod

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from pydantic import ConfigDict
from torch import Tensor

from vectormesh.base import VectorMeshComponent
from vectormesh.errors import VectorMeshError


class BaseAggregator(VectorMeshComponent):
    """Base class for aggregation strategies (Open-Closed Principle).

    Users can extend this class to implement custom aggregation logic
    by simply overriding the _aggregate() method. All type safety and
    validation boilerplate is handled in __call__().

    Shapes:
        Input: (batch, chunks, dim) - 3D tensor from chunked embeddings
        Output: (batch, dim) - 2D tensor suitable for Linear layers

    Example:
        ```python
        from vectormesh.components.aggregation import BaseAggregator
        import torch

        class AttentionAggregator(BaseAggregator):
            '''Custom attention-based aggregation.'''

            def _aggregate(self, embeddings: Tensor) -> Tensor:
                # Your custom aggregation logic - just return the result!
                attention_weights = self._compute_attention(embeddings)
                return torch.sum(embeddings * attention_weights, dim=1)
        ```

    Note:
        Subclasses only need to implement _aggregate(). No decorators needed!
    """

    model_config = ConfigDict(frozen=True)

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, embeddings: Float[Tensor, "batch chunks dim"]
    ) -> Float[Tensor, "batch dim"]:
        """Apply aggregation with full type safety handled here.

        This method handles all the boilerplate: type checking, shape validation,
        and error handling. Subclasses only need to implement _aggregate().

        Args:
            embeddings: 3D tensor of shape (batch, chunks, dim)

        Returns:
            2D tensor of shape (batch, dim) with aggregated embeddings

        Shapes:
            Input: (B, N, D) where B=batch, N=chunks, D=embedding_dim
            Output: (B, D)
        """
        return self._aggregate(embeddings)

    @abstractmethod
    def _aggregate(self, embeddings: Tensor) -> Tensor:
        """Core aggregation logic - override in subclasses.

        This is the only method you need to implement when extending BaseAggregator.
        Simply return the aggregated tensor - no decorators or type annotations needed.

        Args:
            embeddings: Input tensor of shape (batch, chunks, dim)

        Returns:
            Aggregated tensor of shape (batch, dim)

        Note:
            Type safety is handled by __call__() - you don't need decorators here.
        """
        pass


class MeanAggregator(BaseAggregator):
    """Mean pooling aggregation strategy.

    Averages embeddings across the chunks dimension (dim=1), producing
    a single representative vector per document.

    Shapes:
        Input: (batch, chunks, dim)
        Output: (batch, dim)

    Example:
        ```python
        from vectormesh.components.aggregation import MeanAggregator
        import torch

        agg = MeanAggregator()
        embeddings = torch.randn(32, 10, 384)  # 32 docs, 10 chunks, 384 dims
        result = agg(embeddings)  # Shape: (32, 384)
        ```
    """

    def _aggregate(self, embeddings: Tensor) -> Tensor:
        """Average embeddings across chunks dimension."""
        return torch.mean(embeddings, dim=1)


class MaxAggregator(BaseAggregator):
    """Max pooling aggregation strategy.

    Takes the maximum value across the chunks dimension (dim=1) for each
    embedding dimension, capturing the strongest signal from any chunk.

    Shapes:
        Input: (batch, chunks, dim)
        Output: (batch, dim)

    Example:
        ```python
        from vectormesh.components.aggregation import MaxAggregator
        import torch

        agg = MaxAggregator()
        embeddings = torch.randn(32, 10, 384)  # 32 docs, 10 chunks, 384 dims
        result = agg(embeddings)  # Shape: (32, 384)
        ```
    """

    def _aggregate(self, embeddings: Tensor) -> Tensor:
        """Max pool embeddings across chunks dimension."""
        return torch.max(embeddings, dim=1).values


def get_aggregator(strategy: str) -> BaseAggregator:
    """Factory function to load aggregator by class name.

    Dynamically loads an aggregator class from this module by name.
    This enables extensibility - users can define custom aggregators
    and load them by name.

    Args:
        strategy: Aggregator class name (e.g., "MeanAggregator", "MaxAggregator")

    Returns:
        Instance of the requested aggregator

    Raises:
        VectorMeshError: If aggregator not found or invalid

    Example:
        ```python
        from vectormesh.components.aggregation import get_aggregator

        # Load built-in aggregator
        agg = get_aggregator("MeanAggregator")
        result = agg(embeddings)

        # Or use custom aggregator (must be in this module or imported)
        custom_agg = get_aggregator("MyCustomAggregator")
        ```
    """
    import vectormesh.components.aggregation as agg_module

    try:
        aggregator_class = getattr(agg_module, strategy)

        # Verify it's a valid aggregator
        if not (
            isinstance(aggregator_class, type)
            and issubclass(aggregator_class, BaseAggregator)
        ):
            raise VectorMeshError(
                message=f"{strategy} is not a valid aggregator class",
                hint="Aggregators must inherit from BaseAggregator",
                fix="Use 'MeanAggregator', 'MaxAggregator', or create a custom class inheriting from BaseAggregator",
            )

        return aggregator_class()

    except AttributeError:
        raise VectorMeshError(
            message=f"Aggregator '{strategy}' not found",
            hint="Available built-in aggregators: MeanAggregator, MaxAggregator",
            fix=f"Use 'MeanAggregator' or 'MaxAggregator', or define your own {strategy} class",
        )
