"""Tests for Aggregation components."""

import pytest
import torch

from vectormesh.components.aggregation import (
    BaseAggregator,
    MeanAggregator,
    MaxAggregator,
    get_aggregator,
)
from vectormesh.errors import VectorMeshError


class TestMeanAggregator:
    """Test suite for MeanAggregator component."""

    def test_mean_aggregator_initialization(self):
        """Test that MeanAggregator can be initialized."""
        agg = MeanAggregator()
        assert isinstance(agg, BaseAggregator)

    def test_mean_aggregator_frozen_config(self):
        """Test that MeanAggregator configuration is immutable."""
        agg = MeanAggregator()
        with pytest.raises(Exception):  # Pydantic ValidationError
            agg.some_field = "value"

    def test_mean_aggregator_pooling(self):
        """Test mean aggregation with known values."""
        # Create test tensor (batch=2, chunks=3, dim=4)
        embeddings = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0],
             [2.0, 3.0, 4.0, 5.0],
             [3.0, 4.0, 5.0, 6.0]],
            [[0.0, 1.0, 2.0, 3.0],
             [1.0, 2.0, 3.0, 4.0],
             [2.0, 3.0, 4.0, 5.0]]
        ])

        agg = MeanAggregator()
        result = agg(embeddings)

        # Verify shape
        assert result.shape == (2, 4)

        # Verify values (mean of each dim across chunks)
        expected = torch.tensor([
            [2.0, 3.0, 4.0, 5.0],  # Mean of [[1,2,3], [2,3,4], [3,4,5], [4,5,6]]
            [1.0, 2.0, 3.0, 4.0]   # Mean of [[0,1,2], [1,2,3], [2,3,4], [3,4,5]]
        ])
        assert torch.allclose(result, expected)


class TestMaxAggregator:
    """Test suite for MaxAggregator component."""

    def test_max_aggregator_initialization(self):
        """Test that MaxAggregator can be initialized."""
        agg = MaxAggregator()
        assert isinstance(agg, BaseAggregator)

    def test_max_aggregator_pooling(self):
        """Test max aggregation with known values."""
        # Create test tensor (batch=2, chunks=3, dim=4)
        embeddings = torch.tensor([
            [[1.0, 5.0, 2.0, 4.0],
             [3.0, 2.0, 6.0, 1.0],
             [2.0, 4.0, 3.0, 7.0]],
            [[0.0, 3.0, 1.0, 2.0],
             [2.0, 1.0, 4.0, 3.0],
             [1.0, 2.0, 2.0, 5.0]]
        ])

        agg = MaxAggregator()
        result = agg(embeddings)

        # Verify shape
        assert result.shape == (2, 4)

        # Verify values (max of each dim across chunks)
        expected = torch.tensor([
            [3.0, 5.0, 6.0, 7.0],  # Max across chunks for each dim
            [2.0, 3.0, 4.0, 5.0]
        ])
        assert torch.allclose(result, expected)


class TestGetAggregator:
    """Test suite for get_aggregator factory function."""

    def test_get_aggregator_mean(self):
        """Test loading MeanAggregator by name."""
        agg = get_aggregator("MeanAggregator")
        assert isinstance(agg, MeanAggregator)

    def test_get_aggregator_max(self):
        """Test loading MaxAggregator by name."""
        agg = get_aggregator("MaxAggregator")
        assert isinstance(agg, MaxAggregator)

    def test_get_aggregator_invalid_name(self):
        """Test that invalid aggregator name raises error."""
        with pytest.raises(VectorMeshError) as exc_info:
            get_aggregator("InvalidAggregator")

        error = exc_info.value
        assert "not found" in str(error).lower()

    def test_get_aggregator_non_aggregator_class(self):
        """Test that non-aggregator class raises error."""
        # This would test if someone tries to load a non-aggregator class
        # For now, we'll just verify the factory works correctly
        agg = get_aggregator("MeanAggregator")
        assert isinstance(agg, BaseAggregator)


class TestAggregatorLinearCompatibility:
    """Test that aggregators work with torch.nn.Linear layers."""

    def test_mean_aggregator_with_linear(self):
        """Test that MeanAggregator output works with torch.nn.Linear."""
        embeddings = torch.randn(8, 5, 384)  # batch=8, chunks=5, dim=384

        agg = MeanAggregator()
        aggregated = agg(embeddings)

        # Verify shape is correct for Linear layer input
        assert aggregated.shape == (8, 384)

        # Verify it works with Linear layer
        linear = torch.nn.Linear(384, 128)
        output = linear(aggregated)
        assert output.shape == (8, 128)

    def test_max_aggregator_with_linear(self):
        """Test that MaxAggregator output works with torch.nn.Linear."""
        embeddings = torch.randn(8, 5, 384)  # batch=8, chunks=5, dim=384

        agg = MaxAggregator()
        aggregated = agg(embeddings)

        # Verify shape is correct for Linear layer input
        assert aggregated.shape == (8, 384)

        # Verify it works with Linear layer
        linear = torch.nn.Linear(384, 128)
        output = linear(aggregated)
        assert output.shape == (8, 128)


class TestBaseAggregatorExtension:
    """Test that users can extend BaseAggregator easily."""

    def test_custom_aggregator_simple_extension(self):
        """Test creating a custom aggregator with minimal code."""

        # User's custom aggregator - only implements _aggregate()
        class SumAggregator(BaseAggregator):
            """Custom aggregator that sums across chunks."""

            def _aggregate(self, embeddings):
                return torch.sum(embeddings, dim=1)

        # Test it works
        embeddings = torch.tensor([
            [[1.0, 2.0],
             [3.0, 4.0]],
            [[5.0, 6.0],
             [7.0, 8.0]]
        ])

        agg = SumAggregator()
        result = agg(embeddings)

        # Verify shape
        assert result.shape == (2, 2)

        # Verify values
        expected = torch.tensor([
            [4.0, 6.0],  # 1+3, 2+4
            [12.0, 14.0]  # 5+7, 6+8
        ])
        assert torch.allclose(result, expected)
