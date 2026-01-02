import pytest
import torch

from vectormesh.types import OneDTensor
from vectormesh.utils import check_shapes

@check_shapes
def my_vector_func(x: OneDTensor) -> OneDTensor:
    return x

def test_check_shapes_validation():
    # Correct shape: (dim,)
    x = torch.zeros(10)
    assert my_vector_func(x) is x

    # Incorrect shape: (batch, dim) -> (1, 10)
    # Should raise error
    # Note: We now wrap it, so it should be VectorMeshError
    from vectormesh.types import VectorMeshError
    with pytest.raises(VectorMeshError):
        my_vector_func(torch.zeros(1, 10))
