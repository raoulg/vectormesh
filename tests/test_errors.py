import pytest
from vectormesh.types import VectorMeshError

def test_vectormesh_error_fields():
    # Verify friendlier error messages
    err = VectorMeshError(
        "Invalid configuration",
        hint="Check the README",
        fix="Set config.x = 1"
    )
    
    assert str(err) == "Invalid configuration"
    assert err.hint == "Check the README"
    assert err.fix == "Set config.x = 1"
    
def test_vectormesh_error_inheritance():
    # Must inherit from Exception
    with pytest.raises(VectorMeshError):
        raise VectorMeshError("Something broke")
