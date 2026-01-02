import pytest
from pydantic import ValidationError
from vectormesh.base import VectorMeshComponent

class MyComponent(VectorMeshComponent):
    x: int
    y: str

def test_vectormesh_component_validation():
    # Valid
    c = MyComponent(x=1, y="hello")
    assert c.x == 1
    assert c.y == "hello"

    # Invalid type -> ValidationError
    with pytest.raises(ValidationError):
        MyComponent(x="not an int", y="hello")

def test_vectormesh_component_immutability():
    c = MyComponent(x=1, y="hello")
    
    # Frozen -> modification raises Validation Error (or TypeError in some pydantic versions, 
    # generally ValidationError in v2 frozen logic or TypeError)
    with pytest.raises(ValidationError):
        c.x = 2
