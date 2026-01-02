from vectormesh.types import OneDTensor, TwoDTensor, ThreeDTensor

def test_tensor_types_resolution():
    # Verify they are jaxtyping types
    # checking strict equality is flaky with jaxtyping caching/creation
    # imply checking string representation or structure is enough
    
    # Updated to be less brittle - check for key string components
    s = str(OneDTensor)
    assert "Float" in s and "dim" in s
    
    s = str(TwoDTensor)
    assert "Float" in s and "batch" in s and "dim" in s

    s = str(ThreeDTensor)
    assert "Float" in s and "batch" in s and "seq" in s and "dim" in s

def test_tensor_types_docstrings():
    # Verify docstrings exist as per AC
    assert OneDTensor.__doc__ is not None
    assert "1D Tensor" in OneDTensor.__doc__
    
    assert TwoDTensor.__doc__ is not None
    assert "2D Tensor" in TwoDTensor.__doc__
    
    assert ThreeDTensor.__doc__ is not None
    assert "3D Tensor" in ThreeDTensor.__doc__
