from pydantic import BaseModel, ConfigDict

class VectorMeshComponent(BaseModel):
    """
    Base class for all VectorMesh components.
    Enforces strict validation and immutable configuration.
    """
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
