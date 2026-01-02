from typing import Optional

class VectorMeshError(Exception):
    """
    Base exception for all VectorMesh errors.
    Includes educational hints and fixes.
    """
    def __init__(self, message: str, hint: Optional[str] = None, fix: Optional[str] = None):
        super().__init__(message)
        self.hint = hint
        self.fix = fix
