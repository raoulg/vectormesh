from typing import Optional


class VectorMeshError(Exception):
    """
    Base exception for all VectorMesh errors.

    Includes educational hints and fixes to help users understand
    and resolve tensor flow and composition issues.

    Args:
        message: Primary error message
        hint: Educational hint about what went wrong
        fix: Suggested fix or next steps
    """

    def __init__(
        self, message: str, hint: Optional[str] = None, fix: Optional[str] = None
    ):
        super().__init__(message)
        self.hint = hint
        self.fix = fix
