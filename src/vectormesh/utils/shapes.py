import jaxtyping
import functools
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation
from jaxtyping import TypeCheckError
from ..types import VectorMeshError

def check_shapes(func):
    """
    Runtime shape checking decorator.
    Wraps jaxtyping+beartype and converts errors to VectorMeshError
    for better developer experience.
    """
    # Force fresh decoration with explicit typechecker
    decorated_func = jaxtyping.jaxtyped(typechecker=beartype)(func)
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return decorated_func(*args, **kwargs)
        except (BeartypeCallHintParamViolation, TypeCheckError) as e:
            # Re-raise as VectorMeshError with helpful context
            raise VectorMeshError(
                f"Shape mismatch in {func.__name__}: {str(e)}",
                hint="Check the tensor shapes against the type hints.",
                fix=f"Inspect the inputs passed to {func.__name__}."
            ) from e
    return wrapper
