"""Serial and Parallel combinators for component composition."""

from typing import Union, Any
from beartype.typing import List, Tuple
from jaxtyping import jaxtyped
from beartype import beartype as typechecker
from pydantic import Field, field_validator

from vectormesh.types import VectorMeshComponent, TwoDTensor, ThreeDTensor, NDTensor, VectorMeshError
from vectormesh.validation import validate_composition, validate_parallel


class Serial(VectorMeshComponent):
    """Sequential combinator that chains components in order.

    Implements sequential data flow where output of one component becomes
    input to the next component. Includes tensor flow validation to ensure
    component compatibility.

    Args:
        components: List of components to chain sequentially

    Example:
        ```python
        pipeline = Serial(components=[TwoDVectorizer(), MeanAggregator()])
        result = pipeline(["Hello world"])  # TwoDVectorizer → MeanAggregator
        ```

    Shapes:
        Input: List[str] or tensor (depends on first component)
        Output: Depends on final component in the chain
    """

    components: List[Any] = Field(description="List of components to chain")

    @field_validator('components')
    @classmethod
    def validate_component_chain(cls, components: List[Any]) -> List[Any]:
        """Validate that components can be chained sequentially."""
        # Allow empty components for testing, but validate at runtime
        if not components:
            return components

        # Skip validation if any component is itself a combinator (avoid circular dependency)
        from vectormesh.components.combinators import Serial, Parallel
        for component in components:
            if isinstance(component, (Serial, Parallel)):
                return components  # Skip validation for nested combinators

        validation_result = validate_composition(components)
        if not validation_result.is_valid:
            raise ValueError(
                f"Invalid component chain: {validation_result.message}. "
                f"Hint: {validation_result.hint}. Fix: {validation_result.fix}"
            )
        return components

    @jaxtyped(typechecker=typechecker)
    def __call__(self, input_data: Union[List[str], NDTensor]) -> NDTensor:
        """Execute components sequentially.

        Args:
            input_data: Input to first component (text list or tensor)

        Returns:
            NDTensor output from final component

        Raises:
            VectorMeshError: If components are incompatible or execution fails

        Shapes:
            Input: Depends on first component requirements
            Output: Depends on final component output
        """
        if not self.components:
            raise VectorMeshError(
                message="Serial combinator cannot be empty",
                hint="Add at least one component to the Serial chain",
                fix="Serial(components=[TwoDVectorizer(), MeanAggregator()])"
            )

        try:
            result = input_data
            for i, component in enumerate(self.components):
                try:
                    result = component(result)
                except Exception as e:
                    # Provide educational error for component incompatibility
                    raise VectorMeshError(
                        message=f"Component {i+1} ({type(component).__name__}) failed to process input from component {i}",
                        hint="Shape mismatch between components. Check tensor dimensions and compatibility.",
                        fix=f"Ensure component {i} output shape matches component {i+1} input requirements. Use aggregation if converting 3D→2D."
                    ) from e

            return result

        except VectorMeshError:
            # Re-raise our educational errors
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            raise VectorMeshError(
                message=f"Serial execution failed: {str(e)}",
                hint="Unexpected error in component chain execution",
                fix="Check that all components are properly configured and compatible"
            ) from e


class Parallel(VectorMeshComponent):
    """Parallel combinator that broadcasts input to multiple branches.

    Implements parallel data flow where input is sent to all branches
    simultaneously and outputs are collected as a tuple. Includes tensor flow
    validation to ensure all branches accept compatible input types.

    Args:
        branches: List of components or component chains to execute in parallel

    Example:
        ```python
        pipeline = Parallel(branches=[TwoDVectorizer("model1"), TwoDVectorizer("model2")])
        result = pipeline(["Hello"])  # Returns: tuple[TwoDTensor, TwoDTensor]
        ```

    Shapes:
        Input: Broadcast-compatible with all branches
        Output: Tuple of results from each branch
    """

    branches: List[Any] = Field(description="List of branches to execute in parallel")

    @field_validator('branches')
    @classmethod
    def validate_parallel_branches(cls, branches: List[Any]) -> List[Any]:
        """Validate that all branches accept compatible input types."""
        # Allow empty branches for testing, but validate at runtime
        if not branches:
            return branches

        # Skip validation if any branch is itself a combinator (avoid circular dependency)
        from vectormesh.components.combinators import Serial, Parallel
        for branch in branches:
            if isinstance(branch, (Serial, Parallel)):
                return branches  # Skip validation for nested combinators

        validation_result = validate_parallel(branches)
        if not validation_result.is_valid:
            raise ValueError(
                f"Invalid parallel branches: {validation_result.message}. "
                f"Hint: {validation_result.hint}. Fix: {validation_result.fix}"
            )
        return branches

    @jaxtyped(typechecker=typechecker)
    def __call__(self, input_data: Union[List[str], NDTensor]) -> Tuple[NDTensor, ...]:
        """Execute all branches in parallel and collect results.

        Args:
            input_data: Input to broadcast to all branches (text list or tensor)

        Returns:
            Tuple[NDTensor, ...] containing output from each branch

        Raises:
            VectorMeshError: If branches fail or are incompatible

        Shapes:
            Input: Must be compatible with all branch input requirements
            Output: tuple[branch1_output, branch2_output, ...]
        """
        if not self.branches:
            raise VectorMeshError(
                message="Parallel combinator cannot be empty",
                hint="Add at least one branch to the Parallel combinator",
                fix="Parallel(branches=[TwoDVectorizer('model1'), TwoDVectorizer('model2')])"
            )

        try:
            results = []
            for i, branch in enumerate(self.branches):
                try:
                    result = branch(input_data)
                    results.append(result)
                except Exception as e:
                    # Provide educational error for branch incompatibility
                    raise VectorMeshError(
                        message=f"Branch {i+1} ({type(branch).__name__}) failed to process input",
                        hint=f"Input not compatible with branch {i+1}. Check input format and branch requirements.",
                        fix="Ensure input is compatible with all branches. For mixed 2D/3D, normalize with aggregation."
                    ) from e

            return tuple(results)

        except VectorMeshError:
            # Re-raise our educational errors
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            raise VectorMeshError(
                message=f"Parallel execution failed: {str(e)}",
                hint="Unexpected error in parallel branch execution",
                fix="Check that input is compatible with all branches and branches are properly configured"
            ) from e