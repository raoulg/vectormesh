"""Morphism composition validation system for VectorMesh pipelines."""

from typing import Any, List, Type, Union, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from vectormesh.types import TwoDTensor, ThreeDTensor
from vectormesh.errors import VectorMeshError


class TensorDimensionality(Enum):
    """Tensor dimensionality levels - objects in the VectorMesh category."""
    TEXT = "text"          # List[str]
    TWO_D = "2d"          # TwoDTensor: (batch, embed)
    THREE_D = "3d"        # ThreeDTensor: (batch, chunks, embed)
    FOUR_D = "4d"         # FourDTensor: (batch, chunks, tokens, embed) - future


@dataclass
class Morphism:
    """Defines a morphism between tensor types in the VectorMesh category."""
    source: TensorDimensionality      # Source object
    target: TensorDimensionality      # Target object
    component_name: str
    description: str


@dataclass
class ValidationResult:
    """Result of morphism composition validation."""
    is_valid: bool
    message: str
    hint: Optional[str] = None
    fix: Optional[str] = None
    morphism_chain: Optional[List[Morphism]] = None

    @classmethod
    def success(cls, morphism_chain: List[Morphism], message: str = "Pipeline composition successful") -> 'ValidationResult':
        return cls(
            is_valid=True,
            message=message,
            morphism_chain=morphism_chain
        )

    @classmethod
    def error(cls, message: str, hint: str = None, fix: str = None) -> 'ValidationResult':
        return cls(
            is_valid=False,
            message=message,
            hint=hint,
            fix=fix
        )


class MorphismComposition:
    """Validates morphism composition in VectorMesh pipelines."""

    def __init__(self):
        # Component type registry - maps component classes to their morphisms
        self.morphism_compositions = {}
        self._register_core_morphisms()

    def _register_core_morphisms(self):
        """Register morphisms for core VectorMesh components."""
        # Import here to avoid circular imports
        try:
            from vectormesh.components.vectorizers import TwoDVectorizer, ThreeDVectorizer
            from vectormesh.components.aggregation import MeanAggregator

            self.morphism_compositions.update({
                TwoDVectorizer: Morphism(
                    source=TensorDimensionality.TEXT,
                    target=TensorDimensionality.TWO_D,
                    component_name="TwoDVectorizer",
                    description="Text → sentence embedding morphism"
                ),
                ThreeDVectorizer: Morphism(
                    source=TensorDimensionality.TEXT,
                    target=TensorDimensionality.THREE_D,
                    component_name="ThreeDVectorizer",
                    description="Text → chunk embedding morphism"
                ),
                MeanAggregator: Morphism(
                    source=TensorDimensionality.THREE_D,
                    target=TensorDimensionality.TWO_D,
                    component_name="MeanAggregator",
                    description="Chunk aggregation morphism (3D → 2D)"
                )
            })
        except ImportError:
            # Components not available yet - will register when used
            pass

    def register_morphism(self, component_class: Type, morphism: Morphism):
        """Register a morphism for a custom component."""
        self.morphism_compositions[component_class] = morphism

    def get_component_morphism(self, component: Any) -> Optional[Morphism]:
        """Get morphism for a component instance."""
        component_type = type(component)
        return self.morphism_compositions.get(component_type)

    def validate_composition(self, components: List[Any]) -> ValidationResult:
        """Validate that components can be composed sequentially.

        Args:
            components: List of component instances to compose

        Returns:
            ValidationResult with success/failure and detailed morphism information
        """
        if not components:
            return ValidationResult.error(
                "Composition cannot be empty",
                hint="Add at least one component to the composition",
                fix="Serial(components=[TwoDVectorizer(), MeanAggregator()])"
            )

        morphism_chain = []

        for i, component in enumerate(components):
            # Get component morphism
            component_morphism = self.get_component_morphism(component)

            if not component_morphism:
                return ValidationResult.error(
                    message=f"Unknown component type: {type(component).__name__} at position {i}",
                    hint=f"Component {type(component).__name__} is not registered in the morphism composition system",
                    fix=f"Register the morphism: register_morphism({type(component).__name__}, Morphism(source=..., target=...))"
                )

            morphism_chain.append(component_morphism)

            # Check composability with previous component
            if i > 0:
                prev_morphism = morphism_chain[i-1]
                curr_morphism = component_morphism

                if prev_morphism.target != curr_morphism.source:
                    return ValidationResult.error(
                        message=f"Non-composable morphisms at positions {i-1} → {i}",
                        hint=f"{prev_morphism.component_name} has target {prev_morphism.target.value}, but {curr_morphism.component_name} has source {curr_morphism.source.value}",
                        fix=self._suggest_composition_fix(prev_morphism.target, curr_morphism.source, i)
                    )

        return ValidationResult.success(
            morphism_chain,
            f"Composition valid: {' ∘ '.join([m.component_name for m in morphism_chain])}"
        )

    def validate_parallel_branches(self, branches: List[Any], input_type: TensorDimensionality = None) -> ValidationResult:
        """Validate that all parallel branches accept the same input type.

        Args:
            branches: List of component instances for parallel execution
            input_type: Expected input type (inferred from first branch if None)

        Returns:
            ValidationResult with success/failure and branch flow information
        """
        if not branches:
            return ValidationResult.error(
                "Parallel branches cannot be empty",
                hint="Add at least one branch to the parallel combinator",
                fix="Parallel(branches=[TwoDVectorizer('model1'), TwoDVectorizer('model2')])"
            )

        morphism_chain = []
        expected_input = input_type

        for i, branch in enumerate(branches):
            branch_morphism = self.get_component_morphism(branch)

            if not branch_morphism:
                return ValidationResult.error(
                    message=f"Unknown branch component type: {type(branch).__name__} at position {i}",
                    hint=f"Branch component {type(branch).__name__} is not registered in the morphism composition system",
                    fix=f"Register the morphism: register_morphism({type(branch).__name__}, Morphism(source=..., target=...))"
                )

            morphism_chain.append(branch_morphism)

            # Set expected input from first branch
            if expected_input is None:
                expected_input = branch_morphism.source

            # Validate branch input compatibility
            if branch_morphism.source != expected_input:
                return ValidationResult.error(
                    message=f"Incompatible branch input at position {i}",
                    hint=f"Expected {expected_input.value} input, but branch {i} ({branch_morphism.component_name}) expects {branch_morphism.source.value}",
                    fix=f"Ensure all branches accept {expected_input.value} input, or add preprocessing to branch {i}"
                )

        output_types = [m.target.value for m in morphism_chain]
        return ValidationResult.success(
            morphism_chain,
            f"Parallel branches valid: input={expected_input.value}, outputs=({', '.join(output_types)})"
        )

    def validate_concat_compatibility(self, tensor_types: List[TensorDimensionality]) -> ValidationResult:
        """Validate that tensors can be concatenated together.

        Args:
            tensor_types: List of tensor dimensionalities to concatenate

        Returns:
            ValidationResult indicating if concat is valid
        """
        if not tensor_types:
            return ValidationResult.error(
                "Cannot concatenate empty tensor list",
                hint="Provide at least one tensor for concatenation",
                fix="Ensure parallel branches produce outputs before concatenation"
            )

        if len(set(tensor_types)) > 1:
            type_names = [t.value for t in tensor_types]
            return ValidationResult.error(
                message=f"Cannot concatenate mixed dimensionalities",
                hint=f"Got mixed types: {', '.join(type_names)}. All tensors must have same dimensionality.",
                fix="Use aggregators to normalize all tensors to same dimensionality before concatenation"
            )

        concat_type = tensor_types[0]
        return ValidationResult.success(
            [],
            f"Concatenation valid: {len(tensor_types)} tensors of type {concat_type.value}"
        )

    def _suggest_composition_fix(self, target: TensorDimensionality, source: TensorDimensionality, position: int) -> str:
        """Suggest how to fix non-composable morphisms."""
        if target == TensorDimensionality.THREE_D and source == TensorDimensionality.TWO_D:
            return f"Insert MeanAggregator() between components {position-1} and {position} to enable composition (3D → 2D)"
        elif target == TensorDimensionality.TWO_D and source == TensorDimensionality.THREE_D:
            return f"Insert ChunkExpander() between components {position-1} and {position} to enable composition (2D → 3D)"
        elif target == TensorDimensionality.TEXT and source in [TensorDimensionality.TWO_D, TensorDimensionality.THREE_D]:
            return f"Cannot compose components {position-1} and {position} - text target cannot map to tensor source"
        else:
            return f"Ensure component {position-1} target matches component {position} source for composition"


# Global composition validator instance
_composition = MorphismComposition()

def validate_composition(components: List[Any]) -> ValidationResult:
    """Convenience function to validate component composition."""
    return _composition.validate_composition(components)

def validate_parallel(branches: List[Any]) -> ValidationResult:
    """Convenience function to validate parallel branches."""
    return _composition.validate_parallel_branches(branches)

def register_morphism(component_class: Type, morphism: Morphism):
    """Register a custom component's morphism."""
    _composition.register_morphism(component_class, morphism)