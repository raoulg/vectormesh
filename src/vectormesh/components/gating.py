"""Gating mechanisms: Skip connections and signal modulation.

This module implements basic gating components for VectorMesh:
- Skip: Residual connections with add+norm pattern (ResNet, Transformers)
- Gate: Signal modulation with router functions (Highway, GRU-style gates)

These are foundation components - complex gating (MoE, learnable gates) in Story 2.5.
"""

from typing import Optional, Callable, Union, Any
import torch
import torch.nn.functional as F

from vectormesh.types import VectorMeshComponent, NDTensor, VectorMeshError


class Skip(VectorMeshComponent):
    """Residual skip connection with add+norm pattern.

    Implements the standard residual pattern from ResNet and Transformers:
        LayerNorm(input + main(input))

    Or with projection when dimensions change:
        LayerNorm(projection(input) + main(input))

    The projection must be manually provided - no automatic dimension detection
    to avoid magic behavior that confuses users.

    Args:
        main: Component for the main transformation path
        projection: Optional projection to match dimensions (manual only, no auto-magic)

    Example:
        >>> # Simple skip - shapes must match
        >>> skip = Skip(main=Serial([Layer1(), Layer2()]))
        >>>
        >>> # Skip with projection (like ResNet when dimensions change)
        >>> skip = Skip(
        ...     main=DownsampleLayer(),  # 768 → 512
        ...     projection=LinearProjection(768, 512)
        ... )

    Shapes:
        Input: TwoDTensor ℝ^{B×E_in} or ThreeDTensor ℝ^{B×C×E_in}
        Output: Same type as input, ℝ^{B×E_out} or ℝ^{B×C×E_out}
    """

    main: Any  # Any component with __call__ method (flexibility for testing/composition)
    projection: Optional[Any] = None  # Optional projection component

    def __call__(self, input_data: NDTensor) -> NDTensor:
        """Apply residual connection with add+norm.

        Args:
            input_data: Input tensor (2D or 3D)

        Returns:
            NDTensor: LayerNorm(residual + main_output)

        Raises:
            VectorMeshError: If main output shape doesn't match residual shape

        Shapes:
            Input: ℝ^{B×E} or ℝ^{B×C×E}
            Output: Same shape as input (residual preserved)
        """
        # Compute main path transformation
        main_output = self.main(input_data)

        # Compute residual (with projection if provided)
        if self.projection is not None:
            residual = self.projection(input_data)
        else:
            residual = input_data

        # Validate shapes match before addition
        if main_output.shape != residual.shape:
            raise VectorMeshError(
                message=f"Skip shape mismatch: main output {main_output.shape} != residual {residual.shape}",
                hint="Main path changed dimensions without projection. In ResNet/Transformers, "
                     "skip connections require dimension matching.",
                fix=f"Add projection parameter: Skip(main=..., projection=LinearProjection("
                    f"{residual.shape[-1]}, {main_output.shape[-1]}))"
            )

        # Add + Norm (Transformer pattern)
        # Add residual and main output
        added = torch.add(residual, main_output)

        # Apply LayerNorm along feature dimensions (all dims except batch)
        # normalized_shape should be the dimensions to normalize over (last N dims)
        normalized = F.layer_norm(added, normalized_shape=added.shape[1:])

        return normalized


class Gate(VectorMeshComponent):
    """Signal gating with learnable or computed routing.

    Implements router-based signal modulation:
        router(input) * component(input)

    The router function computes gating values from the input tensor.
    This is the foundation for Highway networks, GRU-style gates, and MoE routing.

    Router is always required - no default pass-through because that's not a gate,
    it's just the component itself.

    Args:
        component: Component to gate (apply routing to)
        router: Function that computes gate values from input
                Returns float (scalar gating) or tensor (per-element gating)

    Example:
        >>> # Simple learned gate (scalar)
        >>> def my_router(x: NDTensor) -> float:
        ...     return torch.sigmoid(learned_weight * x.mean()).item()
        >>>
        >>> gate = Gate(component=MyLayer(), router=my_router)
        >>>
        >>> # Per-element gating (like Highway networks)
        >>> def highway_router(x: NDTensor) -> NDTensor:
        ...     return torch.sigmoid(transform_gate(x))
        >>>
        >>> gate = Gate(component=Transform(), router=highway_router)

    Shapes:
        Input: TwoDTensor ℝ^{B×E} or ThreeDTensor ℝ^{B×C×E}
        Output: Same shape as component output (modulated by gate)
    """

    component: Any  # Any component with __call__ method
    router: Callable[[NDTensor], Union[float, NDTensor]]  # Router function (required)

    def __call__(self, input_data: NDTensor) -> NDTensor:
        """Apply gating to component output.

        Args:
            input_data: Input tensor

        Returns:
            NDTensor: gate_value * component(input)

        Raises:
            VectorMeshError: If router returns incompatible tensor shape

        Shapes:
            Input: ℝ^{B×E} or ℝ^{B×C×E}
            Output: Same shape as component output
        """
        # Compute component output
        output = self.component(input_data)

        # Compute gate value from input (not output - router sees original signal)
        gate_value = self.router(input_data)

        # Validate gate_value shape if it's a tensor
        if isinstance(gate_value, torch.Tensor):
            # Check if shapes match or can broadcast
            if gate_value.shape != output.shape:
                # Allow broadcasting for scalar-like gates (single element tensors)
                if gate_value.numel() != 1 and gate_value.shape != output.shape:
                    raise VectorMeshError(
                        message=f"Gate shape mismatch: router returned {gate_value.shape}, "
                                f"cannot broadcast to component output {output.shape}",
                        hint="Router returned tensor with incompatible shape for gating. "
                             "Gate values must either match output shape exactly or be scalar-like.",
                        fix=f"Router should return: (1) scalar float, (2) scalar tensor (shape []), "
                            f"or (3) tensor matching output shape {output.shape}"
                    )

        # Apply gating: multiply gate value with output
        # PyTorch handles broadcasting automatically for scalars and compatible shapes
        gated_output = gate_value * output

        return gated_output
