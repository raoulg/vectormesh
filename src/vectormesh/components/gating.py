"""Gating mechanisms: Skip connections and signal modulation.

This module implements gating components for VectorMesh:
- Skip: Residual connections with add+norm pattern (ResNet, Transformers)
- Gate: Signal modulation with router functions
- Highway: Single-input learned skip connection (Story 2.5)
- Switch: Two-input parallel combiner (Story 2.5)
- LearnableGate: Context-based routing (Story 2.5)
- MoE: Mixture of Experts with sparse routing (Story 2.5)
"""

from typing import Optional, Callable, Union, Any, List, Tuple
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


class Highway(VectorMeshComponent):
    """Highway network with learned skip connection.

    Single input with two paths:
    - Transform path: T(x) processes the input
    - Identity path: x passes through unchanged
    - Learned gate G decides the mixture

    Formula: G * T(x) + (1-G) * x

    This is the Highway network pattern from Srivastava et al. (2015).
    It's essentially a learned skip connection where the gate determines
    how much to use the transformation vs pass-through.

    Args:
        transform: Transformation component (like H in Highway paper)
        gate_fn: Function computing gate logits from input (like T in paper)
                 Will be passed through sigmoid to get G ∈ [0,1]

    Example:
        >>> # Highway with learned gate
        >>> highway = Highway(
        ...     transform=TransformLayer(768, 768),
        ...     gate_fn=lambda x: gate_network(x)  # Learnable network
        ... )
        >>>
        >>> # In pipeline
        >>> pipeline = Serial([
        ...     TwoDVectorizer("bert"),
        ...     Highway(transform=MLPLayer(), gate_fn=learned_gate)
        ... ])

    Shapes:
        Input: TwoDTensor ℝ^{B×E} or ThreeDTensor ℝ^{B×C×E}
        Output: Same shape as input (residual connection preserves dims)

    Literature:
        Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015).
        Highway Networks. arXiv:1505.00387
    """

    transform: Any  # Transform component
    gate_fn: Callable[[NDTensor], NDTensor]  # Gate function

    def __call__(self, input_data: NDTensor) -> NDTensor:
        """Apply Highway network formula.

        Args:
            input_data: Input tensor

        Returns:
            NDTensor: G * T(x) + (1-G) * x

        Raises:
            VectorMeshError: If transform changes shape or gate shape mismatched

        Shapes:
            Input: ℝ^{B×E} or ℝ^{B×C×E}
            Output: Same shape as input
        """
        # Compute transform output
        transform_output = self.transform(input_data)

        # Validate shapes match (Highway preserves dimensions)
        if transform_output.shape != input_data.shape:
            raise VectorMeshError(
                message=f"Highway transform changed shape: {input_data.shape} → {transform_output.shape}",
                hint="Highway networks require transform to preserve dimensions (like residual connections)",
                fix="Ensure transform output matches input shape, or use Skip with projection"
            )

        # Compute gate: G = sigmoid(gate_fn(input))
        gate_logits = self.gate_fn(input_data)
        gate = torch.sigmoid(gate_logits)

        # Validate gate shape
        if gate.shape != input_data.shape:
            raise VectorMeshError(
                message=f"Highway gate shape mismatch: gate {gate.shape} != input {input_data.shape}",
                hint="Gate function must return tensor with same shape as input",
                fix=f"Ensure gate_fn returns shape {input_data.shape}"
            )

        # Highway formula: G * T(x) + (1-G) * x
        output = gate * transform_output + (1 - gate) * input_data

        return output


class Switch(VectorMeshComponent):
    """Switch between two parallel outputs with learned gating.

    Takes TWO separate inputs (typically from Parallel output tuple)
    and gates between them with a learned weight.

    Formula: G * input_1 + (1-G) * input_2

    This is a parallel combiner (different from Highway which has
    single input with two internal paths).

    Args:
        gate_fn: Function computing gate from both inputs
                 Takes tuple (input_1, input_2) and returns gate logits
                 Will be passed through sigmoid to get G ∈ [0,1]

    Example:
        >>> # Switch between two parallel branches
        >>> pipeline = Serial([
        ...     Parallel([Branch1(), Branch2()]),  # → (out1, out2)
        ...     Switch(gate_fn=lambda inputs: router_net(torch.cat(inputs, dim=-1)))
        ... ])
        >>>
        >>> # Context-based switching
        >>> def context_gate(inputs):
        ...     inp1, inp2 = inputs
        ...     # Decide based on properties of both inputs
        ...     return gate_network(torch.stack([inp1.mean(), inp2.mean()]))
        >>>
        >>> switch = Switch(gate_fn=context_gate)

    Shapes:
        Input: Tuple[TwoDTensor ℝ^{B×E}, TwoDTensor ℝ^{B×E}]
        Output: TwoDTensor ℝ^{B×E} (weighted combination)

    Notes:
        - Both inputs must have identical shapes
        - Gate function can see both inputs to make decision
        - Commonly used after Parallel to combine branches
    """

    gate_fn: Callable[[Tuple[NDTensor, NDTensor]], NDTensor]

    def __call__(self, inputs: Tuple[NDTensor, NDTensor]) -> NDTensor:
        """Switch between two inputs with learned gating.

        Args:
            inputs: Tuple of two tensors from Parallel

        Returns:
            NDTensor: G * input_1 + (1-G) * input_2

        Raises:
            VectorMeshError: If input shapes mismatch or gate shape incompatible

        Shapes:
            Input: (ℝ^{B×E}, ℝ^{B×E})
            Output: ℝ^{B×E}
        """
        # Unpack parallel outputs
        input_1, input_2 = inputs

        # Validate shapes match
        if input_1.shape != input_2.shape:
            raise VectorMeshError(
                message=f"Switch shape mismatch: input_1 {input_1.shape} != input_2 {input_2.shape}",
                hint="Both parallel branches must produce same shape for Switch",
                fix="Add projection or aggregation to align dimensions before Switch"
            )

        # Compute gate (can see both inputs for decision)
        gate_logits = self.gate_fn((input_1, input_2))
        gate = torch.sigmoid(gate_logits)

        # Validate gate shape (should broadcast or match)
        if gate.shape != input_1.shape:
            # Allow scalar gates
            if gate.numel() != 1 and gate.shape != input_1.shape:
                raise VectorMeshError(
                    message=f"Switch gate shape mismatch: gate {gate.shape} cannot broadcast to {input_1.shape}",
                    hint="Gate must be scalar or match input shape",
                    fix=f"gate_fn should return scalar or shape {input_1.shape}"
                )

        # Switch formula: G * input_1 + (1-G) * input_2
        output = gate * input_1 + (1 - gate) * input_2

        return output


class LearnableGate(VectorMeshComponent):
    """Learnable gate with context-based routing.

    Separates the gating decision (based on context signal) from the
    data transformation (based on input). This is more flexible than
    gating based on the input itself.

    Pattern:
        output = component(input)      # Transform the data
        gate = router(context)          # Route based on context
        result = gate * output          # Apply context-based gating

    This pattern is useful for attention-like mechanisms, query-key-value
    patterns, or any scenario where gating should be based on a different
    signal than the data being processed.

    Args:
        component: Processes the input data
        router: Computes gating values from context signal
                Can return scalar (global gate) or tensor (element-wise)

    Example:
        >>> # Attention-style gating (query gates keys)
        >>> gate = LearnableGate(
        ...     component=TransformKeys(),
        ...     router=lambda context: gate_network(context)
        ... )
        >>> result = gate(input=keys, context=query)

    Shapes:
        Input: TwoDTensor ℝ^{B×E_in}
        Context: TwoDTensor ℝ^{B×E_ctx}
        Output: TwoDTensor ℝ^{B×E_out} (same as component output, gated)

    Notes:
        - Context can have different dimensionality than input
        - Router must output shape compatible with component output (for gating)
        - Gradients flow through both component and router
    """

    component: Any  # Component to process input
    router: Callable[[NDTensor], Union[float, NDTensor]]  # Context → gate

    # Allow nn.Module fields for learnable routers
    model_config = {"arbitrary_types_allowed": True}

    def __call__(self, input_data: NDTensor, context: NDTensor) -> NDTensor:
        """Apply context-based gating.

        Args:
            input_data: Data to process
            context: Context signal for routing decision

        Returns:
            Context-gated output: router(context) * component(input)

        Raises:
            VectorMeshError: If gate shape incompatible with output

        Shapes:
            Input: ℝ^{B×E_in}
            Context: ℝ^{B×E_ctx}
            Output: ℝ^{B×E_out}
        """
        # Transform input data
        output = self.component(input_data)

        # Compute gate from CONTEXT (not input!)
        gate_value = self.router(context)

        # Validate gate shape if tensor
        if isinstance(gate_value, torch.Tensor):
            if gate_value.shape != output.shape:
                # Allow broadcasting for scalar-like gates
                if gate_value.numel() != 1 and gate_value.shape != output.shape:
                    raise VectorMeshError(
                        message=f"LearnableGate shape mismatch: gate {gate_value.shape} cannot broadcast to output {output.shape}",
                        hint="Router must return scalar or tensor matching component output shape",
                        fix=f"Router should return scalar or shape {output.shape}"
                    )

        # Apply context-based gating
        gated_output = gate_value * output

        return gated_output


class MoE(VectorMeshComponent):
    """Mixture of Experts with sparse routing.

    Routes inputs to top-k experts based on learned routing network.
    Combines expert outputs weighted by routing probabilities.

    Args:
        experts: List of expert components
        router: Function computing routing logits [batch, num_experts]
        top_k: Number of experts to activate per input (sparse routing)
        load_balance: Whether to track load balancing (for future use)

    Example:
        >>> # 4 experts, route to top-2
        >>> moe = MoE(
        ...     experts=[Expert1(), Expert2(), Expert3(), Expert4()],
        ...     router=lambda x: router_network(x),
        ...     top_k=2
        ... )

    Shapes:
        Input: TwoDTensor ℝ^{B×E_in}
        Output: TwoDTensor ℝ^{B×E_out} (expert output dimension)

    Literature:
        Shazeer, N., et al. (2017). Outrageously Large Neural Networks:
        The Sparsely-Gated Mixture-of-Experts Layer. ICLR 2017.
    """

    experts: List[Any]  # List of expert components
    router: Callable[[NDTensor], NDTensor]  # Router function
    top_k: int = 2  # Number of experts to activate
    load_balance: bool = False  # Load balancing (future use)

    def __call__(self, input_data: NDTensor) -> NDTensor:
        """Apply MoE routing and combine expert outputs.

        Args:
            input_data: Input tensor

        Returns:
            NDTensor: Weighted combination of top-k expert outputs

        Raises:
            VectorMeshError: If router shape incorrect

        Shapes:
            Input: ℝ^{B×E}
            Output: ℝ^{B×E}
        """
        batch_size = input_data.shape[0]
        num_experts = len(self.experts)

        # Compute routing logits: [batch, num_experts]
        routing_logits = self.router(input_data)

        # Validate router output shape
        expected_router_shape = (batch_size, num_experts)
        if routing_logits.shape != expected_router_shape:
            raise VectorMeshError(
                message=f"MoE router shape mismatch: {routing_logits.shape} != {expected_router_shape}",
                hint="Router must return [batch, num_experts] logits",
                fix=f"Router should return shape {expected_router_shape} (batch_size={batch_size}, num_experts={num_experts})"
            )

        # Compute routing probabilities with softmax
        routing_probs = F.softmax(routing_logits, dim=-1)  # [batch, num_experts]

        # Select top-k experts per input
        top_k_values, top_k_indices = torch.topk(routing_probs, k=self.top_k, dim=-1)
        # top_k_values: [batch, top_k]
        # top_k_indices: [batch, top_k]

        # Normalize top-k weights to sum to 1
        top_k_weights = top_k_values / top_k_values.sum(dim=-1, keepdim=True)

        # Initialize output tensor
        # Get output shape from first expert (assume all experts have same output shape)
        dummy_output = self.experts[0](input_data[:1])
        output = torch.zeros(batch_size, *dummy_output.shape[1:], device=input_data.device)

        # Compute weighted sum of top-k expert outputs per batch item
        for b in range(batch_size):
            # Get top-k experts for this batch item
            expert_indices = top_k_indices[b]  # [top_k]
            expert_weights = top_k_weights[b]  # [top_k]

            # Compute weighted sum of selected experts
            input_b = input_data[b:b+1]  # [1, ...] preserve batch dim

            for k in range(self.top_k):
                expert_idx = expert_indices[k].item()
                weight = expert_weights[k]
                expert = self.experts[expert_idx]

                expert_output = expert(input_b)  # [1, ...]
                output[b] += weight * expert_output.squeeze(0)

        return output
