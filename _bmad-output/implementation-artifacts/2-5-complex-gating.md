# Story 2.5: Complex Gating & MoE

Status: review

## Story

As an advanced user,
I want complex gating mechanisms (Highway networks, MoE, learnable gates),
so that I can build sophisticated routing and mixture-of-experts architectures.

## Acceptance Criteria

**AC1: Highway Network Pattern (Learned Skip)**
**Given** a transform component and input data
**When** I create `Highway(transform=MyLayer(), gate_fn=learned_gate)`
**Then** it computes: `G * transform(input) + (1-G) * input` (single input, two paths)
**And** G is computed by learned gate function from input
**And** implements the Highway network pattern from Srivastava et al. (2015)
**And** this is essentially a learned skip connection with gating
**And** supports both 2D and 3D tensor flows

**AC2: Mixture of Experts (MoE) with Sparse Routing**
**Given** multiple expert components and a routing network
**When** I create `MoE(experts=[Expert1(), Expert2(), Expert3()], router=my_router)`
**Then** router computes routing probabilities per input
**And** combines expert outputs weighted by routing probabilities
**And** supports sparse routing (top-k experts)
**And** handles load balancing across experts

**AC3: Learnable Gate with Context-Based Routing**
**Given** a component to gate and separate context signal
**When** I create `LearnableGate(component=MyLayer(), router=gate_network)`
**Then** component processes the input data: `output = component(input)`
**And** router computes gating from context: `gate = router(context)`
**And** result is context-gated output: `gate * output`
**And** gradients flow through both component and router
**And** integrates with PyTorch autograd
**And** separates data transformation from gating decision

**AC4: Switch Component for Parallel Combining**
**Given** two outputs from Parallel branches
**When** I create `Switch(gate_fn=learned_router)` after Parallel
**Then** it computes: `G * input_1 + (1-G) * input_2` (two inputs combined)
**And** gate_fn can see both inputs to make routing decision
**And** validates that both inputs have matching shapes
**And** raises educational errors on shape mismatches
**And** this is a parallel combiner (different from Highway's single-input pattern)

## Tasks / Subtasks

- [x] Task 1: Implement Highway component (AC: 1)
  - [x] Create `Highway` class inheriting VectorMeshComponent
  - [x] Add fields: `transform: Any`, `gate_fn: Callable`
  - [x] Implement G * transform(x) + (1-G) * x pattern
  - [x] Add shape validation (transform output must match input)
  - [x] Support learnable gate networks
  - [x] Add educational errors for dimension mismatches

- [x] Task 2: Implement MoE component (AC: 2)
  - [x] Create `MoE` class with multiple expert components
  - [x] Implement router network for computing routing probabilities
  - [x] Add sparse routing (top-k expert selection)
  - [x] Implement load balancing tracking (for future use)
  - [x] Combine expert outputs with routing weights
  - [x] Add educational errors for routing issues

- [x] Task 3: Implement LearnableGate with context (AC: 3)
  - [x] Create `LearnableGate` with separate input and context
  - [x] Add fields: `component: Any`, `router: Callable[[NDTensor], Union[float, NDTensor]]`
  - [x] Implement `__call__(input_data, context)` signature
  - [x] Route based on context: `gate = router(context)`
  - [x] Apply to component output: `gate * component(input)`
  - [x] Ensure gradients flow through both paths
  - [x] Add educational errors for shape mismatches

- [x] Task 4: Implement Switch combiner (AC: 4)
  - [x] Create `Switch` class for parallel combining
  - [x] Add field: `gate_fn: Callable[[Tuple[NDTensor, NDTensor]], NDTensor]`
  - [x] Implement `__call__` accepting tuple from Parallel
  - [x] Compute gate from both inputs (can see both for decision)
  - [x] Formula: `G * input_1 + (1-G) * input_2`
  - [x] Validate both inputs have matching shapes
  - [x] Add educational errors for shape mismatches

- [x] Task 5: Integration and testing
  - [x] Unit tests for Highway (single input, learned skip)
  - [x] Unit tests for Switch (two inputs, parallel combiner)
  - [x] Unit tests for MoE with different numbers of experts
  - [x] Unit tests for LearnableGate with context-based routing
  - [x] Test gradient flow through all learnable components
  - [x] Integration tests simulating Serial/Parallel behavior
  - [x] Test Switch after Parallel (simulated Parallel output)
  - [x] Test LearnableGate with separate input/context tensors
  - [x] All tests passing (18/18)

## Dev Notes

### Critical Implementation Requirements

**🔥 ADVANCED GATING - Learnable, complex routing, MoE architectures!**

**Scope: Complex mechanisms deferred from Story 2.4:**
- Highway networks: G * x' + (1-G) * x pattern
- Mixture of Experts with sparse routing
- Learnable gate networks (not just router functions)
- Add aggregator for manual residual composition

**Dependencies:**
- Story 2.1: Serial/Parallel combinators
- Story 2.4: Basic Skip/Gate patterns

**Architecture Patterns:**
- New file: `src/vectormesh/components/advanced_gating.py`
- All components inherit VectorMeshComponent
- Support PyTorch autograd for learnable components
- Frozen Pydantic models with learnable sub-networks

### Highway Network Design

**Pattern: Single Input, Two Paths (Learned Skip Connection)**

```python
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
```

### Switch Combiner Design

**Pattern: Two Inputs from Parallel (Parallel Output Combiner)**

```python
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
```

### Mixture of Experts Design

**Pattern: Sparse Routing to Multiple Expert Networks**

```python
from typing import List

class MoE(VectorMeshComponent, frozen=True):
    """Mixture of Experts with sparse routing.

    Routes inputs to top-k experts based on learned routing network.
    Combines expert outputs weighted by routing probabilities.

    Args:
        experts: List of expert components
        router: Network computing routing logits
        top_k: Number of experts to activate per input (sparse routing)
        load_balance: Whether to compute load balancing loss

    Example:
        >>> # 4 experts, route to top-2
        >>> moe = MoE(
        ...     experts=[Expert1(), Expert2(), Expert3(), Expert4()],
        ...     router=RouterNetwork(input_dim=768, num_experts=4),
        ...     top_k=2
        ... )

    Shapes:
        Input: TwoDTensor ℝ^{B×E_in}
        Output: TwoDTensor ℝ^{B×E_out} (expert output dimension)
    """
    experts: List[VectorMeshComponent]
    router: Callable[[NDTensor], NDTensor]  # Returns routing logits
    top_k: int = 2
    load_balance: bool = True

    def __call__(self, input_data: NDTensor) -> NDTensor:
        batch_size = input_data.shape[0]

        # Compute routing logits
        routing_logits = self.router(input_data)  # Shape: [B, num_experts]

        # Top-k sparse routing
        top_k_values, top_k_indices = torch.topk(
            routing_logits, k=self.top_k, dim=1
        )
        routing_weights = F.softmax(top_k_values, dim=1)  # Normalize top-k

        # Compute expert outputs (only for selected experts)
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # Check if this expert is in top-k for any batch item
            mask = (top_k_indices == i).any(dim=1)
            if mask.any():
                output = expert(input_data[mask])
                expert_outputs.append((i, output, mask))

        # Combine expert outputs with routing weights
        # (Implementation details for weighted combination)
        combined_output = self._combine_expert_outputs(
            expert_outputs, routing_weights, top_k_indices, batch_size
        )

        return combined_output

    def _combine_expert_outputs(self, expert_outputs, weights, indices, batch_size):
        """Combine sparse expert outputs with routing weights."""
        # Implementation for weighted combination
        pass
```

### LearnableGate with Context Design

**Pattern: Context-Based Routing (Separate Input and Context)**

```python
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
        >>>
        >>> # In pipeline with separate context branch
        >>> pipeline = Serial([
        ...     Parallel([
        ...         DataBranch(),      # Processes main data
        ...         ContextBranch()    # Computes context signal
        ...     ]),
        ...     # Use second branch output to gate first branch
        ...     lambda outputs: LearnableGate(
        ...         component=lambda x: outputs[0],  # Data from first branch
        ...         router=lambda ctx: gate_net(ctx)
        ...     )(outputs[0], outputs[1])
        ... ])

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
```

### Integration with Previous Stories

**Story 2.4 (Basic Gating):**
- Highway is learned Skip (single input, two internal paths)
- Switch is parallel combiner (two inputs from Parallel)
- LearnableGate extends Gate with context-based routing
- All build on Skip/Gate foundation from Story 2.4

**Story 2.1 (Combinators):**
- All components work in Serial/Parallel
- MoE can contain Parallel branches as experts
- Highway can be composed with other components

### Testing Strategy

**Unit Tests:**
- Highway with learned gates (single input, two paths)
- Switch with two inputs from Parallel
- Switch gate_fn seeing both inputs for decision
- MoE with 2-4 experts, top-1 and top-2 routing
- LearnableGate with context-based routing
- LearnableGate gradient flow validation (both input and context)
- Shape mismatch error validation for all components

**Integration Tests:**
- Highway in Serial pipelines (learned skip)
- Switch after Parallel: `Serial([Parallel([...]), Switch(...)])`
- LearnableGate with separate input/context from Parallel branches
- MoE with experts containing Serial compositions
- Complex routing: Highway → MoE → Switch
- Gradient flow through all learnable components (Highway, Switch, LearnableGate, MoE)

### References

**Research Papers:**
- Highway Networks: Srivastava et al. (2015)
- Mixture of Experts: Shazeer et al. (2017) - Outrageously Large Neural Networks
- Sparse Routing: Switch Transformers (Google, 2021)

**Previous Stories:**
- [Story 2.4: Gating Mechanisms](./2-4-gating-mechanisms.md) - Basic Skip/Gate
- [Story 2.1: Combinators](./2-1-combinators-serial-parallel.md) - Composition framework

**User Innovation:**
- Separated simple (2.4) from complex (2.5) gating
- Deferred magic and complexity to dedicated story
- Add aggregator for flexible manual composition

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.5 (Implementation)

### Debug Log References

- All 18 unit tests passing: `tests/components/test_complex_gating.py`
- Linting passed: `ruff check` (all issues fixed)
- Integration: Highway, Switch, LearnableGate, MoE all working

### Completion Notes List

**✅ STORY 2.5 IMPLEMENTED - COMPLEX GATING & MoE**

**Implementation Summary:**
- ✅ **Highway**: Single-input learned skip connection (`G * T(x) + (1-G) * x`)
- ✅ **Switch**: Two-input parallel combiner (`G * input_1 + (1-G) * input_2`)
- ✅ **LearnableGate**: Context-based routing (`router(context) * component(input)`)
- ✅ **MoE**: Mixture of Experts with sparse top-k routing

**Technical Achievements:**
- All 4 components follow VectorMeshComponent pattern
- Educational error handling with hint/fix fields
- Full gradient flow support for learnable components
- Shape validation for 2D and 3D tensors
- 18 comprehensive unit tests (100% passing)
- Integration with existing gating.py module
- Linting compliance (ruff check passed)

**Key Implementation Details:**
- Highway preserves input dimensions (like residual connections)
- Switch validates both inputs have matching shapes
- LearnableGate supports different dimensionality for input vs context
- MoE implements efficient sparse routing with top-k expert selection
- All components use `Any` for flexibility (testing/composition)
- Proper sigmoid activation for gate values (G ∈ [0,1])

**Test Coverage:**
- Highway: Basic learned skip, shape validation, 3D tensors, error handling
- Switch: Parallel combining, shape mismatches, gate sees both inputs
- LearnableGate: Context routing, different dims, gradient flow, shape validation
- MoE: Basic routing, top-k selection, different batch routing, shape validation
- Integration: Highway in pipelines, Switch after Parallel simulation, LearnableGate with context

**Date Completed:** 2026-01-08

**Core Components (Corrected Architecture):**
- **Highway**: `G * T(x) + (1-G) * x` - Single input, learned skip connection
- **Switch**: `G * input_1 + (1-G) * input_2` - Two inputs, parallel combiner
- **LearnableGate**: Context-based routing - `router(context) * component(input)`
- **MoE**: Sparse routing to multiple experts with top-k selection

**Critical Architectural Clarifications:**
- Highway ≠ Switch: Highway has ONE input (two internal paths), Switch has TWO inputs
- LearnableGate separates concerns: data transformation vs routing decision
- Switch is a parallel output combiner (follows Parallel in pipeline)

**Deferred from Story 2.4:**
- Complex routing mechanisms (Highway, Switch, MoE)
- Context-based learnable gates (input vs context separation)
- MoE architectures with sparse routing

**Dependencies:**
- Story 2.1: Serial/Parallel composition
- Story 2.4: Basic Skip/Gate patterns

### File List

**Files Modified:**
- `/Users/rgrouls/code/MADS/courses/packages/vectormesh/src/vectormesh/components/gating.py` - Added Highway, Switch, MoE, LearnableGate (369 lines added)
- `/Users/rgrouls/code/MADS/courses/packages/vectormesh/src/vectormesh/components/__init__.py` - Exported Highway, Switch, MoE, LearnableGate

**Files Created:**
- `/Users/rgrouls/code/MADS/courses/packages/vectormesh/tests/components/test_complex_gating.py` - 380 lines, 18 comprehensive tests
