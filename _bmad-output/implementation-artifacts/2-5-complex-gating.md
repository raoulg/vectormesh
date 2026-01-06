# Story 2.5: Complex Gating & MoE

Status: backlog

## Story

As an advanced user,
I want complex gating mechanisms (Highway networks, MoE, learnable gates),
so that I can build sophisticated routing and mixture-of-experts architectures.

## Acceptance Criteria

**AC1: Highway Network Pattern**
**Given** a transform component and input data
**When** I create `Highway(transform=MyLayer(), gate_fn=learned_gate)`
**Then** it computes: `G * transform(input) + (1-G) * input`
**And** G is computed by learned gate function
**And** implements the Highway network pattern from literature
**And** supports both 2D and 3D tensor flows

**AC2: Mixture of Experts (MoE) with Sparse Routing**
**Given** multiple expert components and a routing network
**When** I create `MoE(experts=[Expert1(), Expert2(), Expert3()], router=my_router)`
**Then** router computes routing probabilities per input
**And** combines expert outputs weighted by routing probabilities
**And** supports sparse routing (top-k experts)
**And** handles load balancing across experts

**AC3: Learnable Gate Components**
**Given** a component to gate
**When** I create `LearnableGate(component=MyLayer(), gate_network=GateNet())`
**Then** gate_network is a learnable neural network
**And** it computes gating values from input features
**And** gradients flow through both component and gate network
**And** integrates with PyTorch autograd

**AC4: Add Aggregator for Manual Composition**
**Given** multiple tensor outputs from Parallel
**When** I create `Add(normalize=True)`
**Then** it sums tensors element-wise
**And** optionally applies LayerNorm after addition
**And** validates all input tensors have matching shapes
**And** raises educational errors on shape mismatches

## Tasks / Subtasks

- [ ] Task 1: Implement Highway component (AC: 1)
  - [ ] Create `Highway` class inheriting VectorMeshComponent
  - [ ] Add fields: `transform: VectorMeshComponent`, `gate_fn: Callable`
  - [ ] Implement G * transform(x) + (1-G) * x pattern
  - [ ] Add shape validation (transform output must match input)
  - [ ] Support learnable gate networks
  - [ ] Add educational errors for dimension mismatches

- [ ] Task 2: Implement MoE component (AC: 2)
  - [ ] Create `MoE` class with multiple expert components
  - [ ] Implement router network for computing routing probabilities
  - [ ] Add sparse routing (top-k expert selection)
  - [ ] Implement load balancing loss computation
  - [ ] Combine expert outputs with routing weights
  - [ ] Add educational errors for routing issues

- [ ] Task 3: Implement LearnableGate (AC: 3)
  - [ ] Create `LearnableGate` with neural network gate
  - [ ] Ensure gradients flow through gate network
  - [ ] Support different gate architectures (linear, MLP)
  - [ ] Add Pydantic validation for gate_network
  - [ ] Test with PyTorch autograd

- [ ] Task 4: Implement Add aggregator (AC: 4)
  - [ ] Create `Add` class for element-wise addition
  - [ ] Add `normalize: bool` parameter for optional LayerNorm
  - [ ] Validate input tensor shapes match
  - [ ] Support adding 2+ tensors
  - [ ] Add educational errors for shape mismatches

- [ ] Task 5: Integration and testing
  - [ ] Unit tests for Highway network
  - [ ] Unit tests for MoE with different numbers of experts
  - [ ] Unit tests for LearnableGate with gradient flow
  - [ ] Unit tests for Add aggregator
  - [ ] Integration tests with Serial/Parallel from Story 2.1
  - [ ] Integration tests with Skip/Gate from Story 2.4
  - [ ] Test complex compositions (Highway + MoE)

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

**Pattern: Learned Gating Between Transform and Identity**

```python
class Highway(VectorMeshComponent, frozen=True):
    """Highway network with learned gating.

    Implements: G * transform(input) + (1-G) * input
    where G = sigmoid(gate_fn(input))

    This is the Highway network pattern from Srivastava et al. (2015).

    Args:
        transform: Transformation component (like H in Highway paper)
        gate_fn: Function/network computing gate values (like T in paper)

    Example:
        >>> # Simple highway layer
        >>> highway = Highway(
        ...     transform=DenseLayer(512, 512),
        ...     gate_fn=GateNetwork(512)
        ... )
        >>>
        >>> # Highway in Serial pipeline
        >>> pipeline = Serial([
        ...     TwoDVectorizer("bert"),
        ...     Highway(transform=TransformLayer(), gate_fn=LearnedGate())
        ... ])

    Shapes:
        Input: TwoDTensor ℝ^{B×E}
        Output: TwoDTensor ℝ^{B×E} (same as input - residual)
    """
    transform: VectorMeshComponent
    gate_fn: Callable[[NDTensor], NDTensor]

    def __call__(self, input_data: NDTensor) -> NDTensor:
        # Compute transform
        transform_output = self.transform(input_data)

        # Validate shapes match (Highway requires same dims)
        if transform_output.shape != input_data.shape:
            raise VectorMeshError(
                message=f"Highway transform changed shape: {input_data.shape} → {transform_output.shape}",
                hint="Highway networks require transform to preserve dimensions",
                fix="Use Skip with projection if dimensions must change"
            )

        # Compute gate (G)
        gate = torch.sigmoid(self.gate_fn(input_data))

        # Highway formula: G * transform(x) + (1-G) * x
        return gate * transform_output + (1 - gate) * input_data
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

### Add Aggregator Design

**Pattern: Element-wise Addition with Optional Normalization**

```python
class Add(VectorMeshComponent, frozen=True):
    """Element-wise tensor addition aggregator.

    Sums multiple tensors element-wise with optional normalization.
    Typically used with Parallel outputs for manual residual connections.

    Args:
        normalize: Whether to apply LayerNorm after addition

    Example:
        >>> # Manual DualPath pattern (Parallel + Add)
        >>> pipeline = Serial([
        ...     Parallel([Path1(), Path2()]),  # → (ℝ^{B×E}, ℝ^{B×E})
        ...     Add(normalize=True)  # → ℝ^{B×E}
        ... ])

    Shapes:
        Input: Tuple of N tensors, all with shape ℝ^{B×E}
        Output: Single tensor ℝ^{B×E}
    """
    normalize: bool = True

    def __call__(self, inputs: tuple[NDTensor, ...]) -> NDTensor:
        # Validate all tensors have same shape
        first_shape = inputs[0].shape
        for i, tensor in enumerate(inputs[1:], start=1):
            if tensor.shape != first_shape:
                raise VectorMeshError(
                    message=f"Add shape mismatch: tensor {i} has shape {tensor.shape}, expected {first_shape}",
                    hint="All inputs to Add must have identical shapes",
                    fix="Use projection or reshape components to align dimensions before Add"
                )

        # Sum all tensors
        result = inputs[0]
        for tensor in inputs[1:]:
            result = torch.add(result, tensor)

        # Optional normalization
        if self.normalize:
            result = F.layer_norm(result, normalized_shape=result.shape[1:])

        return result
```

### Integration with Previous Stories

**Story 2.4 (Basic Gating):**
- Highway builds on Skip pattern (residual connection)
- LearnableGate extends Gate with neural networks
- Add enables manual composition of Skip-like patterns

**Story 2.1 (Combinators):**
- All components work in Serial/Parallel
- MoE can contain Parallel branches as experts
- Highway can be composed with other components

### Testing Strategy

**Unit Tests:**
- Highway with learned gates
- MoE with 2-4 experts, top-1 and top-2 routing
- LearnableGate with gradient flow validation
- Add with 2-5 input tensors
- Add with normalize=True/False
- Shape mismatch error validation

**Integration Tests:**
- Highway + Skip in same pipeline
- MoE with experts containing Serial compositions
- Complex routing: Highway → MoE → Add
- Gradient flow through learnable components

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

Claude Sonnet 4.5

### Debug Log References

N/A - Story creation phase

### Completion Notes List

**✅ STORY 2.5 CREATED - COMPLEX GATING & MoE**

**Core Components:**
- Highway: G * transform(x) + (1-G) * x pattern
- MoE: Sparse routing to multiple experts
- LearnableGate: Neural network gates with autograd
- Add: Element-wise aggregator with optional normalization

**Deferred from Story 2.4:**
- Complex routing mechanisms
- Learnable gates (vs simple router functions)
- MoE architectures
- Add aggregator for manual composition

**Dependencies:**
- Story 2.1: Serial/Parallel composition
- Story 2.4: Basic Skip/Gate patterns

### File List

**Files to Create:**
- `/Users/rgrouls/code/MADS/courses/packages/vectormesh/src/vectormesh/components/advanced_gating.py`
- `/Users/rgrouls/code/MADS/courses/packages/vectormesh/tests/test_advanced_gating.py`

**Files to Modify:**
- `/Users/rgrouls/code/MADS/courses/packages/vectormesh/src/vectormesh/components/__init__.py` - Add Highway, MoE, LearnableGate, Add exports
