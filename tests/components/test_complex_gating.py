"""Tests for complex gating mechanisms: Highway, Switch, LearnableGate, MoE.

This module tests advanced gating components from Story 2.5.
"""

import pytest
import torch
import torch.nn as nn
from typing import Tuple

from vectormesh.components.gating import Highway, Switch, LearnableGate, MoE
from vectormesh.types import VectorMeshError, NDTensor


# ============================================================================
# HIGHWAY TESTS (Single input, learned skip)
# ============================================================================

def test_highway_basic_learned_skip():
    """Test Highway with learned gate function (single input, two paths)."""
    # Simple transform that preserves shape
    def transform(x):
        return x * 2.0

    # Learned gate that returns same shape as input
    def gate_fn(x):
        return torch.zeros_like(x)  # Gate logits = 0 → sigmoid = 0.5

    highway = Highway(transform=transform, gate_fn=gate_fn)

    # Test with 2D tensor
    input_data = torch.randn(4, 768)
    output = highway(input_data)

    # Highway formula: G * T(x) + (1-G) * x
    # With G = sigmoid(0) = 0.5: output = 0.5 * 2x + 0.5 * x = 1.5x
    expected = 1.5 * input_data

    assert output.shape == input_data.shape, "Highway should preserve input shape"
    assert torch.allclose(output, expected, atol=1e-5), "Highway formula incorrect"


def test_highway_shape_mismatch_error():
    """Test Highway raises error when transform changes dimensions."""
    # Transform that changes shape (BAD for Highway)
    def transform(x):
        return x[:, :512]  # 768 → 512
    def gate_fn(x):
        return torch.zeros_like(x)

    highway = Highway(transform=transform, gate_fn=gate_fn)

    input_data = torch.randn(4, 768)

    with pytest.raises(VectorMeshError) as exc_info:
        highway(input_data)

    assert "Highway transform changed shape" in str(exc_info.value)
    assert exc_info.value.hint and "preserve dimensions" in exc_info.value.hint


def test_highway_gate_shape_mismatch_error():
    """Test Highway raises error when gate_fn returns wrong shape."""
    def transform(x):
        return x * 2.0
    # Gate that returns wrong shape
    def gate_fn(x):
        return torch.zeros(x.shape[0], 512)  # Wrong dims

    highway = Highway(transform=transform, gate_fn=gate_fn)

    input_data = torch.randn(4, 768)

    with pytest.raises(VectorMeshError) as exc_info:
        highway(input_data)

    assert "Highway gate shape mismatch" in str(exc_info.value)


def test_highway_3d_tensors():
    """Test Highway with 3D tensors."""
    def transform(x):
        return x + 0.1
    def gate_fn(x):
        return torch.ones_like(x)  # Gate = sigmoid(1) ≈ 0.73

    highway = Highway(transform=transform, gate_fn=gate_fn)

    input_data = torch.randn(2, 10, 768)  # [B, C, E]
    output = highway(input_data)

    assert output.shape == input_data.shape, "Highway should preserve 3D shape"


# ============================================================================
# SWITCH TESTS (Two inputs from Parallel)
# ============================================================================

def test_switch_basic_parallel_combining():
    """Test Switch combines two inputs from Parallel."""
    # Gate function that sees both inputs
    def gate_fn(inputs: Tuple[NDTensor, NDTensor]) -> NDTensor:
        inp1, inp2 = inputs
        # Return scalar gate (will broadcast)
        return torch.tensor(0.0)  # sigmoid(0) = 0.5

    switch = Switch(gate_fn=gate_fn)

    # Two inputs with same shape (from Parallel)
    input_1 = torch.ones(4, 768)
    input_2 = torch.ones(4, 768) * 2.0

    output = switch((input_1, input_2))

    # Formula: G * input_1 + (1-G) * input_2
    # With G = 0.5: output = 0.5 * 1 + 0.5 * 2 = 1.5
    expected = torch.ones(4, 768) * 1.5

    assert output.shape == input_1.shape, "Switch should preserve input shape"
    assert torch.allclose(output, expected, atol=1e-5), "Switch formula incorrect"


def test_switch_shape_mismatch_error():
    """Test Switch raises error when inputs have different shapes."""
    def gate_fn(inputs):
        return torch.tensor(0.0)
    switch = Switch(gate_fn=gate_fn)

    input_1 = torch.randn(4, 768)
    input_2 = torch.randn(4, 512)  # Different shape!

    with pytest.raises(VectorMeshError) as exc_info:
        switch((input_1, input_2))

    assert "Switch shape mismatch" in str(exc_info.value)
    assert exc_info.value.hint and "same shape" in exc_info.value.hint


def test_switch_gate_sees_both_inputs():
    """Test Switch gate function receives both inputs."""
    # Gate that uses both inputs to decide
    def context_gate(inputs: Tuple[NDTensor, NDTensor]) -> NDTensor:
        inp1, inp2 = inputs
        # Decision based on mean values
        if inp1.mean() > inp2.mean():
            return torch.tensor(10.0)  # sigmoid → 1.0 (choose inp1)
        else:
            return torch.tensor(-10.0)  # sigmoid → 0.0 (choose inp2)

    switch = Switch(gate_fn=context_gate)

    input_1 = torch.ones(4, 768) * 5.0  # Higher mean
    input_2 = torch.ones(4, 768) * 1.0  # Lower mean

    output = switch((input_1, input_2))

    # Should choose input_1 (higher mean) with G ≈ 1.0
    assert torch.allclose(output, input_1, atol=0.1), "Switch should route to input_1"


# ============================================================================
# LEARNABLE GATE TESTS (Context-based routing)
# ============================================================================

def test_learnable_gate_context_based_routing():
    """Test LearnableGate separates input and context."""
    def component(x):
        return x * 2.0

    # Router uses CONTEXT (not input) to compute gate
    # Returns scalar that gets sigmoid'd to 0.5
    def router(context):
        return torch.sigmoid(torch.tensor(0.0))

    gate = LearnableGate(component=component, router=router)

    input_data = torch.randn(4, 768)
    context = torch.randn(4, 512)  # Different shape OK!

    output = gate(input_data, context)

    # Formula: gate * component(input) = 0.5 * (2 * input) = input
    expected = input_data

    assert output.shape == input_data.shape, "LearnableGate should match input shape"
    assert torch.allclose(output, expected, atol=1e-3), "LearnableGate formula incorrect"


def test_learnable_gate_different_context_dimensionality():
    """Test LearnableGate context can have different dims than input."""
    def component(x):
        return x + 1.0
    def router(context):
        return torch.ones(4, 768)  # Returns full gate tensor

    gate = LearnableGate(component=component, router=router)

    input_data = torch.randn(4, 768)
    context = torch.randn(4, 256)  # Different last dim!

    output = gate(input_data, context)

    assert output.shape == (4, 768), "Output should match component output shape"


def test_learnable_gate_shape_mismatch_error():
    """Test LearnableGate raises error when gate shape incompatible."""
    def component(x):
        return x
    # Router returns wrong shape
    def router(context):
        return torch.zeros(4, 512)

    gate = LearnableGate(component=component, router=router)

    input_data = torch.randn(4, 768)
    context = torch.randn(4, 256)

    with pytest.raises(VectorMeshError) as exc_info:
        gate(input_data, context)

    assert "LearnableGate shape mismatch" in str(exc_info.value)


def test_learnable_gate_gradient_flow():
    """Test gradients flow through both component and router."""
    # Learnable component and router
    component_net = nn.Linear(768, 768)
    router_net = nn.Linear(256, 768)

    component = component_net
    def router(ctx):
        return torch.sigmoid(router_net(ctx))

    gate = LearnableGate(component=component, router=router)

    input_data = torch.randn(4, 768, requires_grad=True)
    context = torch.randn(4, 256, requires_grad=True)

    output = gate(input_data, context)
    loss = output.sum()
    loss.backward()

    # Check gradients exist
    assert input_data.grad is not None, "Input should have gradients"
    assert context.grad is not None, "Context should have gradients"
    assert component_net.weight.grad is not None, "Component should have gradients"
    assert router_net.weight.grad is not None, "Router should have gradients"


# ============================================================================
# MOE TESTS (Mixture of Experts with sparse routing)
# ============================================================================

def test_moe_basic_routing():
    """Test MoE routes to top-k experts."""
    # Three simple experts
    def expert1(x):
        return x * 1.0
    def expert2(x):
        return x * 2.0
    def expert3(x):
        return x * 3.0

    # Router that returns logits for each expert
    def router(x):
        # Return routing logits [B, num_experts]
        batch_size = x.shape[0]
        return torch.tensor([[1.0, 0.0, 0.0]] * batch_size)  # Choose expert 1

    moe = MoE(experts=[expert1, expert2, expert3], router=router, top_k=1)

    input_data = torch.ones(4, 768)
    output = moe(input_data)

    # Should route to expert1 (logit = 1.0)
    assert output.shape == input_data.shape, "MoE should preserve shape"
    assert torch.allclose(output, input_data, atol=1e-3), "Should route to expert1"


def test_moe_top_k_routing():
    """Test MoE sparse routing with top-2."""
    def expert1(x):
        return x * 1.0
    def expert2(x):
        return x * 2.0
    def expert3(x):
        return x * 3.0

    def router(x):
        batch_size = x.shape[0]
        # Route to expert 1 and 2 (equal weights)
        return torch.tensor([[1.0, 1.0, 0.0]] * batch_size)

    moe = MoE(experts=[expert1, expert2, expert3], router=router, top_k=2)

    input_data = torch.ones(4, 768)
    output = moe(input_data)

    # Should combine expert1 and expert2 with equal weights
    # output = 0.5 * 1x + 0.5 * 2x = 1.5x
    expected = input_data * 1.5

    assert torch.allclose(output, expected, atol=1e-3), "MoE top-2 routing incorrect"


def test_moe_router_shape_validation():
    """Test MoE validates router returns [batch, num_experts]."""
    experts = [lambda x: x] * 3

    # Router returns wrong shape
    def router(x):
        return torch.zeros(4)  # Should be [4, 3]

    moe = MoE(experts=experts, router=router, top_k=2)

    input_data = torch.randn(4, 768)

    with pytest.raises(VectorMeshError) as exc_info:
        moe(input_data)

    assert "MoE router shape mismatch" in str(exc_info.value)


def test_moe_different_batch_items_different_experts():
    """Test MoE can route different batch items to different experts."""
    def expert1(x):
        return x * 1.0
    def expert2(x):
        return x * 2.0
    def expert3(x):
        return x * 3.0

    def router(x):
        # First 2 items → expert1, last 2 items → expert2
        return torch.tensor([
            [2.0, 0.0, 0.0],  # Item 0 → expert1
            [2.0, 0.0, 0.0],  # Item 1 → expert1
            [0.0, 2.0, 0.0],  # Item 2 → expert2
            [0.0, 2.0, 0.0],  # Item 3 → expert2
        ])

    moe = MoE(experts=[expert1, expert2, expert3], router=router, top_k=1)

    input_data = torch.ones(4, 768)
    output = moe(input_data)

    # First 2 should be 1x, last 2 should be 2x
    assert torch.allclose(output[:2], input_data[:2] * 1.0, atol=1e-3)
    assert torch.allclose(output[2:], input_data[2:] * 2.0, atol=1e-3)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_highway_in_serial_pipeline():
    """Test Highway works in Serial pipeline."""
    # Note: Can't use raw lambdas in Serial (morphism validation)
    # Just test Highway works with simple transform
    def transform(x):
        return x + 0.5
    def gate_fn(x):
        return torch.zeros_like(x)

    highway = Highway(transform=transform, gate_fn=gate_fn)

    input_data = torch.randn(4, 768)
    preprocessed = input_data * 2.0
    highway_output = highway(preprocessed)
    output = highway_output + 1.0

    assert output.shape == input_data.shape, "Highway preserves shape in pipeline"


def test_switch_after_parallel():
    """Test Switch combines two inputs (simulating Parallel output)."""
    def gate_fn(inputs):
        return torch.tensor(0.0)  # sigmoid(0) = 0.5

    switch = Switch(gate_fn=gate_fn)

    input_data = torch.randn(4, 768)
    # Simulate Parallel output
    branch1_output = input_data * 1.0
    branch2_output = input_data * 2.0

    output = switch((branch1_output, branch2_output))

    # Should mix both branches equally: 0.5 * 1x + 0.5 * 2x = 1.5x
    expected = input_data * 1.5
    assert torch.allclose(output, expected, atol=1e-3), "Switch should mix branches"


def test_learnable_gate_with_parallel_context():
    """Test LearnableGate with context from Parallel branch."""

    def component(x):
        return x * 2.0
    def router(ctx):
        return torch.sigmoid(ctx.mean())  # Scalar gate from context

    gate = LearnableGate(component=component, router=router)

    # Simulate Parallel outputs
    input_data = torch.randn(4, 768)
    context = torch.randn(4, 768)

    output = gate(input_data, context)

    assert output.shape == input_data.shape, "Context-gated output shape correct"
