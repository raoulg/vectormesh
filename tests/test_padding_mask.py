"""Tests for the padding mask: padders return a `Padded` batch, and the mask travels
through components, `Serial`, `Parallel` and the connectors to every component that
takes one.

The values at padded positions are set to something other than zero in several tests:
the mask, not the values, says what is padding.
"""

import pytest
import torch
from jaxtyping import TypeCheckError
from torch import nn

from vectormesh.components import (
    Attention,
    AttentionAggregator,
    Concatenate2D,
    Concatenate3D,
    ConcatenateSequence,
    Conv1D,
    DynamicPadding,
    FixedPadding,
    LastAggregator,
    MaskedMeanAggregator,
    MeanAggregator,
    MoE,
    NeuralNet,
    Parallel,
    Pool1D,
    Projection,
    Recurrent,
    RNNAggregator,
    Serial,
    Skip,
    SparseMoE,
    TransformerBlock,
)
from vectormesh.types import Padded, VectorMeshError

DIM = 8


def _items(*lengths: int) -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(n, DIM) for n in lengths]


def _with_padding_set_to(batch: Padded, value: float) -> Padded:
    values = batch.values.masked_fill(~batch.mask.unsqueeze(-1), value)
    return Padded(values, batch.mask)


def _alone(module: nn.Module, item: torch.Tensor) -> torch.Tensor:
    """``module`` on one item without padding, batch axis removed."""
    out = module(item.unsqueeze(0))
    return (out.values if isinstance(out, Padded) else out)[0]


# --- padders ---------------------------------------------------------------


def test_fixed_padding_pads_cuts_and_masks():
    batch = FixedPadding(max_chunks=4)(_items(2, 6))
    assert batch.shape == (2, 4, DIM)
    assert batch.mask.tolist() == [[True, True, False, False], [True] * 4]
    assert batch.lengths.tolist() == [2, 4]
    assert (batch.values[0, 2:] == 0).all()


def test_dynamic_padding_masks_to_the_longest_item():
    batch = DynamicPadding()(_items(3, 1))
    assert batch.shape == (2, 3, DIM)
    assert batch.mask.tolist() == [[True, True, True], [True, False, False]]


def test_padded_checks_that_mask_and_values_agree():
    with pytest.raises(TypeCheckError):
        Padded(torch.zeros(2, 3, DIM), torch.ones(2, 4, dtype=torch.bool))


def test_padded_to_moves_values_and_keeps_the_mask_boolean():
    batch = FixedPadding(3)(_items(1, 2)).to(torch.float64)
    assert batch.values.dtype == torch.float64
    assert batch.mask.dtype == torch.bool


# --- the case a value-based mask misses: padding after a biased Projection ----


def test_masked_mean_after_a_biased_projection_ignores_padding():
    torch.manual_seed(1)
    items = _items(2, 5)
    model = Serial([Projection(DIM, 4), MaskedMeanAggregator()])
    out = model(FixedPadding(max_chunks=6)(items))
    for row, item in zip(out, items):
        assert torch.allclose(row, model.components[0](item).mean(dim=0), atol=1e-6)
    # the biased projection turns padding into the bias, so a plain mean differs
    plain = Serial([Projection(DIM, 4), MeanAggregator()])
    plain.load_state_dict(model.state_dict())
    assert not torch.allclose(out, plain(FixedPadding(max_chunks=6)(items)))


def test_masked_mean_called_directly_on_a_padder_output():
    items = _items(3, 1)
    out = MaskedMeanAggregator()(FixedPadding(4)(items))
    assert torch.allclose(out[1], items[1][0])


def test_masked_mean_without_a_mask_raises():
    with pytest.raises(VectorMeshError, match="mask"):
        MaskedMeanAggregator()(torch.randn(2, 3, DIM))


def test_rnn_aggregator_after_a_biased_projection_ignores_padding():
    torch.manual_seed(2)
    items = _items(2, 4)
    model = Serial([Projection(DIM, 4), RNNAggregator(4)]).eval()
    out = model(FixedPadding(6)(items))
    for row, item in zip(out, items):
        assert torch.allclose(row, _alone(model, item), atol=1e-6)


def test_attention_aggregator_gives_padding_no_weight():
    torch.manual_seed(3)
    agg = AttentionAggregator(DIM)
    batch = FixedPadding(5)(_items(2, 5))
    out = agg(_with_padding_set_to(batch, 3.0))
    assert torch.allclose(out[0], _alone(agg, batch.values[0, :2]), atol=1e-6)


# --- attention --------------------------------------------------------------


@pytest.mark.parametrize(
    "block", [TransformerBlock(DIM, num_heads=2), Attention(DIM, 2)]
)
def test_attention_excludes_padded_positions(block):
    torch.manual_seed(4)
    block.eval()
    items = _items(3, 5)
    batch = FixedPadding(5)(items)
    model = Serial([Projection(DIM, DIM), block])
    out = model(_with_padding_set_to(batch, 5.0))
    assert isinstance(out, Padded)
    assert torch.equal(out.mask, batch.mask)
    other = model(_with_padding_set_to(batch, -9.0))
    # the real positions do not depend on what the padding holds
    assert torch.allclose(out.values[batch.mask], other.values[batch.mask], atol=1e-5)


def test_transformer_block_on_padding_matches_the_item_alone():
    torch.manual_seed(5)
    block = TransformerBlock(DIM, num_heads=2).eval()
    items = _items(3, 5)
    out = block(FixedPadding(5)(items))
    assert torch.allclose(out.values[0, :3], _alone(block, items[0]), atol=1e-5)


def test_an_all_padding_item_stays_finite_in_attention():
    batch = Padded(torch.randn(2, 3, DIM), torch.tensor([[True] * 3, [False] * 3]))
    out = TransformerBlock(DIM, num_heads=2).eval()(batch)
    assert torch.isfinite(out.values).all()


# --- recurrence -------------------------------------------------------------


def test_recurrent_runs_over_real_steps_and_zeros_padding():
    torch.manual_seed(6)
    rnn = Recurrent(DIM, 4).eval()
    items = _items(2, 4)
    out = rnn(_with_padding_set_to(FixedPadding(4)(items), 2.0))
    assert torch.allclose(out.values[0, :2], _alone(rnn, items[0]), atol=1e-6)
    assert (out.values[0, 2:] == 0).all()


def test_last_after_recurrent_is_the_state_after_the_last_real_step():
    torch.manual_seed(7)
    model = Serial([Recurrent(DIM, 4), LastAggregator()]).eval()
    items = _items(2, 4)
    out = model(FixedPadding(4)(items))
    assert torch.allclose(out[0], _alone(model, items[0]), atol=1e-6)


# --- sliding windows --------------------------------------------------------


def test_conv1d_maps_the_mask_to_its_windows():
    conv = Conv1D(DIM, 4, kernel_size=3, stride=2, padding=1)
    batch = Padded(
        torch.randn(1, 7, DIM),
        torch.tensor([[True, True, True, False, False, False, False]]),
    )
    out = conv(batch)
    # windows over positions [-1..1], [1..3], [3..5], [5..7]: the first two read real ones
    assert out.mask.tolist() == [[True, True, False, False]]
    assert out.shape[1] == conv.output_length(7)


def test_pool1d_maps_the_mask_to_its_windows():
    batch = Padded(torch.randn(1, 6, DIM), torch.tensor([[True] * 3 + [False] * 3]))
    assert Pool1D(2)(batch).mask.tolist() == [[True, True, False]]


def test_a_component_that_changes_the_length_without_output_mask_raises():
    class Halve(Projection):
        def forward(self, tensors, mask=None):
            return tensors[:, ::2]

    with pytest.raises(VectorMeshError, match="output_mask"):
        Halve(DIM, DIM)(FixedPadding(4)(_items(1, 4)))


# --- Parallel and the connectors --------------------------------------------


def test_concatenate3d_keeps_a_chunk_real_when_any_branch_has_it():
    a = Padded(torch.randn(2, 3, 4), torch.tensor([[True, False, False]] * 2))
    b = Padded(torch.randn(2, 3, 2), torch.tensor([[True, True, False]] * 2))
    out = Serial([Parallel([Projection(4, 4), Projection(2, 2)]), Concatenate3D()])(
        (a, b)
    )
    assert out.shape == (2, 3, 6)
    assert out.mask.tolist() == [[True, True, False]] * 2


def test_concatenate_sequence_joins_the_masks_and_a_plain_branch_is_all_real():
    a = FixedPadding(3)(_items(1, 2))
    b = torch.randn(2, 2, DIM)
    out = ConcatenateSequence()((a, b))
    assert out.mask.tolist() == [
        [True, False, False, True, True],
        [True, True, False, True, True],
    ]


def test_rnn_skips_padding_in_the_middle_after_concatenate_sequence():
    torch.manual_seed(8)
    rnn = RNNAggregator(DIM).eval()
    first, second = FixedPadding(3)(_items(1, 1)), FixedPadding(2)(_items(2, 2))
    joined = ConcatenateSequence()((first, second))
    real = torch.cat([first.values[0, :1], second.values[0]])
    assert torch.allclose(rnn(joined)[0], _alone(rnn, real), atol=1e-6)


def test_a_vector_connector_rejects_padded_branches():
    with pytest.raises(VectorMeshError, match="join masks"):
        Concatenate2D()((FixedPadding(2)(_items(1)), torch.randn(1, DIM)))


# --- the forms that hold components -----------------------------------------


def test_skip_passes_the_mask_to_its_transform():
    torch.manual_seed(9)
    skip = Skip(TransformerBlock(DIM, num_heads=2), in_size=DIM).eval()
    items = _items(2, 4)
    out = skip(FixedPadding(4)(items))
    assert torch.allclose(out.values[0, :2], _alone(skip, items[0]), atol=1e-5)


def test_moe_passes_the_mask_to_transformer_experts():
    torch.manual_seed(10)
    experts: list[nn.Module] = [TransformerBlock(DIM, num_heads=2) for _ in range(2)]
    moe = MoE(experts, hidden_size=DIM, out_size=DIM).eval()
    items = _items(2, 4)
    out = moe(FixedPadding(4)(items))
    assert torch.allclose(out.values[0, :2], _alone(moe, items[0]), atol=1e-5)


def test_sparse_moe_in_a_serial_routes_only_real_positions():
    torch.manual_seed(11)
    model = Serial([SparseMoE([NeuralNet(DIM, DIM) for _ in range(3)], DIM, DIM)])
    out = model(FixedPadding(4)(_items(2, 4)))
    assert (out.values[0, 2:] == 0).all()


# --- the Trainer moves a Padded batch ---------------------------------------


def test_trainer_moves_padded_batches_to_its_device(tmp_path):
    from vectormesh.training import Trainer, TrainerSettings

    items = _items(2, 3, 1, 4)
    y = torch.eye(2)[[0, 1, 0, 1]]
    batches = [(FixedPadding(4)(items), y)]
    settings = TrainerSettings(
        epochs=1,
        metrics=[],
        logdir=tmp_path,
        train_steps=1,
        valid_steps=1,
        earlystop_kwargs=None,
        scheduler_kwargs=None,
    )
    trainer = Trainer(
        model=Serial([Projection(DIM, 4), MaskedMeanAggregator(), NeuralNet(4, 2)]),
        settings=settings,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        traindataloader=batches,
        validdataloader=batches,
        device="cpu",
        progress=False,
    )
    result = trainer.loop()
    assert result.test_loss > 0
