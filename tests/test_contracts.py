"""Tests for the families in `components/contracts.py` and `data/vectorizers.py`.

A family is a shape contract held by a class, so the tests check two things: every
component and vectorizer sits in the family its shape belongs to, and a `Connector`
checks every branch rather than the one element beartype samples from a tuple.
"""

import pytest
import torch
from beartype import beartype
from jaxtyping import Float, TypeCheckError, jaxtyped
from torch import Tensor, nn

from vectormesh import (
    BaseVectorizer,
    ChunkedRegexVectorizer,
    ChunkedVectorizer,
    ImageVectorizer,
    ItemVectorizer,
    PatchImageVectorizer,
    RegexVectorizer,
    VectorCache,
    Vectorizer,
)
from vectormesh.components import (
    Attention,
    AttentionAggregator,
    BaseAggregator,
    Concatenate2D,
    Concatenate3D,
    ConcatenateSequence,
    Connector,
    Gate,
    GaussianNoise,
    Highway,
    MaskedMeanAggregator,
    MeanAggregator,
    MoE,
    NeuralNet,
    Parallel,
    Projection,
    RankPolymorphicComponent,
    RNNAggregator,
    SequenceComponent,
    Serial,
    Skip,
    SparseMoE,
    Stack2D,
    TransformerBlock,
)

COMPONENT_FAMILIES = {
    SequenceComponent: [Attention, TransformerBlock],
    RankPolymorphicComponent: [
        NeuralNet,
        Projection,
        Gate,
        Highway,
        MoE,
        SparseMoE,
        GaussianNoise,
    ],
    BaseAggregator: [
        MeanAggregator,
        MaskedMeanAggregator,
        AttentionAggregator,
        RNNAggregator,
    ],
    Connector: [Concatenate2D, Concatenate3D, ConcatenateSequence, Stack2D],
}

VECTORIZER_FAMILIES: dict[type[BaseVectorizer], list[type[BaseVectorizer]]] = {
    ItemVectorizer: [ImageVectorizer, RegexVectorizer],
    ChunkedVectorizer: [Vectorizer, PatchImageVectorizer, ChunkedRegexVectorizer],
}

FAMILIES = {**COMPONENT_FAMILIES, **VECTORIZER_FAMILIES}

MEMBERSHIP = [
    (family, member) for family, members in FAMILIES.items() for member in members
]


@pytest.mark.parametrize(
    ("family", "member"), MEMBERSHIP, ids=[m.__name__ for _, m in MEMBERSHIP]
)
def test_member_is_in_its_family_only(family, member):
    assert issubclass(member, family)
    for other in FAMILIES:
        if other is not family:
            assert not issubclass(member, other)


@pytest.mark.parametrize(
    ("family", "rank"), [(ItemVectorizer, 1), (ChunkedVectorizer, 2)]
)
def test_vectorizer_annotation_has_its_family_rank(family, rank):
    """`VectorCache` reads the cache rank off `__call__`'s annotation, so a member's
    annotation must keep the rank its family declares."""
    for member in VECTORIZER_FAMILIES[family]:
        assert VectorCache.get_dtensor(member) == rank, member.__name__


def test_connector_checks_every_branch():
    """The odd branch is last of four: sampling one element of a tuple would miss it
    three times in four, so twenty calls would almost surely let one through."""
    concat = Concatenate3D()
    branches = (*[torch.rand(2, 5, 3)] * 3, torch.rand(2, 6, 3))
    for _ in range(20):
        with pytest.raises(TypeCheckError):
            concat(branches)


def test_connector_checks_every_branch_dtype():
    concat = Concatenate2D()
    branches = (*[torch.rand(2, 3)] * 3, torch.zeros(2, 3, dtype=torch.int64))
    for _ in range(20):
        with pytest.raises(TypeCheckError):
            concat(branches)


def test_named_axis_binds_across_branches():
    """`Stack2D` names its width: `torch.stack` needs equal shapes, and the contract
    says so before torch does."""
    with pytest.raises(TypeCheckError):
        Stack2D()((torch.rand(2, 3), torch.rand(2, 4)))


def test_concatenate_sequence_joins_the_chunk_axis():
    out = ConcatenateSequence()((torch.rand(2, 3, 4), torch.rand(2, 5, 4)))
    assert out.shape == (2, 8, 4)


def test_concatenate_sequence_needs_one_width():
    with pytest.raises(TypeCheckError):
        ConcatenateSequence()((torch.rand(2, 3, 4), torch.rand(2, 5, 6)))


def test_connector_after_parallel():
    pipeline = Serial([Parallel([Projection(4, 3), Projection(5, 2)]), Concatenate2D()])
    assert pipeline((torch.rand(2, 4), torch.rand(2, 5))).shape == (2, 5)


@pytest.mark.parametrize("form", [Serial, Parallel, Skip])
def test_architecture_form_is_in_no_family(form):
    for family in COMPONENT_FAMILIES:
        assert not issubclass(form, family)


@pytest.mark.parametrize(
    "component",
    [Gate(4), Highway(NeuralNet(4, 4), 4), Skip(NeuralNet(4, 4), 4)],
    ids=["Gate", "Highway", "Skip"],
)
@pytest.mark.parametrize("shape", [(2, 4), (2, 3, 4)], ids=["2D", "3D"])
def test_takes_any_rank_and_keeps_the_leading_axes(component, shape):
    assert component(torch.rand(*shape)).shape == shape


def test_rank_polymorphic_component_rejects_the_wrong_width():
    with pytest.raises(TypeCheckError):
        Gate(4)(torch.rand(2, 3, 5))


def test_lead_binds_input_to_output():
    """A rank-polymorphic member that drops a leading axis breaks its own contract."""

    class Drops(RankPolymorphicComponent):
        @jaxtyped(typechecker=beartype)
        def forward(
            self, tensors: Float[Tensor, "*lead dim"]
        ) -> Float[Tensor, "*lead dim"]:
            return tensors[0]

    with pytest.raises(TypeCheckError):
        Drops()(torch.rand(2, 3, 4))


def test_rank_polymorphic_component_takes_both_rungs():
    net = NeuralNet(4, 2)
    assert net(torch.rand(2, 4)).shape == (2, 2)
    assert net(torch.rand(2, 3, 4)).shape == (2, 3, 2)


def test_skip_checks_its_input_width():
    with pytest.raises(TypeCheckError):
        Skip(nn.Identity(), in_size=4)(torch.rand(2, 5))


def test_gaussian_noise_preserves_shape():
    noise = GaussianNoise(sigma=0.1).train()
    assert noise(torch.rand(2, 3, 4)).shape == (2, 3, 4)
