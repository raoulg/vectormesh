"""Tests for the cache fingerprint that VectorCache hands to `dataset.map`.

Without an explicit fingerprint, `datasets` derives one by pickle-hashing the
mapped function -- which closes over the vectorizer, i.e. the whole torch model.
That costs a flat ~21s per `VectorCache.create` call regardless of dataset size.

The fingerprint names the arrow cache file `map` reuses, so it has to be both
deterministic (else nothing is ever reused) and sensitive to everything that
changes the output (else a stale result is served silently). These tests pin
down both halves.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
from beartype import beartype
from datasets import Dataset, load_from_disk
from jaxtyping import Float, jaxtyped
from pydantic import model_validator

from vectormesh.data.cache import VectorCache
from vectormesh.data.vectorizers import (
    BaseVectorizer,
    RegexVectorizer,
    build_legal_reference_pattern,
    harmonize_legal_reference,
)

STUB_DIM = 4
TEXTS = ["a", "bb", "ccc", "dddd", "eeeee", "ffffff"]


class StubVectorizer(BaseVectorizer):
    """Vectorizer without a model: each text becomes `len(text) * scale`.

    `scale` stands in for the kind of plain field a subclass might add later:
    it changes the output, and nothing here tells the fingerprint about it.
    """

    model_name: str = "stub-model"
    col_name: str = "embed"
    input_col: str = "text"
    device: str = "cpu"
    scale: float = 1.0

    @model_validator(mode="after")
    def initialize_model(self):
        self._metadata = SimpleNamespace(hidden_size=STUB_DIM)
        self._effective_max_length = None
        return self

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, inputs: list, batchsize: int
    ) -> dict[str, list[Float[torch.Tensor, "dim"]]]:
        vectors = [torch.full((STUB_DIM,), len(t) * self.scale) for t in inputs]
        return {self.col_name: vectors}


class PickleProbe:
    """Stand-in for the torch model: counts attempts to serialise it.

    Both this and ProbedVectorizer live at module level on purpose -- dill gives
    up on classes defined inside a function, which would make the probe silently
    never fire.
    """

    def __init__(self):
        self.pickled = 0

    def __reduce__(self):
        self.pickled += 1
        return (PickleProbe, ())


class ProbedVectorizer(StubVectorizer):
    probe: object = None


def fingerprint(
    vectorizer: BaseVectorizer,
    dataset: Dataset,
    vector_batch: int = 2,
    map_batch: int = 2,
    remove_columns: Optional[list[str]] = None,
) -> str:
    """Fingerprint `dataset.map(vectorizer)` the way VectorCache.create does."""
    features = VectorCache.get_features(
        dataset,
        VectorCache.get_dtensor(vectorizer),
        embedding_column=vectorizer.col_name,
    )
    return VectorCache._map_fingerprint(
        dataset, vectorizer, features, vector_batch, map_batch, remove_columns
    )


def regex_vectorizer(training_texts: list[str]) -> RegexVectorizer:
    """A RegexVectorizer that differs from its siblings only in what it fit on."""
    return RegexVectorizer(
        pattern_builder=build_legal_reference_pattern,
        harmonizer=harmonize_legal_reference,
        min_doc_frequency=1,
        max_features=2,
        training_texts=training_texts,
    )


def saved_dataset(path: Path) -> Dataset:
    """Round-trip a dataset through disk.

    `map` only consults its on-disk arrow cache for a disk-backed dataset; an
    in-memory one never reuses anything, and would pass the cache tests below no
    matter what the fingerprint said.
    """
    Dataset.from_dict({"text": TEXTS}).save_to_disk(path)
    dataset = load_from_disk(path)
    assert isinstance(dataset, Dataset)
    return dataset


@pytest.fixture
def dataset() -> Dataset:
    return Dataset.from_dict({"text": TEXTS})


def test_fingerprint_is_stable_across_calls(dataset):
    """Same transformation, same fingerprint -- or a cache is never reused."""
    vectorizer = StubVectorizer()
    assert fingerprint(vectorizer, dataset) == fingerprint(vectorizer, dataset)


def test_fingerprint_is_a_valid_cache_filename(dataset):
    """`datasets` rejects fingerprints over 64 chars or with path characters."""
    from datasets.fingerprint import validate_fingerprint

    validate_fingerprint(fingerprint(StubVectorizer(), dataset))


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_name", "other-model"),
        ("col_name", "other_column"),
        ("input_col", "other_input"),
        ("device", "meta"),
        ("scale", 2.0),
    ],
)
def test_fingerprint_tracks_vectorizer_fields(dataset, field, value):
    """Every public field reaches the fingerprint, including ones added later.

    `scale` is the interesting case: no fingerprint code mentions it by name,
    yet two vectorizers differing only in `scale` produce different vectors and
    so must not share a cache entry.
    """
    baseline = fingerprint(StubVectorizer(), dataset)
    assert fingerprint(StubVectorizer(**{field: value}), dataset) != baseline


def test_fingerprint_tracks_the_source_data(dataset):
    """Different rows, different fingerprint."""
    other = Dataset.from_dict({"text": [t.upper() for t in TEXTS]})
    assert fingerprint(StubVectorizer(), other) != fingerprint(
        StubVectorizer(), dataset
    )


@pytest.mark.parametrize(
    "vector_batch,map_batch,remove_columns",
    [(8, 2, None), (2, 8, None), (2, 2, ["text"])],
)
def test_fingerprint_tracks_map_kwargs(
    dataset, vector_batch, map_batch, remove_columns
):
    """Batch sizes affect float accumulation; remove_columns affects the schema."""
    vectorizer = StubVectorizer()
    baseline = fingerprint(vectorizer, dataset)
    changed = fingerprint(vectorizer, dataset, vector_batch, map_batch, remove_columns)
    assert changed != baseline


def test_fingerprint_tracks_the_regex_fit(dataset):
    """Two fits of the same regex vectorizer must not collide.

    `model_name` is a constant for RegexVectorizer and both vectorizers below
    end up with the same hidden size, so everything the metadata records about
    them is identical -- only the fitted vocabulary differs, and that is exactly
    what decides which feature lands in which column.
    """
    fitted_on_civil = regex_vectorizer(
        ["artikel 7:2 Burgerlijk Wetboek", "artikel 265 Boek 3 Burgerlijk Wetboek"] * 5
    )
    fitted_on_soil = regex_vectorizer(
        ["artikel 55 Wet Bodembescherming", "artikel 12 Wet Bodembescherming"] * 5
    )

    assert fitted_on_civil.get_metadata == fitted_on_soil.get_metadata
    assert fingerprint(fitted_on_civil, dataset) != fingerprint(fitted_on_soil, dataset)


def test_fingerprint_ignores_vectorizer_usage_counters(dataset):
    """Running a vectorizer must not change what it fingerprints as.

    Vectorizer.chunk_sizes is a public field that fills up during vectorization;
    if it reached the fingerprint, the second create() of a session would never
    reuse the first one's cache.
    """

    class CountingVectorizer(StubVectorizer):
        calls: list = []

    vectorizer = CountingVectorizer()
    baseline = fingerprint(vectorizer, dataset)
    vectorizer.calls.append("used")
    assert "calls" not in CountingVectorizer.FINGERPRINT_EXCLUDE
    assert fingerprint(vectorizer, dataset) != baseline, (
        "sanity check: an unexcluded field does reach the fingerprint"
    )

    class ExcludingVectorizer(CountingVectorizer):
        FINGERPRINT_EXCLUDE = frozenset({"calls"})

    excluding = ExcludingVectorizer()
    baseline = fingerprint(excluding, dataset)
    excluding.calls.append("used again")
    assert fingerprint(excluding, dataset) == baseline


def test_unserialisable_field_does_not_leak_an_address(dataset):
    """A field the base class cannot describe must still fingerprint stably.

    Falling back to repr() would embed a memory address, which changes every
    run and would silently disable all cache reuse.
    """

    class CallableFieldVectorizer(StubVectorizer):
        hook: object = build_legal_reference_pattern

    first = CallableFieldVectorizer()
    second = CallableFieldVectorizer()
    assert fingerprint(first, dataset) == fingerprint(second, dataset)
    assert "<unhashable:function>" in str(first.fingerprint_fields())


def test_vectorize_passes_the_fingerprint_to_map(dataset):
    """The mapped dataset carries our fingerprint, not one datasets computed."""
    vectorizer = StubVectorizer()
    features = VectorCache.get_features(dataset, 1, embedding_column="embed")

    result = VectorCache._vectorize(dataset, vectorizer, features, 2, 2, None)

    assert result._fingerprint == fingerprint(vectorizer, dataset)


def test_vectorize_does_not_hash_the_vectorizer(dataset):
    """The point of the whole exercise: no pickling of the model per call.

    `datasets` fingerprints a map function by pickling it, and that function
    closes over the vectorizer -- for a real vectorizer, that means serialising
    the entire torch model. PickleProbe stands in for the model and records any
    attempt, so this fails if the explicit fingerprint stops being passed.
    """
    probe = PickleProbe()
    vectorizer = ProbedVectorizer(probe=probe)
    features = VectorCache.get_features(dataset, 1, embedding_column="embed")

    VectorCache._vectorize(dataset, vectorizer, features, 2, 2, None)

    assert probe.pickled == 0


def test_different_vectorizers_do_not_share_a_cache_file(tmp_path):
    """End to end: a second vectorizer must not be served the first one's vectors."""
    dataset = saved_dataset(tmp_path / "source")

    def vectors(vectorizer: StubVectorizer) -> torch.Tensor:
        cache = VectorCache.create(
            cache_dir=tmp_path / "cache",
            vectorizer=vectorizer,
            dataset=dataset,
            dataset_tag=vectorizer.model_name,
            vector_batch=2,
            map_batch=2,
        )
        return torch.stack(list(cache["embed"]))

    single = vectors(StubVectorizer(model_name="single", scale=1.0))
    double = vectors(StubVectorizer(model_name="double", scale=2.0))

    assert torch.allclose(double, single * 2)


def test_same_vectorizer_reuses_the_cache_file(tmp_path):
    """The flip side: an identical run does hit the cache instead of recomputing."""
    dataset = saved_dataset(tmp_path / "source")
    features = VectorCache.get_features(dataset, 1, embedding_column="embed")

    calls = []

    class RecordingVectorizer(StubVectorizer):
        def __call__(self, inputs, batchsize):
            calls.append(len(inputs))
            return super().__call__(inputs, batchsize)

    vectorizer = RecordingVectorizer()
    first = VectorCache._vectorize(dataset, vectorizer, features, 2, 2, None)
    calls.clear()
    second = VectorCache._vectorize(dataset, vectorizer, features, 2, 2, None)

    assert calls == [], "second run recomputed instead of reusing the cache file"
    assert first["embed"] == second["embed"]
