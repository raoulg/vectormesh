"""Tests for `VectorCache.push_to_hub` and the dataset card it generates.

The thing worth pinning down is the payload. A published cache is downloaded by everyone
who uses it, so an upload that silently carries a second copy of the vectors -- `datasets`
leaves `cache-<hash>.arrow` files next to any dataset it has mapped over -- costs every
consumer that download. `push_to_hub` re-serialises what it is going to send instead of
copying the cache folder, and these tests hold it to that.

Everything except the `integration` cases runs offline: the hub is a stub that records
the calls it was handed.
"""

import io
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import (
    ClassLabel,
    Dataset,
    Features,
    Image,
    Sequence,
    Value,
    load_from_disk,
)
from loguru import logger
from PIL import Image as PILImage

from vectormesh.data import hub
from vectormesh.data.cache import VectorCache
from vectormesh.types import VectorMeshError

DIM = 4
ROWS = 6
SHA = "a" * 40
EXPECTED_PAYLOAD = {
    "data-00000-of-00001.arrow",
    "dataset_info.json",
    "state.json",
    "metadata.json",
}


def _metadata(columns: list[str], column: str = "embed") -> dict:
    return {
        column: {
            "vectormesh_version": "0.9.2",
            "model_tag": "facebook/dinov2-small",
            "vectorizer_type": "ImageVectorizer",
            "tensordtype": 1,
            "hidden_size": DIM,
            "context_size": None,
            "chunk_sizes": None,
        },
        "features": columns,
        "created_at": "2026-08-09T12:00:00",
        "num_observations": ROWS,
    }


@pytest.fixture
def cache(tmp_path: Path) -> VectorCache:
    """A minimal published-shaped cache: vectors, a label, and a source pointer."""
    rows = {
        "label": [i % 2 for i in range(ROWS)],
        "source_idx": list(range(100, 100 + ROWS)),
        "embed": [[float(i)] * DIM for i in range(ROWS)],
    }
    dataset = Dataset.from_dict(rows)
    dataset.set_format(type="torch")
    return VectorCache(
        name="20260809120000_stub_embed",
        cache_dir=tmp_path,
        dataset=dataset,
        metadata=_metadata(list(rows)),
    )


class StubApi:
    """Stands in for `HfApi`, recording what a push handed it.

    `upload_folder` reads the folder back, so a test can assert on what would actually
    have travelled rather than on the arguments alone.
    """

    calls: list[tuple[str, dict]] = []

    def __init__(self, token=None):
        self.token = token

    def create_repo(self, repo_id, **kwargs):
        StubApi.calls.append(("create_repo", dict(kwargs, repo_id=repo_id)))

    def upload_folder(self, **kwargs):
        folder = Path(kwargs["folder_path"])
        with open(folder / "metadata.json") as f:
            kwargs["metadata"] = json.load(f)
        kwargs["files"] = sorted(p.name for p in folder.rglob("*") if p.is_file())
        kwargs["columns"] = load_from_disk(folder).column_names
        StubApi.calls.append(("upload_folder", kwargs))

    def upload_file(self, **kwargs):
        StubApi.calls.append(("upload_file", kwargs))

    def list_repo_files(self, *args, **kwargs):
        raise RuntimeError("no such repo")

    def repo_info(self, repo_id, **kwargs):
        return SimpleNamespace(sha=SHA)


@pytest.fixture
def stub_hub(monkeypatch) -> type[StubApi]:
    """Replace the hub with a recorder, for both the upload and the card."""
    import huggingface_hub

    StubApi.calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", StubApi)
    return StubApi


def call(stub: type[StubApi], name: str) -> dict:
    return next(kwargs for made, kwargs in stub.calls if made == name)


@pytest.fixture
def logs() -> Iterator[list[str]]:
    """Capture loguru output: warnings are part of this method's contract."""
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m), level="WARNING")
    yield messages
    logger.remove(sink)


# --- the payload ------------------------------------------------------------------


def test_dry_run_stages_the_expected_files(cache: VectorCache, stub_hub):
    upload = cache.push_to_hub("acme/stub-dinov2-small", dry_run=True)

    assert set(upload.files) == EXPECTED_PAYLOAD
    assert upload.uploaded is False
    assert upload.nbytes > 0
    assert upload.url == "https://huggingface.co/datasets/acme/stub-dinov2-small"
    assert stub_hub.calls == [], "a dry run must not touch the hub"


def test_upload_carries_only_the_staged_payload(cache: VectorCache, stub_hub):
    """The upload is a folder we wrote ourselves, not the cache directory."""
    cache.push_to_hub("acme/stub-dinov2-small", split="train")

    folder = call(stub_hub, "upload_folder")
    assert set(folder["files"]) == EXPECTED_PAYLOAD
    assert folder["columns"] == ["label", "source_idx", "embed"]
    assert folder["path_in_repo"] == "train"
    assert folder["repo_type"] == "dataset"
    assert folder["allow_patterns"] == ["*.arrow", "*.json"]


def test_stale_map_cache_files_are_not_uploaded(cache: VectorCache, tmp_path, stub_hub):
    """A `cache-<hash>.arrow` next to the cache is the ~50 MB nobody meant to publish."""
    folder = tmp_path / cache.name
    cache.dataset.with_format(None).save_to_disk(folder)
    with open(folder / "metadata.json", "w") as f:
        json.dump(cache.metadata, f)
    (folder / "cache-deadbeef.arrow").write_bytes(b"x" * 10_000)

    upload = VectorCache.load(folder).push_to_hub("acme/stub", dry_run=True)

    assert set(upload.files) == EXPECTED_PAYLOAD
    assert not any(f.startswith("cache-") for f in upload.files)


def test_stage_payload_refuses_unexpected_files(cache: VectorCache, tmp_path):
    """The tripwire: anything that is not arrow or json never reaches the hub.

    `save_to_disk` writes into a directory without emptying it first, so whatever was
    already there is part of the payload.
    """
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "notes.txt").write_text("hello")

    with pytest.raises(VectorMeshError, match="arrow and json only"):
        hub.stage_payload(cache.dataset.with_format(None), cache.metadata, staged)


def test_payload_round_trips_through_load(cache: VectorCache, tmp_path):
    """What is staged is what `from_hub` will `load()` on the other side."""
    staged = tmp_path / "train"
    hub.stage_payload(cache.dataset.with_format(None), cache.metadata, staged)

    reloaded = VectorCache.load(staged)
    assert len(reloaded) == ROWS
    assert reloaded.vector_columns == ["embed"]
    assert reloaded.metadata["embed"]["model_tag"] == "facebook/dinov2-small"
    assert reloaded[0]["source_idx"].item() == 100


# --- columns ----------------------------------------------------------------------


@pytest.fixture
def image_cache(tmp_path: Path) -> VectorCache:
    """A cache that still carries its pixels -- the size trap `drop_columns` exists for."""
    buffer = io.BytesIO()
    PILImage.new("RGB", (2, 2)).save(buffer, format="PNG")
    png = {"bytes": buffer.getvalue(), "path": None}

    dataset = Dataset.from_dict(
        {
            "image": [png] * ROWS,
            "label": [0] * ROWS,
            "embed": [[0.0] * DIM] * ROWS,
        },
        features=Features(
            {
                "image": Image(),
                "label": ClassLabel(num_classes=2),
                "embed": Sequence(Value("float32")),
            }
        ),
    )
    return VectorCache(
        name="stub",
        cache_dir=tmp_path,
        dataset=dataset,
        metadata=_metadata(["image", "label", "embed"]),
    )


def test_drop_columns_leaves_them_out_of_payload_and_metadata(image_cache, stub_hub):
    image_cache.push_to_hub("acme/stub", drop_columns=["image"])

    folder = call(stub_hub, "upload_folder")
    assert folder["columns"] == ["label", "embed"]
    assert folder["metadata"]["features"] == ["label", "embed"]
    assert set(folder["files"]) == EXPECTED_PAYLOAD


def test_raw_media_columns_are_warned_about(image_cache, stub_hub, logs):
    image_cache.push_to_hub("acme/stub", dry_run=True)
    assert any("['image'] carry raw media" in message for message in logs)


def test_a_missing_source_idx_is_warned_about(image_cache, stub_hub, logs):
    image_cache.push_to_hub("acme/stub", drop_columns=["image"], dry_run=True)
    assert any("no 'source_idx' column" in message for message in logs)
    assert not any("raw media" in message for message in logs)


def test_drop_columns_rejects_a_name_that_is_not_a_column(cache: VectorCache):
    with pytest.raises(VectorMeshError, match="cannot drop 'imagee'"):
        cache.push_to_hub("acme/stub", drop_columns=["imagee"], dry_run=True)


def test_dropping_every_vector_column_raises(cache: VectorCache):
    with pytest.raises(VectorMeshError, match="nothing left to publish"):
        cache.push_to_hub("acme/stub", drop_columns=["embed"], dry_run=True)


def test_empty_cache_is_refused(tmp_path):
    empty = VectorCache(
        name="stub",
        cache_dir=tmp_path,
        dataset=Dataset.from_dict({"embed": []}),
        metadata=_metadata(["embed"]),
    )
    with pytest.raises(VectorMeshError, match="empty cache"):
        empty.push_to_hub("acme/stub", dry_run=True)


# --- the repo id ------------------------------------------------------------------


def test_hub_repo_id_derives_from_the_model_tag(cache: VectorCache):
    assert cache.hub_repo_id("pttrn-io", "eurosat") == "pttrn-io/eurosat-dinov2-small"


def test_push_derives_the_repo_id_from_org_and_dataset_name(
    cache: VectorCache, stub_hub
):
    """The ordinary call names the org and the dataset; the encoder comes from metadata."""
    upload = cache.push_to_hub(org="pttrn-io", dataset_name="eurosat", dry_run=True)

    assert upload.repo_id == "pttrn-io/eurosat-dinov2-small"


def test_push_needs_a_repo_id_from_somewhere(cache: VectorCache):
    with pytest.raises(VectorMeshError, match="somewhere to publish to"):
        cache.push_to_hub(dry_run=True)


def test_push_refuses_a_repo_id_and_a_derivation_at_once(cache: VectorCache):
    with pytest.raises(VectorMeshError, match="not both"):
        cache.push_to_hub("acme/stub", org="pttrn-io", dry_run=True)


def test_hub_repo_id_refuses_an_ambiguous_cache(cache: VectorCache, tmp_path):
    joined = VectorCache(
        name="joined",
        cache_dir=tmp_path,
        dataset=cache.dataset,
        metadata=dict(
            cache.metadata, embed_resnet18={"model_tag": "microsoft/resnet-18"}
        ),
    )
    with pytest.raises(VectorMeshError, match="no single encoder"):
        joined.hub_repo_id("pttrn-io", "eurosat")


# --- the card ---------------------------------------------------------------------


def test_card_reports_what_metadata_says(cache: VectorCache, stub_hub):
    card = cache.push_to_hub(
        "acme/eurosat-dinov2-small",
        source_dataset="blanchon/EuroSAT_RGB",
        dry_run=True,
    ).card

    assert card.startswith("---\nlicense: other")
    assert "# eurosat-dinov2-small" in card
    assert "`facebook/dinov2-small`" in card
    assert f"| `embed` | `facebook/dinov2-small`, revision `{SHA}` | {DIM} |" in card
    assert f"| rows | {ROWS:,} train |" in card
    assert "| columns | `label`, `source_idx`, `embed` |" in card
    assert "| built with | vectormesh 0.9.2, 2026-08-09 |" in card
    assert 'VectorCache.from_hub("acme/eurosat-dinov2-small", split="train")' in card
    assert "`datasets.load_dataset()` will **not** read" in card
    assert "`blanchon/EuroSAT_RGB`" in card
    assert f'revision="{SHA}"' in card, "the source pointer must be pinned"


def test_card_skips_the_source_section_without_a_source_dataset(cache, stub_hub):
    card = cache.push_to_hub("acme/stub", dry_run=True).card
    assert "## Getting the original back" not in card
    assert "its source dataset" in card


def test_card_survives_an_unresolvable_revision(cache: VectorCache, monkeypatch):
    monkeypatch.setattr(hub, "hub_revision", lambda *args, **kwargs: None)

    card = cache.push_to_hub("acme/stub", source_dataset="acme/gone", dry_run=True).card

    assert "revision `" not in card
    assert "revision=SHA" in card, "an unpinned snippet must still say what to pass"
    assert "`acme/gone`" in card


def test_card_covers_every_split_in_the_repo(cache: VectorCache, monkeypatch, tmp_path):
    """Pushing `test` must not rewrite the card as if `train` were gone."""
    monkeypatch.setattr(hub, "hub_revision", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        VectorCache,
        "_hub_splits",
        staticmethod(lambda *args, **kwargs: ["train", "test"]),
    )
    published = tmp_path / "train_metadata.json"
    published.write_text(
        json.dumps(
            dict(_metadata(["label", "source_idx", "embed"]), num_observations=99)
        )
    )
    asked = []

    def fake_download(repo_id, filename, **kwargs):
        asked.append(filename)
        return str(published)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    card = cache.push_to_hub("acme/stub", split="test", dry_run=True).card

    assert asked == ["train/metadata.json"]
    assert f"| rows | 99 train / {ROWS:,} test |" in card, "train first, then test"
    assert 'VectorCache.from_hub("acme/stub", split="test")' in card


def test_card_can_be_supplied_verbatim(cache: VectorCache, stub_hub):
    upload = cache.push_to_hub("acme/stub", card="# mine", dry_run=True)
    assert upload.card == "# mine"


def test_write_card_false_leaves_the_readme_alone(cache: VectorCache, stub_hub):
    cache.push_to_hub("acme/stub", write_card=False)
    assert [made for made, _ in stub_hub.calls] == ["create_repo", "upload_folder"]


def test_card_is_uploaded_as_readme(cache: VectorCache, stub_hub):
    cache.push_to_hub("acme/stub", private=True)

    assert call(stub_hub, "create_repo")["private"] is True
    card = call(stub_hub, "upload_file")
    assert card["path_in_repo"] == "README.md"
    assert card["repo_type"] == "dataset"
    assert card["path_or_fileobj"].decode().startswith("---\nlicense: other")


def test_upload_failure_explains_the_token(cache: VectorCache, monkeypatch, stub_hub):
    def refuse(*args, **kwargs):
        raise OSError("401 Unauthorized")

    monkeypatch.setattr(StubApi, "create_repo", refuse)
    with pytest.raises(VectorMeshError, match="write token"):
        cache.push_to_hub("acme/stub")


# --- chunk sizes ------------------------------------------------------------------


@pytest.mark.parametrize(
    "histogram, expected",
    [
        (None, "n/a -- one vector per row"),
        ({}, "n/a -- one vector per row"),
        ({"1": 100}, "median 1, p95 1, max 1"),
        ({"1": 50, "2": 45, "27": 5}, "median 1, p95 2, max 27"),
        ({1: 50, 2: 45, 27: 5}, "median 1, p95 2, max 27"),
    ],
)
def test_chunk_summary(histogram, expected):
    """Keys arrive as strings after a json round trip and as ints before one."""
    assert hub.chunk_summary(histogram) == expected


def test_card_summarises_chunked_caches(cache: VectorCache, tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "hub_revision", lambda *args, **kwargs: None)
    metadata = _metadata(["label", "source_idx", "embed"])
    metadata["embed"]["tensordtype"] = 2
    metadata["embed"]["chunk_sizes"] = {"1": 50, "3": 45, "27": 5}
    chunked = VectorCache(
        name="stub", cache_dir=tmp_path, dataset=cache.dataset, metadata=metadata
    )

    card = chunked.push_to_hub("acme/stub", dry_run=True).card
    assert "`(chunks, dim)`" in card
    assert "median 1, p95 3, max 27" in card


# --- integration ------------------------------------------------------------------


@pytest.mark.integration
def test_dry_run_against_a_real_published_cache():
    """Round trip: pull a real cache off the hub and render the card it would get.

    Read-only -- `dry_run` uploads nothing, so this needs no write token.
    """
    cache = VectorCache.from_hub("pttrn-io/eurosat-dinov2-small", split="test")
    upload = cache.push_to_hub(
        org="pttrn-io",
        dataset_name="eurosat",
        split="test",
        source_dataset="blanchon/EuroSAT_RGB",
        dry_run=True,
    )

    assert upload.repo_id == "pttrn-io/eurosat-dinov2-small"
    assert upload.uploaded is False
    assert all(f.endswith((".arrow", ".json")) for f in upload.files)
    assert "metadata.json" in upload.files
    assert "`facebook/dinov2-small`" in upload.card
    assert f"{len(cache):,} test" in upload.card
