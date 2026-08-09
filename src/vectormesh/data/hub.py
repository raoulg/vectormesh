"""Publishing a `VectorCache` to the HuggingFace hub.

`VectorCache.push_to_hub` lives in `cache.py`; this module holds the parts of it that
are about the hub rather than about the cache: staging the payload, resolving
revisions, and rendering the dataset card.

Nothing here imports `VectorCache` -- everything takes a `Dataset` and a metadata dict
-- so the card can also be rendered for a split that only exists on the hub.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from datasets import Dataset
from loguru import logger
from pydantic import BaseModel, ConfigDict

from vectormesh.types import VectorMeshError

# Keys in metadata.json that describe the cache as a whole rather than one column.
META_NON_COLUMN_KEYS = frozenset({"features", "created_at", "num_observations"})

# What `Dataset.save_to_disk` plus our own metadata.json may produce. A payload that
# contains anything else is not something we are willing to upload sight unseen.
PAYLOAD_SUFFIXES = frozenset({".arrow", ".json"})

REPO_URL = "https://github.com/raoulg/vectormesh"


class HubUpload(BaseModel):
    """What `push_to_hub` did, or -- with `dry_run=True` -- what it would have done."""

    model_config = ConfigDict(frozen=True)

    repo_id: str
    split: str
    files: list[str]
    nbytes: int
    card: str
    url: str
    uploaded: bool

    @property
    def megabytes(self) -> float:
        return self.nbytes / 1e6


def vector_columns(metadata: dict) -> list[str]:
    """Columns described by metadata.json, i.e. the ones a vectorizer produced."""
    return [k for k in metadata if k not in META_NON_COLUMN_KEYS]


def stage_payload(dataset: Dataset, metadata: dict, dest: Path) -> list[Path]:
    """Serialise exactly what will be uploaded into an empty `dest`.

    Publishing re-serialises rather than uploading the cache folder as it sits on
    disk. `datasets` writes its own `cache-<hash>.arrow` files next to a dataset it
    has mapped over, so "upload the folder" can mean uploading several copies of the
    vectors without anyone noticing until the download is a gigabyte. Staging means
    the payload is a set of files we wrote ourselves and can count.

    Raises:
        VectorMeshError: if the staged directory holds anything but arrow and json.
    """
    dataset.save_to_disk(dest)
    with open(dest / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    files = sorted(p for p in dest.rglob("*") if p.is_file())
    unexpected = [p.name for p in files if p.suffix not in PAYLOAD_SUFFIXES]
    if unexpected:
        raise VectorMeshError(
            f"refusing to upload {unexpected}: a cache payload is arrow and json only."
        )
    return files


def hub_revision(
    repo_id: str, repo_type: str, token: Optional[str] = None
) -> Optional[str]:
    """The current commit sha of a hub repo, or None if it cannot be resolved.

    Best effort: a private, renamed or offline source is a gap in the card, not a
    reason to abandon an upload that has already happened.
    """
    try:
        from huggingface_hub import HfApi

        return HfApi().repo_info(repo_id, repo_type=repo_type, token=token).sha
    except Exception as e:  # pragma: no cover - network path
        logger.warning(f"could not resolve the {repo_type} revision of {repo_id}: {e}")
        return None


def chunk_summary(chunk_sizes: Optional[dict]) -> str:
    """median / p95 / max over a `{chunk_count: n_rows}` histogram."""
    if not chunk_sizes:
        return "n/a -- one vector per row"
    hist = {int(k): int(v) for k, v in chunk_sizes.items()}
    total = sum(hist.values())
    running, median, p95 = 0, None, None
    for k in sorted(hist):
        running += hist[k]
        if median is None and running >= 0.5 * total:
            median = k
        if p95 is None and running >= 0.95 * total:
            p95 = k
    return f"median {median}, p95 {p95}, max {max(hist)}"


def _shape(tensordtype: Optional[int]) -> str:
    return {1: "`(dim,)`", 2: "`(chunks, dim)`"}.get(tensordtype or 0, "unknown")


def _column_table(metadata: dict, token: Optional[str]) -> str:
    """One row per vector column: what produced it, and what it looks like."""
    header = (
        "| column | encoder | hidden_size | shape per row | chunks |\n"
        "|---|---|---|---|---|\n"
    )
    rows = []
    for column in vector_columns(metadata):
        spec = metadata[column]
        tag = spec.get("model_tag", "unknown")
        revision = hub_revision(tag, "model", token=token)
        encoder = f"`{tag}`" + (f", revision `{revision}`" if revision else "")
        rows.append(
            f"| `{column}` | {encoder} | {spec.get('hidden_size')} | "
            f"{_shape(spec.get('tensordtype'))} | "
            f"{chunk_summary(spec.get('chunk_sizes'))} |"
        )
    return header + "\n".join(rows)


def _built_with(metadata: dict) -> str:
    columns = vector_columns(metadata)
    version = (
        metadata.get(columns[0], {}).get("vectormesh_version") if columns else None
    )
    created = str(metadata.get("created_at", ""))[:10]
    return f"vectormesh {version or '?'}, {created or 'unknown date'}"


def _source_section(
    repo_id: str,
    source_dataset: str,
    revision: Optional[str],
    columns: list[str],
) -> str:
    """How to get from a cached row back to the row it was built from."""
    pin = f'revision="{revision}"' if revision else "revision=SHA"
    if "source_split" in columns:
        # A cache pooled from several source splits needs both keys to land on a row.
        key = (
            "Every row carries `source_idx` **and** `source_split`, because this cache "
            "pools more than one split of the source dataset -- both are needed to land "
            "on the right row."
        )
        lookup = f"""source = load_dataset("{source_dataset}", {pin})
row = cache[5]
source[row["source_split"]][row["source_idx"]]      # the original behind cache row 5"""
    else:
        key = (
            "Every row carries `source_idx`: its position in the **source** split, so a "
            "vector can be traced back to the image or text it came from without this "
            "repo having to carry it."
        )
        lookup = f"""source = load_dataset("{source_dataset}", {pin}, split="train")
source[cache[5]["source_idx"]]      # the original row behind cache row 5"""

    return f"""
## Getting the original back

{key}

```python
from datasets import load_dataset
from vectormesh import VectorCache

cache = VectorCache.from_hub("{repo_id}")
{lookup}
```

Pass that revision. A cache is usually a shuffled subsample, so `source_idx` is not the
row number, and an index into a different revision of `{source_dataset}` resolves to the
wrong row silently rather than failing.
"""


def render_card(
    repo_id: str,
    splits: dict[str, dict],
    source_dataset: Optional[str] = None,
    license: str = "other",
    token: Optional[str] = None,
) -> str:
    """Render the dataset card from the caches' own metadata.

    Args:
        repo_id: where the card will live, e.g. "pttrn-io/eurosat-dinov2-small".
        splits: split name -> that split's metadata.json. Ordered as rendered.
        source_dataset: hub id of the dataset the vectors were built from. Pinned to a
            revision in the card when it resolves.
        license: the `license` field of the card's frontmatter.
        token: for resolving revisions of private repos.
    """
    if not splits:
        raise VectorMeshError("cannot render a dataset card without any split metadata")

    first = next(iter(splits.values()))
    columns = list(first.get("features", []))
    encoders = ", ".join(
        "`" + str(first[c].get("model_tag", "?")).split("/")[-1] + "`"
        for c in vector_columns(first)
    )
    rows = " / ".join(
        f"{meta.get('num_observations', 0):,} {split}" for split, meta in splits.items()
    )
    source = f"`{source_dataset}`" if source_dataset else "its source dataset"
    source_revision = (
        hub_revision(source_dataset, "dataset", token=token) if source_dataset else None
    )
    source_row = (
        f"| source dataset | `{source_dataset}`"
        + (f", revision `{source_revision}`" if source_revision else "")
        + " |\n"
        if source_dataset
        else ""
    )
    loads = "\n".join(
        f'{split} = VectorCache.from_hub("{repo_id}", split="{split}")'
        for split in splits
    )
    original = (
        _source_section(repo_id, source_dataset, source_revision, columns)
        if source_dataset and "source_idx" in columns
        else ""
    )

    return f"""---
license: {license}
tags:
- vectormesh
- vector-cache
- embeddings
---

# {repo_id.split("/")[-1]}

Frozen embeddings of {source}, encoded by {encoders}: vectors and labels only, so a head
trains on a CPU laptop in seconds instead of an encoder running over the raw data first.

{_column_table(first, token)}

| | |
|---|---|
{source_row}| rows | {rows} |
| columns | {", ".join(f"`{c}`" for c in columns)} |
| built with | {_built_with(first)} |

## Load it

```python
from vectormesh import VectorCache

{loads}
{next(iter(splits))}.metadata      # read this before writing any model code
```

This is a `save_to_disk` layout, not parquet: `datasets.load_dataset()` will **not** read
it and [`VectorCache.from_hub`]({REPO_URL}) will. Only the split you ask for is downloaded.
{original}
## Licence

Embeddings are a derived work: they inherit the terms of {source}. Cite it as its authors
ask, and check that its licence permits redistribution before reusing this cache.

_Card generated by `VectorCache.push_to_hub`, {datetime.now().strftime("%Y-%m-%d")}._
"""
