# Changelog

## 0.12.0

### Fixed

- **Long texts keep all their chunks.** `tokenizers` 0.23.1 and 0.23.2 return only the
  first window and one short remainder for `return_overflowing_tokens`, so `Vectorizer`
  and `ChunkedRegexVectorizer` cut every long text to two chunks that cover little more
  than its first context window: at `max_length=512` a 9778-token text became a chunk of
  tokens 0-512 and one of tokens 461-514, and the rest was never embedded. vectormesh
  0.11.0 locked `tokenizers` 0.23.1. vectormesh now requires `tokenizers>=0.23.3` and
  `transformers>=5.16`.
  Caches built with vectormesh 0.11.0 and `tokenizers` 0.23.1/0.23.2 hold truncated
  chunks and should be rebuilt; `VectorCache.get_or_create` reuses a cache folder by tag,
  so delete those folders first. A cache's `chunk_sizes` metadata shows it: no count
  above 2 on a corpus with long texts.
- `RNNAggregator` skips padded chunks: the GRU's final state is the one after an item's
  last real chunk, also when padding sits between real chunks.

### Added

- `ConcatenateSequence` joins branches along the chunk axis,
  `(B, C1, D), (B, C2, D) -> (B, C1+C2, D)`, so a `TransformerBlock` attends across them.

### Changed

- Components belong to a family by the rank they work on (`components/contracts.py`):
  `SequenceComponent`, `RankPolymorphicComponent`, `BaseAggregator`, `Connector`.
  Vectorizers are `ItemVectorizer` or `ChunkedVectorizer`.
- `Gate` and `Highway` are rank-polymorphic; `VectorComponent` is removed.
- `Skip` lives in `components/pipelines.py`, next to `Serial` and `Parallel`.
- Connectors check every branch, not one sampled branch.
- The summed-width axis of `Concatenate2D` and `Concatenate3D` is named `total`.
- Importing vectormesh no longer adds a `logs/dataset.log` sink.
