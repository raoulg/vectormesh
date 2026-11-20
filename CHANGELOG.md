# Changelog

## 0.17.0

### Fixed

- **Early stopping reports the best epoch.** With `earlystop_kwargs` set, `Trainer.loop()`
  returns the `TrainResult` of the epoch with the lowest validation loss and leaves that epoch's
  weights in `trainer.model`, also with `save=False`: `EarlyStopping` copies the `state_dict` in
  memory at every improvement and `loop()` loads it back into the same model object when it ends,
  whether patience ran out or the epoch budget did. vectormesh 0.16.0 and earlier returned the
  metrics and, with `save=False`, the weights of the last epoch, `patience` epochs past the best.
  `save=True` still writes every improvement to `log_dir / "checkpoint.pt"`.
- `trainer.last_epoch` after an early stop is the epoch it stopped at, so a second `loop()`
  numbers its epochs on from there.
- `EarlyStopping(save=True, verbose=True)` logs its first checkpoint without a `TypeError`.

### Added

- `TrainResult.stopped_epoch`: the last epoch that ran, next to `epoch`, the epoch the result
  and the model belong to. Without early stopping the two are equal.
- `EarlyStopping.best_state` (the best epoch's `state_dict`) and `EarlyStopping.improved`
  (whether the last call improved the best loss).

## 0.16.0

### Added

- `Dense(in_size, out_size, norm="none", activation="gelu", dropout=0.0)`: one hidden
  layer (Linear, Norm, Activation, Dropout), typed `(*lead, in_size) -> (*lead, out_size)`.
- `Activation(kind)`: `"gelu"`, `"relu"`, `"tanh"`, `"sigmoid"` or `"none"`, chosen by name
  as `Norm(kind=...)` is; an unknown kind is a `VectorMeshError` that lists the kinds.
- `Design.build(config, **widths)`: the module a config chooses, from the config dataclass
  or from a config a trial drew.

### Changed

- `Design.show(example, **widths)` builds every combination of the options itself. The form
  `Design.show(build, example)` is removed: call `DESIGN.show(example, dim=..., ...)` where
  it was `DESIGN.show(lambda chosen: Module(DESIGN.drawn(chosen), ...), example)`.

## 0.15.0

### Added

- `vectormesh.design`: `Design`, `option`, `axis_of` and `StageShape`. A design is an
  `nn.Module` with one typed method per stage, with `Serial`/`Parallel` compositions inside
  a stage; `@option` marks a stage a search may leave out, and `Design.show` prints the
  shape after every stage for every combination of the options (docs §5.8).

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
