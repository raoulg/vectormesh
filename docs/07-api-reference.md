# 7. API reference

Signatures as implemented in `src/vectormesh/` (version 0.12.0). `B` = batch, `C` = chunks,
`D` = dim.

---

## 7.1 Top level — `from vectormesh import ...`

```
VectorCache, Vectorizer, ImageVectorizer, PatchImageVectorizer, RegexVectorizer,
ChunkedRegexVectorizer, BaseVectorizer, ItemVectorizer, ChunkedVectorizer, HubUpload,
LabelEncoder, build, __version__,
Trainer, TrainerSettings, EarlyStopping, Reporter, Step, TrainResult
```

## 7.2 Data — `from vectormesh.data import ...`

```
VectorCache, DatasetSchema, OneHot, Collate, CollateParallel, LabelEncoder, build, HubUpload,
BaseVectorizer, ItemVectorizer, ChunkedVectorizer,
Vectorizer, ImageVectorizer, PatchImageVectorizer, RegexVectorizer, ChunkedRegexVectorizer
```

Not re-exported, import from the module: `detect_device`, `build_legal_reference_pattern`,
`harmonize_legal_reference`, `build_imdb_review_pattern`, `harmonize_imdb_match` (all in
`vectormesh.data.vectorizers`); `aktes_threshold`, `generate_splits` (in
`vectormesh.data.dataset`).

## 7.3 Components — `from vectormesh.components import ...`

```
RankPolymorphicComponent, SequenceComponent, SlidingComponent, ImageComponent,
BaseAggregator, Connector,
Serial, Parallel, Skip, Lift, Siamese, JEPA,
FixedPadding, DynamicPadding, Padded,
MeanAggregator, MaskedMeanAggregator, AttentionAggregator, RNNAggregator, LastAggregator,
NeuralNet, Projection, Dense, Attention, TransformerBlock,
Conv1D, Pool1D, Conv2D, Pool2D, GlobalPool2D, Recurrent,
Concatenate2D, Concatenate3D, ConcatenateSequence, Stack2D,
Gate, Highway, MoE, SparseMoE, load_balancing_loss,
GaussianNoise, Norm, Activation, RepresentationStd
```

The names a component takes by `kind` are types of their own: `NormKind` and `ActivationKind`
(`Literal` types), for the field of a config that holds the choice.

The first two lines list the families (`components/contracts.py`); every other component with a
shape contract is a member of one, except `GlobalPool2D`, the step from an image to a vector —
see [§7.7](#77-components).

Metrics live in `vectormesh.components.metrics` and are **not** in `components.__all__`:
`Metric, Accuracy, F1Score, MAE, MASE`.

---

## 7.4 Vectorizers

### `BaseVectorizer` (abstract, pydantic, frozen)

| Field | Type | Default |
|---|---|---|
| `model_name` | `str` | required |
| `col_name` | `str` | required |
| `input_col` | `str` | `"text"` |
| `device` | `str` | `detect_device()` |

Properties: `get_metadata -> dict`, `get_hidden_size -> int`, `get_context_size -> int | None`.
Abstract: `initialize_model()`, `__call__(inputs: list, batchsize: int)`.
Also: `fingerprint_fields() -> dict` (what the cache hashes; every public field by default,
minus the names in the `FINGERPRINT_EXCLUDE` class var).

### Families: `ItemVectorizer(BaseVectorizer)`, `ChunkedVectorizer(BaseVectorizer)`

Abstract. They fix the shape of one item: `ItemVectorizer.__call__` returns
`dict[str, list[Float[Tensor, "dim"]]]` (cache `tensordtype == 1`), `ChunkedVectorizer.__call__`
returns `dict[str, list[Float[Tensor, "_ dim"]]]` (cache `tensordtype == 2`). Every concrete
vectorizer below is a member of one.

### `Vectorizer(ChunkedVectorizer)`

| Field | Type | Default |
|---|---|---|
| `model_name` | `str` | required |
| `col_name` | `str` | required |
| `max_length` | `int \| None` | `None` (= model maximum) |
| `device` | `str` | auto |

`__call__(inputs: list[str], batchsize: int) -> dict[str, list[Float[Tensor, "_ dim"]]]`

Class constant `STRIDE_DIVISOR = 10`. Extra properties: `get_model`, `get_tokenizer`,
`get_stride`. Public methods `tokenize`, `embed`, `aggregate`, `extend`. Attribute
`chunk_sizes: Counter`.

### `ImageVectorizer(_ImageModelVectorizer, ItemVectorizer)`

| Field | Type | Default |
|---|---|---|
| `model_name` | `str` | required |
| `col_name` | `str` | `"embed"` |
| `input_col` | `str` | `"image"` |
| `device` | `str` | auto |

`__call__(inputs: list, batchsize: int) -> dict[str, list[Float[Tensor, "dim"]]]`

Extra properties: `get_model`, `get_processor`.

### `RegexVectorizer(_RegexFeatures, ItemVectorizer)`

| Field | Type | Default |
|---|---|---|
| `model_name` | `str` | `"regex_vectorizer"` |
| `col_name` | `str` | `"regex_features"` |
| `pattern_builder` | `Callable[[], re.Pattern]` | required |
| `harmonizer` | `Callable[[tuple], str]` | required |
| `training_texts` | `list[str] \| None` | `None` (fits on construction if given) |
| `min_doc_frequency` | `int` | `50` |
| `max_features` | `int` | `1000` |

`__call__(inputs: list[str], batchsize: int = 32) -> dict[str, list[Float[Tensor, "hidden_size"]]]`

Methods: `fit(texts) -> RegexVectorizer`, `print_stats(texts=None, top_k=20, plot=True)`.

### `ChunkedRegexVectorizer(_RegexFeatures, ChunkedVectorizer)`

A sibling of `RegexVectorizer`, sharing its regex fields through `_RegexFeatures`. Adds:

| Field | Type | Default |
|---|---|---|
| `model_name` | `str` | `"chunked_regex_vectorizer"` |
| `col_name` | `str` | `"chunked_regex"` |
| `tokenizer_name` | `str` | required — the embedder's HF tag |
| `context_size` | `int` | required |
| `stride` | `int` | required |

`__call__(...) -> dict[str, list[Float[Tensor, "_ hidden_size"]]]`

Extra properties: `get_stride`, `get_offsets_supported`.

### Pattern helpers

| Function | Returns |
|---|---|
| `build_legal_reference_pattern()` | `re.Pattern` for Dutch legal article references |
| `harmonize_legal_reference(match: tuple)` | e.g. `"7:2 BW"` |
| `build_imdb_review_pattern()` | `re.Pattern` for film sentiment / genre / craft vocabulary |
| `harmonize_imdb_match(match: str)` | lowercased term |
| `detect_device()` | `"cuda"` \| `"mps"` \| `"cpu"` |

---

## 7.5 `VectorCache`

```python
VectorCache.create(
    cache_dir: Path,
    vectorizer: TVectorizer,
    dataset: Dataset,
    dataset_tag: str = "default",
    features: Features | None = None,
    vector_batch: int = 32,
    map_batch: int = 32,
    column_name: str | None = None,
    remove_columns: list[str] | None = None,
) -> VectorCache

VectorCache.load(path: Path) -> VectorCache

VectorCache.from_hub(
    repo_id: str,
    split: str = "train",
    revision: str | None = None,
    token: str | None = None,
    cache_dir: Path | None = None,
) -> VectorCache

cache.push_to_hub(
    repo_id: str | None = None,          # or derive it from org + dataset_name
    split: str = "train",
    *,
    org: str | None = None,              # e.g. "pttrn-io"
    dataset_name: str | None = None,     # e.g. "eurosat"; encoder appended from metadata
    source_dataset: str | None = None,   # pinned to a revision in the card
    drop_columns: list[str] | None = None,
    card: str | None = None,             # default: generated from metadata.json
    write_card: bool = True,
    license: str = "other",
    private: bool = False,
    token: str | None = None,
    dry_run: bool = False,
) -> HubUpload

cache.hub_repo_id(org: str, dataset_name: str) -> str   # {org}/{dataset_name}-{encoder}

cache.join(
    other: VectorCache,
    on: str = "source_idx",
    column: str | None = None,     # default: other's single vector column
    into: str | None = None,       # default: column, suffixed with other's encoder slug
) -> VectorCache
```

Exactly one of `repo_id` or `org`+`dataset_name` — passing both raises, passing neither
raises.

`HubUpload` (frozen pydantic model, importable from `vectormesh`): `repo_id`, `split`,
`files: list[str]`, `nbytes: int`, `megabytes: float`, `card: str`, `url: str`,
`uploaded: bool`. `files` is the whole payload — arrow and json only, re-serialised rather
than copied from `cache_dir`.

Fields: `name: str`, `cache_dir: Path`, `dataset: Dataset | None`, `metadata: dict | None`.

Properties: `vector_columns -> list[str]` — the columns described by `metadata.json`, i.e.
the ones a vectorizer produced (as opposed to labels, keys and source columns).

Static/class helpers: `update_metadata(path, new_metadata)`,
`get_features(dataset, tensord, embedding_column)`, `get_dtensor(vectorizer)`.

Dunders proxy to the wrapped `Dataset`: `__len__`, `__getitem__`, `__iter__`, `__getattr__` —
so `cache.select`, `cache.map`, `cache.features`, `cache.column_names`, `cache["text"]` work.

Output folder name: `{YYYYMMDDHHMMSS}_{dataset_tag}_{column_name}`.

---

## 7.6 Schema, targets, batching

| Class | Signature |
|---|---|
| `DatasetSchema` | fields `input_col: str`, `label_col: str \| None`; `DatasetSchema.infer(dataset, *, input_col=None, label_col=None)` |
| `OneHot` | `OneHot(num_classes: int, label_col: str, target_col: str)`; call on an observation |
| `Collate` | `Collate(embedding_col: str, target_col: str, padder: Callable)` → `(X, y)` |
| `CollateParallel` | `CollateParallel(vec1_col, vec2_col, target_col, padder, padder2=None)` → `((X1, X2), y)` |
| `LabelEncoder` | `LabelEncoder(train_codes: list[int])`; `onehot`, `encode`, `decode(vector, threshold)`, `save(path)`, `from_file(path)` |
| `build` | `build(input_file, threshold, trainsplit, testvalsplit, output_dir) -> None` |
| `aktes_threshold` | `(file_path, threshold) -> (Dataset, set[int])` |
| `generate_splits` | `(path, threshold, trainsplit, testvalsplit) -> (dict[str, Dataset], set[int])` |

`DatasetSchema.INPUT_ALIASES` = `text, texts, sentence, sentences, content, document, image, img`
`DatasetSchema.LABEL_ALIASES` = `label, labels, target, targets, class, classes, category`

---

## 7.7 Components

| Component | Family | Constructor | Input → Output |
|---|---|---|---|
| `Serial` | — | `(components: list[nn.Module])` | passthrough |
| `Parallel` | — | `(branches: list[nn.Module])` | tuple → tuple |
| `FixedPadding` | — (collate callable) | `(max_chunks: int)` | `list[(C_i, D)]` → `Padded`, values `(B, max_chunks, D)` |
| `DynamicPadding` | — (collate callable) | `()` | `list[(C_i, D)]` → `Padded`, values `(B, max C_i, D)` |
| `MeanAggregator` | `BaseAggregator` | `()` | `(B, _, D)` → `(B, D)` |
| `MaskedMeanAggregator` | `BaseAggregator` | `()` | `(B, C, D)` → `(B, D)`; needs `mask` (a `Padded` batch) |
| `AttentionAggregator` | `BaseAggregator` | `(hidden_size: int)` | `(B, _, D)` → `(B, D)` |
| `RNNAggregator` | `BaseAggregator` | `(hidden_size: int)` | `(B, _, D)` → `(B, D)` |
| `LastAggregator` | `BaseAggregator` | `()` | `(B, _, D)` → `(B, D)`, the last step |
| `NeuralNet` | `RankPolymorphicComponent` | `(hidden_size: int \| list[int], out_size: int)` | `(…, hidden)` → `(…, out)` |
| `Projection` | `RankPolymorphicComponent` | `(hidden_size: int, out_size: int)` | `(…, hidden)` → `(…, out)` |
| `Dense` | `RankPolymorphicComponent` | `(in_size: int, out_size: int, norm: 'layer' \| 'batch' \| 'rms' \| 'none' = 'none', activation: 'gelu' \| 'relu' \| 'tanh' \| 'sigmoid' \| 'none' = 'gelu', dropout: float = 0.0)` | `(…, in_size)` → `(…, out_size)`: Linear, Norm, Activation, Dropout |
| `Attention` | `SequenceComponent` | `(hidden_size, num_heads=8, dropout=0.1)` | `(B, seq, D)` → `(B, seq, D)` |
| `TransformerBlock` | `SequenceComponent` | `(hidden_size, num_heads=8, dropout=0.1)` | `(B, seq, hidden)` → `(B, seq, hidden)`, shape-preserving |
| `Recurrent` | `SequenceComponent` | `(in_size: int, hidden_size: int, kind: 'rnn' \| 'gru' \| 'lstm' = 'gru', n_layers: int = 1)` | `(B, seq, in_size)` → `(B, seq, hidden)`, the state at every step |
| `Conv1D` | `SlidingComponent` | `(in_size: int, out_size: int, kernel_size: int, stride=1, padding=0, dilation=1)` | `(B, seq, in_size)` → `(B, output_length(seq), out_size)` |
| `Pool1D` | `SlidingComponent` | `(kernel_size: int, kind: 'max' \| 'mean' = 'max')` | `(B, seq, D)` → `(B, seq // kernel_size, D)` |
| `Conv2D` | `ImageComponent` | `(in_size: int, out_size: int, kernel_size: int, stride=1, padding=0, dilation=1)` | `(B, in_size, H, W)` → `(B, out_size, *output_size(H, W))` |
| `Pool2D` | `ImageComponent` | `(kernel_size: int, kind: 'max' \| 'mean' = 'max')` | `(B, C, H, W)` → `(B, C, H // kernel_size, W // kernel_size)` |
| `GlobalPool2D` | — (image to vector) | `(kind: 'max' \| 'mean' = 'mean')` | `(B, C, H, W)` → `(B, C)` |
| `Concatenate2D` | `Connector` | `()` | `((B, D1), …)` → `(B, ΣD)` |
| `Concatenate3D` | `Connector` | `()` | `((B, C, D1), …)` → `(B, C, ΣD)` |
| `ConcatenateSequence` | `Connector` | `()` | `((B, C1, D), …)` → `(B, ΣC, D)` |
| `Stack2D` | `Connector` | `()` | `((B, D), …)` → `(B, n, D)` |
| `Skip` | — (architecture form) | `(transform: nn.Module, in_size: int, projection: nn.Module \| None = None)` | `(…, in_size)` → `(…, out)` |
| `Norm` | `RankPolymorphicComponent` | `(hidden_size: int, kind: 'layer' \| 'batch' \| 'rms' \| 'none' = 'layer')` | `(…, hidden)` → `(…, hidden)` |
| `Activation` | `RankPolymorphicComponent` | `(kind: 'gelu' \| 'relu' \| 'tanh' \| 'sigmoid' \| 'none' = 'gelu')` | `(…, hidden)` → `(…, hidden)`; an unknown kind is a `VectorMeshError` |
| `Gate` | `RankPolymorphicComponent` | `(hidden_size: int)` | `(…, hidden)` → `(…, hidden)` |
| `Highway` | `RankPolymorphicComponent` | `(transform: nn.Module, hidden_size: int)` | `(…, hidden)` → `(…, hidden)` |
| `MoE` | `RankPolymorphicComponent` | `(experts: list[nn.Module], hidden_size: int, out_size: int)` | `(…, hidden)` → `(…, out)` |
| `SparseMoE` | `RankPolymorphicComponent` | `(experts: Sequence[nn.Module], hidden_size: int, out_size: int, k: int = 1, normalize: bool = False)`; `.aux_loss` after each forward | `(…, hidden)` → `(…, out)`; `forward(tensors, mask=None)` skips padded positions |
| `GaussianNoise` | `RankPolymorphicComponent` | `(sigma: float = 0.1, relative: bool = True)` | `(…, D)` → `(…, D)`, train-only |

A `Connector`'s `forward(tensors)` calls `connect(*tensors)`; the shape contract is on
`connect`, so every branch is checked.

Note the keyword is `hidden_size` (not `in_size`) for `NeuralNet` and `Projection`; `Skip` is the
one that takes `in_size`. `NeuralNet(hidden_size=[d, w, …], out_size)` names the whole width chain
from the input dimension through every hidden layer; a bare `int` means one hidden layer as wide
as the input.

`Padded(values, mask)` (`vectormesh.components` or `vectormesh.types`) is a frozen dataclass:
`values: (B, seq, …)`, `mask: Bool (B, seq)`, True for real positions; `.shape` is the values'
shape, `.lengths` the real positions per item, `.to(device)` moves both. Every component takes
one (see [§4.2](04-components.md#42-padding--paddingpy)). The components whose `forward` takes
`mask: Bool[Tensor, "batch seq"] | None = None` read it: `MaskedMeanAggregator` (required),
`AttentionAggregator`, `RNNAggregator`, `LastAggregator`, `Attention`, `TransformerBlock`,
`Recurrent`, `SparseMoE`, and `Skip`, `Highway`, `MoE`, which pass it on. `Conv1D` and `Pool1D`
have `output_mask(mask)`. A `Connector` joins masks with `connect_mask(*masks)`:
`Concatenate3D` (any branch real), `ConcatenateSequence` (concatenated).

---

## 7.8 Metrics — `vectormesh.components.metrics`

| Class | Constructor | `__repr__` |
|---|---|---|
| `Accuracy` | `()` | `"Accuracy"` |
| `F1Score` | `(average="micro", threshold=0.5, epsilon=1e-7)` | `"F1-micro"` |
| `MAE` | `()` | `"MAE"` |
| `MASE` | `(train: Iterator, horizon: int)` | `"MASE(scale=…)"` |

All are called as `metric(y, yhat)` with `torch.Tensor` or `np.ndarray`, and return `float`.

---

## 7.9 Types — `vectormesh.types`

| Name | Purpose |
|---|---|
| `VectorMeshError(message, hint=None, fix=None)` | library exception carrying an educational hint and suggested fix |
| `Cachable` | pydantic `BaseModel` with `frozen=True`, `arbitrary_types_allowed=True` |
| `BaseComponent` | `nn.Module` + `ABC` with abstract `forward` returning a tensor |
| `TensorInput` | `Union[Float[Tensor, "..."], Tuple[Float[Tensor, "..."], ...]]` — used by `Serial`/`Parallel` |

---

## 7.9b Design — `from vectormesh.design import ...`

| Name | Purpose |
|---|---|
| `option(axis)` | decorator on a stage (a method of the design's `nn.Module`) the search may leave out; `axis` is the axis of the search space that decides it |
| `axis_of(stage)` | the axis `option` named on `stage`, or `None` |
| `Design(module, config)` | the design class and the config dataclass that holds its choices |
| `Design.options()` | the axis names of the options, in the order the stages are written |
| `Design.stages()` | the names of the stages: the module's public methods other than `forward` |
| `Design.choices()` | the field names of the config |
| `Design.drawn(config)` | the config dataclass built from the design's choices in a drawn config |
| `Design.combinations()` | every combination of the options, each on or off |
| `Design.trace(model, example)` | `list[StageShape]`: the stages `model(example)` runs, each with the shape it gives |
| `Design.build(config, **widths)` | the module a config chooses: `config` is the config dataclass or a drawn config; `widths` go to the module after its config |
| `Design.show(example, **widths)` | builds every combination of the options with `build` and prints, per combination, the number of params and the shape after every stage |
| `StageShape(stage, shape)` | pydantic model: one stage a forward pass ran, and the shape it gave |

See [Architectures §5.8](05-architectures.md).

---

## 7.10 Training — `from vectormesh.training import ...` (also top-level)

```python
Trainer(
    model: nn.Module,
    settings: TrainerSettings,
    loss_fn: Callable,
    optimizer: type[Optimizer],
    traindataloader: Iterable,
    validdataloader: Iterable,
    scheduler: Callable | None = None,
    device: str | None = None,
    reporters: Sequence[Reporter] = (),
    step: Step | None = None,
    progress: bool = True,
)
Trainer.loop() -> TrainResult
```

| Class | Fields / signature |
|---|---|
| `TrainerSettings` (pydantic) | `epochs: int`, `metrics: list[Callable]`, `logdir: Path`, `train_steps: int`, `valid_steps: int`, `optimizer_kwargs: dict = {}`, `scheduler_kwargs: dict \| None = {"factor": 0.1, "patience": 10}`, `earlystop_kwargs: dict \| None` **(required, no default)**. Plain data — no logging-backend field; see `reporters` above. |
| `TrainResult` (dataclass) | `epoch: int`, `train_loss: float`, `test_loss: float`, `metric_dict: dict[str, float]`, `stopped_epoch: int` — what `.loop()` returns. The first four describe the model `trainer.model` holds after the loop: the best epoch by validation loss with early stopping, the last epoch without. `stopped_epoch` is the last epoch that ran (defaults to `epoch`). |
| `Reporter` (`Protocol`) | `__call__(epoch: int, train_loss: float, test_loss: float, metric_dict: dict[str, float]) -> None` — any matching callable qualifies, no subclassing |
| `Step` (`Protocol`) | `__call__(model: nn.Module, x: BatchTensor, y: torch.Tensor) -> torch.Tensor`. Default (built from `loss_fn`): `lambda model, x, y: loss_fn(model(x), y)`. Governs the loss computation in both `trainbatches()` and `evalbatches()` — see [§6.6](06-training.md#66-step-pluggable-loss-computation). |
| `EarlyStopping` | `(log_dir: Path, patience=7, verbose=False, delta=0.0, save=False)`; called as `early_stopping(val_loss, model)`; keeps `.best_loss`, `.best_state` (an in-memory `state_dict` copy of the best epoch), `.improved` (the last call improved) and `.early_stop`; `.get_best() -> nn.Module` loads the `save=True` checkpoint from disk |

`Trainer.log_dir` is computed at construction (`logdir / f"{timestamp}-{hex6}"`) but never created
on disk unless something writes to it — currently only `EarlyStopping.save_checkpoint` when
`save=True`. mlflow/ray/tensorboard adapters: [10. Reporters cookbook](10-reporters.md).

---

## 7.11 Easily-confused signatures

These are the ones that bite in practice:

| Looks like | Actually |
|---|---|
| `MoE(..., top_k=…)` | no `top_k` — `MoE` is **dense**: every expert runs, softmax-blended |
| `Projection(in_size=…)` | `Projection(hidden_size=…, out_size=…)` |
| `NeuralNet(..., dropout=…)` | no dropout — two `Linear`s + `GELU`. Add `nn.Dropout` yourself in the `Serial` |
| `TransformerBlock(h, output_size=…)` | shape-preserving, single `hidden_size`. Use `Projection` to change width |
| `Skip(hidden_size=…)` | `Skip` takes `in_size`; `NeuralNet`/`Projection`/`Gate`/`Highway`/`MoE` take `hidden_size` |
| `Collate(padder=…)` is always a padder | for 1D vector columns (image, plain regex) pass `padder=torch.stack` |

Next: [Teaching path](08-teaching-path.md).
