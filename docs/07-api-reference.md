# 7. API reference

Signatures as implemented in `src/vectormesh/` (version 0.3.0). `B` = batch, `C` = chunks,
`D` = dim.

---

## 7.1 Top level — `from vectormesh import ...`

```
VectorCache, Vectorizer, ImageVectorizer, RegexVectorizer, ChunkedRegexVectorizer,
BaseVectorizer, LabelEncoder, build, __version__
```

## 7.2 Data — `from vectormesh.data import ...`

```
VectorCache, DatasetSchema, OneHot, Collate, CollateParallel, LabelEncoder, build,
BaseVectorizer, Vectorizer, ImageVectorizer, RegexVectorizer, ChunkedRegexVectorizer
```

Not re-exported, import from the module: `detect_device`, `build_legal_reference_pattern`,
`harmonize_legal_reference`, `build_imdb_review_pattern`, `harmonize_imdb_match` (all in
`vectormesh.data.vectorizers`); `aktes_threshold`, `generate_splits` (in
`vectormesh.data.dataset`).

## 7.3 Components — `from vectormesh.components import ...`

```
Serial, Parallel,
FixedPadding, DynamicPadding,
BaseAggregator, MeanAggregator, MaskedMeanAggregator, AttentionAggregator, RNNAggregator,
NeuralNet, Projection, Attention, TransformerBlock,
Concatenate2D, Concatenate3D, Stack2D,
Skip, Gate, Highway, MoE,
GaussianNoise
```

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

### `Vectorizer(BaseVectorizer)`

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

### `ImageVectorizer(BaseVectorizer)`

| Field | Type | Default |
|---|---|---|
| `model_name` | `str` | required |
| `col_name` | `str` | `"embed"` |
| `input_col` | `str` | `"image"` |
| `device` | `str` | auto |

`__call__(inputs: list, batchsize: int) -> dict[str, list[Float[Tensor, "dim"]]]`

Extra properties: `get_model`, `get_processor`.

### `RegexVectorizer(BaseVectorizer)`

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

### `ChunkedRegexVectorizer(RegexVectorizer)`

Adds:

| Field | Type | Default |
|---|---|---|
| `model_name` | `str` | `"chunked_regex_vectorizer"` |
| `col_name` | `str` | `"chunked_regex"` |
| `tokenizer_name` | `str` | required — the embedder's HF tag |
| `context_size` | `int` | required |
| `stride` | `int` | required |

`__call__(...) -> dict[str, list[Float[Tensor, "chunks hidden_size"]]]`

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
```

Fields: `name: str`, `cache_dir: Path`, `dataset: Dataset | None`, `metadata: dict | None`.

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

| Component | Constructor | Input → Output |
|---|---|---|
| `Serial` | `(components: list[nn.Module])` | passthrough |
| `Parallel` | `(branches: list[nn.Module])` | tuple → tuple |
| `FixedPadding` | `(max_chunks: int)` | `list[(C_i, D)]` → `(B, max_chunks, D)` |
| `DynamicPadding` | `()` | `list[(C_i, D)]` → `(B, max C_i, D)` |
| `MeanAggregator` | `()` | `(B, _, D)` → `(B, D)` |
| `MaskedMeanAggregator` | `()` | `(B, _, D)` → `(B, D)` |
| `AttentionAggregator` | `(hidden_size: int)` | `(B, _, D)` → `(B, D)` |
| `RNNAggregator` | `(hidden_size: int)` | `(B, _, D)` → `(B, D)` |
| `NeuralNet` | `(hidden_size: int, out_size: int)` | `(…, hidden)` → `(…, out)` |
| `Projection` | `(hidden_size: int, out_size: int)` | `(…, hidden)` → `(…, out)` |
| `Attention` | `(hidden_size, num_heads=8, dropout=0.1)` | `(B, seq, D)` → `(B, seq, D)` |
| `TransformerBlock` | `(hidden_size, num_heads=8, dropout=0.1)` | `(B, seq, hidden)` → `(B, seq, hidden)`, shape-preserving |
| `Concatenate2D` | `()` | `((B, D1), …)` → `(B, ΣD)` |
| `Concatenate3D` | `()` | `((B, C, D1), …)` → `(B, C, ΣD)` |
| `Stack2D` | `()` | `((B, D), …)` → `(B, n, D)` |
| `Skip` | `(transform: nn.Module, in_size: int, projection: nn.Module \| None = None)` | `(…)` → `(…)` |
| `Gate` | `(hidden_size: int)` | `(B, D)` → `(B, D)` |
| `Highway` | `(transform: nn.Module, hidden_size: int)` | `(B, D)` → `(B, D)` |
| `MoE` | `(experts: list[nn.Module], hidden_size: int, out_size: int)` | `(…, hidden)` → `(…, out)` |
| `GaussianNoise` | `(sigma: float = 0.1, relative: bool = True)` | `(…)` → `(…)`, train-only |

Note the keyword is `hidden_size` (not `in_size`) for `NeuralNet` and `Projection`; `Skip` is the
one that takes `in_size`.

---

## 7.8 Metrics — `vectormesh.components.metrics`

| Class | Constructor | `__repr__` |
|---|---|---|
| `Accuracy` | `()` | `"Accuracy"` |
| `F1Score` | `(average="micro", threshold=0.5, epsilon=1e-7)` | `"F1-micro"` |
| `MAE` | `()` | `"MAE"` |
| `MASE` | `(train: Iterator, horizon: int)` | `"MASE-scale=…"` |

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

## 7.10 Easily-confused signatures

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
