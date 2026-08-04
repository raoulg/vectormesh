# 4. Components

Source: `src/vectormesh/components/`. Everything here is an `nn.Module`; most inherit
`BaseComponent` (`types.py`), which adds nothing but an abstract `forward` returning a tensor.

Shorthand used below: `B` = batch, `C` = chunks, `D` = dim.

---

## 4.1 Pipelines — `pipelines.py`

### `Serial(components: list[nn.Module])`

Runs components in order, feeding each output into the next. That is all it does — no shape
checking of its own, because each component checks itself.

```python
pipeline = Serial([
    MaskedMeanAggregator(),          # (B, C, D) -> (B, D)
    NeuralNet(hidden_size=768, out_size=32),   # (B, 768) -> (B, 32)
])
```

`Serial` is itself an `nn.Module`, so it nests inside another `Serial` or inside a `Parallel`
branch, and its parameters register normally with the optimizer.

### `Parallel(branches: list[nn.Module])`

Takes a **tuple** of tensors, zips it against the branch list, and returns a tuple of outputs.
Branch *i* receives tensor *i* — they do not all see the same input.

```python
parallel = Parallel([
    Serial([MeanAggregator(), NeuralNet(768, 32)]),   # branch 1 gets X1: (B, C, 768)
    Serial([NeuralNet(123, 32)]),                     # branch 2 gets X2: (B, 123)
])
```

The number of branches must match the number of tensors produced by `CollateParallel`. Since
`zip` stops at the shorter of the two, a mismatch silently drops a branch rather than raising —
count them yourself.

A `Parallel` returns a tuple, so the *next* element in the enclosing `Serial` must be a
connector (`Concatenate2D`, `Concatenate3D`, `Stack2D`) that consumes tuples.

---

## 4.2 Padding — `padding.py`

Padders are plain callables (not `nn.Module`s) used inside `Collate`, before the model.

| Class | Signature | Output | Trade-off |
|---|---|---|---|
| `FixedPadding(max_chunks)` | `list[(C_i, D)]` → `(B, max_chunks, D)` | pads short, **truncates long** | static shape; loses tail of long documents |
| `DynamicPadding()` | `list[(C_i, D)]` → `(B, max(C_i), D)` | pads to batch max | no data loss; shape varies per batch |

Pick `max_chunks` from the `chunk_sizes` histogram in the cache metadata. In the legal dataset,
`30` covers the bulk of documents; `45` (used by `scripts/train_moe_parallel.py`) trades compute
for more coverage.

**Question the notebooks pose:** which architectures can handle a chunk axis that changes size
between batches, and which cannot? (RNNs and masked-mean pooling can; anything with a learned
parameter *per chunk position* cannot.)

---

## 4.3 Aggregation — `aggregation.py`

All aggregators are `(B, C, D) -> (B, D)`. This is the step that lets a variable-length document
become a fixed-size representation.

| Class | Parameters | Behaviour |
|---|---|---|
| `MeanAggregator()` | none | plain `tensors.mean(dim=1)` — **counts zero-padding into the mean** |
| `MaskedMeanAggregator()` | none | detects padded chunks (all-zero rows) and excludes them; `clamp(min=1)` guards against an all-padding row |
| `AttentionAggregator(hidden_size)` | `Linear(D, 1)` | softmax weights over chunks, then weighted sum — learns *which* chunks matter |
| `RNNAggregator(hidden_size)` | `GRU(D, D)` | runs a GRU over chunks, returns the last hidden state — the only aggregator that is order-sensitive |

**Choosing one:** if your padder is `FixedPadding`, `MaskedMeanAggregator` is almost always the
right default — `MeanAggregator` will systematically shrink short documents toward zero.

> **Caveat:** `MaskedMeanAggregator` detects padding by looking for **all-zero rows**. That works
> immediately after a padder, and survives `Concatenate3D` (zeros concatenated with zeros are
> still zeros). It does **not** survive a `TransformerBlock` or any other block with a bias or a
> `LayerNorm`, which writes non-zero values at padded positions. Put it directly after the
> padding, or accept that it has nothing left to mask — see
> [§5.5](05-architectures.md#55-chunk-level-fusion-with-an-moe-transformer).

`AttentionAggregator` is the natural next step when you suspect only a few chunks carry the
signal. `RNNAggregator` is the one to reach for when chunk *order* is meaningful; note it does
not mask, so padding chunks are fed to the GRU.

---

## 4.4 Neural blocks — `neural.py`

### `NeuralNet(hidden_size, out_size)`

Two-layer MLP: `Linear(h, h) → GELU → Linear(h, out)`. Annotated `"... {self.hidden_size}"`, so
it works both as a classification head on `(B, D)` and as a position-wise transform on
`(B, C, D)` — which is exactly how it is used as an expert inside `MoE`.

### `Projection(hidden_size, out_size)`

A single `Linear`. Same rank polymorphism. Use it to reshape a dimension without adding
non-linearity — e.g. matching a fused `emb + regex` width down to the model's working width.

### `Attention(hidden_size, num_heads=8, dropout=0.1)`

Bare multi-head self-attention on `(B, seq, D)`, no residual, no norm, no padding mask. A
teaching primitive; prefer `TransformerBlock` in real pipelines.

### `TransformerBlock(hidden_size, num_heads=8, dropout=0.1)`

A minimal **pre-norm** block:

```
x = x + Attention(norm1(x))
x = x + FFN(norm2(x))
```

Three things worth knowing:

- **Shape-preserving by construction.** Both residual additions require the block to return the
  width it received, so there is a single `hidden_size` and no output width to choose. Put a
  `Projection` before or after the block to change width.
- **`_pad_mask`** reconstructs the key-padding mask from all-zero rows, so you do not have to
  thread a mask through the pipeline. A fully-padded row would make the attention softmax
  produce `NaN`, so such a row is treated as fully valid instead.
- It is rank-3 only (`(B, seq, D)`), by design.

---

## 4.5 Connectors — `connectors.py`

Consume the tuple that a `Parallel` returns; produce a single tensor.

| Class | Input | Output |
|---|---|---|
| `Concatenate2D()` | `((B, D1), (B, D2), …)` | `(B, D1+D2+…)` — widths may differ |
| `Concatenate3D()` | `((B, C, D1), (B, C, D2), …)` | `(B, C, D1+D2+…)` |
| `Stack2D()` | `((B, D), (B, D), …)` | `(B, n, D)` — a new axis |

`Concatenate3D` requires the branches to share the chunk axis, which is why both padders in
`CollateParallel` must use the same `max_chunks`. A useful property: a padded chunk stays
all-zero after concatenation, so downstream padding detection (`MaskedMeanAggregator`,
`TransformerBlock._pad_mask`) still fires correctly.

`Stack2D` is the interesting one pedagogically: it converts *n parallel representations* into a
*sequence of length n*, which you can then feed to attention. Fusion by attention rather than
by concatenation.

---

## 4.6 Gating — `gating.py`

The theme of this module is **learned interpolation**: instead of committing to a transform,
learn how much of it to apply.

### `Skip(transform, in_size, projection=None)`

```
y = transform(norm(x)) + (projection(norm(x)) if projection else norm(x))
```

Pre-norm residual connection. `projection` is needed only when `transform` changes the
dimension. Annotated `"..."` — works at any rank.

### `Gate(hidden_size)`

```
y = sigmoid(W x) * x
```

Element-wise multiplicative gate: the network learns to suppress dimensions. Rank-2 only.
Composes nicely *inside* a `Skip`'s transform.

### `Highway(transform, hidden_size)`

```
y = g * T(x) + (1 - g) * x,   g = sigmoid(W · norm(x))
```

A learned interpolation between "transform it" and "pass it through". Where `Skip` always adds
the transform, `Highway` can learn to route around it entirely. See *Highway Networks* in
`references/`. Rank-2 only.

### `MoE(experts, hidden_size, out_size)`

```
y = Σ_i softmax(W x)_i · expert_i(x)
```

A **dense** mixture of experts: every expert runs, and a router produces a softmax weighting
over them. The module docstring lays out the family clearly:

```
Gate:     sigmoid(Wx) * x                    # 1 gate,  1 transform
Highway:  g * T(x) + (1 - g) * x             # 1 gate,  2 experts (T, identity)
MoE:      sum_i softmax(Wx)_i * expert_i(x)  # N gates, N experts
```

Routing is **per position**, so the layer accepts `(B, D)` and `(B, C, D)` alike — with rank-3
input each chunk gets its own expert mixture. Experts must map `(..., hidden_size)` to
`(..., out_size)`; `NeuralNet` and `TransformerBlock` both qualify.

> **Note on the paper.** This is the Jacobs & Jordan (1991) *dense* formulation, not the
> sparsely-gated top-k routing of Shazeer et al. (2017) in `references/`. There is no `top_k`
> argument, no noisy gating, and no load-balancing loss: every expert is evaluated on every
> input, so this buys **capacity and specialisation, not compute savings**. Sparse top-k routing
> is a natural extension exercise — and one that is genuinely hard to make train at this
> dataset scale.

---

## 4.7 Augmentation — `augmentation.py`

### `GaussianNoise(sigma=0.1, relative=True)`

Adds Gaussian noise to *embeddings*, active only in `train()` mode (a no-op under `eval()`, so
validation stays deterministic).

Why feature-space rather than pixel-space: classic augmentation (random crops, flips) wants a
fresh view every epoch, which would mean re-running the frozen encoder every epoch — the exact
cost the cache exists to avoid. Perturbing the cached vector is cheap, fresh on every step, and
a genuine regulariser for a small head trained on few examples.

`relative=True` (the default) scales the noise by each sample's own standard deviation over the
feature axis, so one `sigma` behaves consistently across encoders whose embeddings live on
different scales (a ResNet pooler vs. a DINOv2 CLS vector).

Annotated `"..."`, shape-preserving, so it drops straight in front of a head:

```python
Serial([GaussianNoise(sigma=0.1), NeuralNet(hidden_size, n_classes)])
```

Sanity check used in notebook 4:

```python
pipeline.train(); assert not torch.allclose(pipeline(X), pipeline(X))
pipeline.eval();  assert     torch.allclose(pipeline(X), pipeline(X))
```

---

## 4.8 Metrics — `metrics.py`

All metrics subclass `Metric`, handle numpy/tensor conversion in `__call__`, and return a plain
`float` so `mltrainer` can log them.

| Class | Use |
|---|---|
| `Accuracy()` | top-1. Accepts integer targets `(B,)` **or** one-hot/soft targets `(B, n)` — in the latter case it takes `argmax` of both sides |
| `F1Score(average="micro", threshold=0.5, epsilon=1e-7)` | element-wise on sigmoid probabilities; `"micro"` or `"macro"` |
| `MAE()` | mean absolute error (regression) |
| `MASE(train, horizon)` | mean absolute scaled error against a naive forecast (time series) |

`F1Score` applies `sigmoid` internally and thresholds — it is built for the **multi-label** case
where a document carries several labels, which is why the legal task pairs it with
`BCEWithLogitsLoss`. `Accuracy` assumes single-label, so on a genuinely multi-label task it
measures only whether the top prediction is *a* correct label.

Each metric defines `__repr__` (`"F1-micro"`, `"Accuracy"`) — that string is what appears in
TensorBoard and MLflow, so keep it stable.

Next: [Architectures](05-architectures.md).
