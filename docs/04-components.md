# 4. Components

Source: `src/vectormesh/components/`. Everything here is an `nn.Module`; most inherit
`BaseComponent` (`types.py`), which adds nothing but an abstract `forward` returning a tensor.
A component with a shape contract inherits it from its family in `contracts.py` —
`RankPolymorphicComponent`, `SequenceComponent`, `SlidingComponent`, `ImageComponent`,
`BaseAggregator` or `Connector` — so the
class says which rung of the ladder it works on; see
[§2.6](02-tensor-contracts.md#26-families-the-rung-is-the-class). The architecture forms in
`pipelines.py` (`Serial`, `Parallel`, `Skip`, `Lift`) have no family: they take the shape of
what they hold.

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

### `Skip(transform, in_size, projection=None)`

```
y = transform(norm(x)) + (projection(norm(x)) if projection else norm(x))
```

Pre-norm residual connection around `transform`. `projection` is needed only when `transform`
changes the width. Like `Serial`, `Skip` has no rank of its own: it takes `(*lead, in_size)`
and works at whatever rank its transform does.

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
| `RNNAggregator(hidden_size)` | `GRU(D, D)` | runs a GRU over the real chunks (all-zero rows skipped), returns the hidden state after the last one — the only aggregator with parameters that is order-sensitive |
| `LastAggregator()` | none | the last step, `tensors[:, -1]` — after a `Recurrent`, the state after the whole sequence; padding at the end is read as the last step |

**Choosing one:** if your padder is `FixedPadding`, `MaskedMeanAggregator` is almost always the
right default — `MeanAggregator` will systematically shrink short documents toward zero.

> **Caveat:** `MaskedMeanAggregator` detects padding by looking for **all-zero rows**. That works
> immediately after a padder, and survives `Concatenate3D` (zeros concatenated with zeros are
> still zeros). It does **not** survive a `TransformerBlock` or any other block with a bias or a
> `LayerNorm`, which writes non-zero values at padded positions. Put it directly after the
> padding, or accept that it has nothing left to mask — see
> [§5.5](05-architectures.md#55-chunk-level-fusion-with-an-moe-transformer).

`AttentionAggregator` is the natural next step when you suspect only a few chunks carry the
signal. `RNNAggregator` is the one to reach for when chunk *order* is meaningful. It detects
padding the same way as `MaskedMeanAggregator`, moves each item's real chunks to the front in
their own order, and packs the sequence, so the GRU never sees a padded chunk: padding at the end
and gaps in the middle (after concatenating two padded sequences) give the same result as the
real chunks alone. The same caveat applies: put it directly after the padding.

---

## 4.4 Neural blocks — `neural.py`

### `NeuralNet(hidden_size, out_size)`

Feedforward MLP with GELU between every pair of `Linear`s. `hidden_size` is an `int` or a
`list[int]`. An int `h` gives `Linear(h, h) → GELU → Linear(h, out)`; a list names the whole
width chain from the input dimension, so `NeuralNet([384, 128], 47)` is
`Linear(384, 128) → GELU → Linear(128, 47)`. Annotated `"*lead {self.hidden_size}"`, so
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

Consume the tuple that a `Parallel` returns; produce a single tensor. All four are
`Connector`s: `forward(tensors)` hands the tuple on as `connect(*tensors)`, so every branch is
checked and the named axes bind across branches.

| Class | Input | Output |
|---|---|---|
| `Concatenate2D()` | `((B, D1), (B, D2), …)` | `(B, D1+D2+…)` — widths may differ |
| `Concatenate3D()` | `((B, C, D1), (B, C, D2), …)` | `(B, C, D1+D2+…)` |
| `ConcatenateSequence()` | `((B, C1, D), (B, C2, D), …)` | `(B, C1+C2+…, D)` — chunk counts may differ |
| `Stack2D()` | `((B, D), (B, D), …)` | `(B, n, D)` — a new axis |

`Concatenate3D` requires the branches to share the chunk axis, which is why both padders in
`CollateParallel` must use the same `max_chunks`. A useful property: a padded chunk stays
all-zero after concatenation, so downstream padding detection (`MaskedMeanAggregator`,
`TransformerBlock._pad_mask`) still fires correctly.

`ConcatenateSequence` lengthens the sequence instead of the features: it puts the branches one
after the other, so a `TransformerBlock` can attend across them (text chunks next to image
patches). The widths must match, so a branch usually ends in a `Projection`. The padding of each
branch lands in the middle of the joined sequence; `MaskedMeanAggregator`, `RNNAggregator` and
`TransformerBlock._pad_mask` find all-zero rows at any position.

`Stack2D` is the interesting one pedagogically: it converts *n parallel representations* into a
*sequence of length n*, which you can then feed to attention. Fusion by attention rather than
by concatenation.

---

## 4.5b Normalisation — `normalization.py`

### `Norm(hidden_size, kind="layer")`

```
layer:  (x - mean(x)) / std(x)      over each vector's features
batch:  (x - mean_B) / std_B        over the batch, per feature (running stats at eval)
rms:    x / rms(x)                  over each vector's features, no centring
none:   x
```

One component, four values of one axis, so a search can ask "does normalisation help,
and which" as `tune.choice(["none", "layer", "batch"])`. Any rank: at `(B, C, D)` the
batch norm pools every chunk of the batch. Keeps the width.

---

## 4.5c Convolution — `convolution.py`

A window slides along a sequence or over an image. Every component says what shape it will
return before it runs: `output_length(length)` for a sequence, `output_size(height, width)` for
an image, the formula PyTorch uses:

```
out = (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
```

With `stride=1` and an odd kernel, `padding=(kernel_size - 1) // 2 * dilation` keeps the length.

### `Conv1D(in_size, out_size, kernel_size, stride=1, padding=0, dilation=1)`

A `SlidingComponent`: `(B, seq, in_size) -> (B, seq_out, out_size)`. The channels stay on the
last axis, as everywhere in the library; `Conv1D` moves them to axis 1 for `nn.Conv1d` and back.
`in_size` is checked at the boundary, so a sequence in PyTorch's `(B, channels, seq)` layout is
refused rather than convolved along the wrong axis.

```python
conv = Conv1D(in_size=3, out_size=16, kernel_size=5)
x = torch.randn(8, 200, 3)  # (B, seq, channels)
assert conv.output_length(200) == 196
assert conv(x).shape == (8, 196, 16)
```

### `Pool1D(kernel_size, kind="max")`

A `SlidingComponent` that keeps the width: each window of `kernel_size` steps becomes one step,
the maximum (`"max"`) or the mean (`"mean"`) of each channel. A tail shorter than a window is
dropped.

```python
pool = Pool1D(4, kind="mean")
assert pool.output_length(196) == 49
assert pool(torch.randn(8, 196, 16)).shape == (8, 49, 16)
```

### `Conv2D(in_size, out_size, kernel_size, stride=1, padding=0, dilation=1)` and `Pool2D(kernel_size, kind="max")`

`ImageComponent`s: `(B, channels, H, W) -> (B, out, H_out, W_out)`, PyTorch's image layout,
channels second. A square kernel, the same stride, padding and dilation on both axes.

```python
conv = Conv2D(in_size=3, out_size=8, kernel_size=3, padding=1)
pool = Pool2D(2)
image = torch.randn(4, 3, 32, 32)
assert conv.output_size(32, 32) == (32, 32)
assert pool.output_size(32, 32) == (16, 16)
assert pool(conv(image)).shape == (4, 8, 16, 16)
```

Because the channels are not the last axis, the components that work on the last axis
(`Gate`, `Norm`, `Skip`, `Projection`) see an image's width there. Go to a vector first.

### `GlobalPool2D(kind="mean")`

`(B, channels, H, W) -> (B, channels)`: each channel averaged (or, with `"max"`, maximised) over
the whole image. The step from an image to a vector, as an aggregator is the step from a
sequence to a vector; it belongs to no family.

```python
head = Serial([Conv2D(3, 8, 3, padding=1), GlobalPool2D(), Projection(8, 10)])
assert head(torch.randn(4, 3, 32, 32)).shape == (4, 10)
```

---

## 4.5d Recurrence — `recurrent.py`

### `Recurrent(in_size, hidden_size, kind="gru", n_layers=1)`

A `SequenceComponent`: `(B, seq, in_size) -> (B, seq, hidden_size)`, the state at every step and
nothing else, so it composes in a `Serial`. `kind` is `"rnn"`, `"gru"` or `"lstm"`; `n_layers`
stacks layers, and the output is the top one's. Put `LastAggregator` after it for the state
after the whole sequence.

```python
model = Serial(
    [
        Conv1D(3, 16, kernel_size=5),  # (B, 200, 3) -> (B, 196, 16)
        Pool1D(4),  # -> (B, 49, 16)
        Recurrent(16, 32, kind="lstm"),  # -> (B, 49, 32)
        LastAggregator(),  # -> (B, 32)
        Projection(32, 5),  # -> (B, 5)
    ]
)
assert model(torch.randn(8, 200, 3)).shape == (8, 5)
```

Every step is computed, padding included. For padded chunks, `RNNAggregator` runs its GRU
over the real chunks only.

---

## 4.6 Gating — `gating.py`

The theme of this module is **learned interpolation**: instead of committing to a transform,
learn how much of it to apply. All of them work on the last axis at any rank.

### `Gate(hidden_size)`

```
y = sigmoid(W x) * x
```

Element-wise multiplicative gate: the network learns to suppress dimensions. Any rank: on
`(B, C, D)` it gates every chunk.
Composes nicely *inside* a `Skip`'s transform.

### `Highway(transform, hidden_size)`

```
y = g * T(x) + (1 - g) * x,   g = sigmoid(W · norm(x))
```

A learned interpolation between "transform it" and "pass it through". Where `Skip` always adds
the transform, `Highway` can learn to route around it entirely. See *Highway Networks* in
`references/`. Any rank; the transform must keep the width.

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
> input, so this buys **capacity and specialisation, not compute savings**. Top-k routing is
> `SparseMoE`, below.

### `SparseMoE(experts, hidden_size, out_size, k=1)`

```
y = Σ_{i ∈ topk(W x)} softmax(topk logits)_i · expert_i(x)
```

A **sparse** mixture of experts: the router picks the top-`k` experts per example, each example
is gathered into only those experts' input batches, and the outputs are scattered back weighted
by a softmax over just the `k` chosen logits. An expert an example was not routed to never sees
it, so parameters scale with the number of experts while compute per example scales with `k`.
`k=1` is Switch-style routing (Fedus et al., 2021); `k>1` blends the top-`k` experts as in
Shazeer et al. (2017). `k` must satisfy `1 <= k <= len(experts)`, otherwise `VectorMeshError`.

Same interface as `MoE`: any rank (leading dimensions are flattened into one routing axis and
restored), and experts map `(..., hidden_size)` to `(..., out_size)`. There is no noisy gating and
no load-balancing loss, so nothing keeps the router from concentrating on a few experts.

```python
experts = [NeuralNet([hidden_size, 128], hidden_size) for _ in range(8)]
SparseMoE(experts, hidden_size=hidden_size, out_size=hidden_size, k=2)
```

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

Annotated `"*lead dim"` on both sides, shape-preserving, so it drops straight in front of a
head:

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
`float` so `vectormesh.training.Trainer` can log them.

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
