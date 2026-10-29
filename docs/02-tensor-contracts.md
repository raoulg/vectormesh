# 2. Tensor contracts

Every `forward` in VectorMesh declares the shape it accepts and the shape it returns, and that
declaration is **checked at runtime**. This is the single most important thing to understand
before reading any component source.

## 2.1 The two libraries

```python
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

@jaxtyped(typechecker=beartype)
def forward(self, tensors: Float[Tensor, "batch chunks dim"]) -> Float[Tensor, "batch dim"]:
    return tensors.mean(dim=1)
```

- **jaxtyping** gives the shape vocabulary: `Float[Tensor, "batch chunks dim"]` means a
  float tensor of rank 3 whose axes we choose to *call* batch, chunks, dim.
- **beartype** does the actual runtime check on every call.
- `@jaxtyped(...)` binds the named axes **within one call**, so if `batch` appears in both the
  argument and the return type, the two must have the same size. That is what turns a shape
  annotation into a real contract rather than a comment.

## 2.2 Reading the axis vocabulary

| Annotation | Meaning |
|---|---|
| `Float[Tensor, "batch dim"]` | rank 2; any sizes, but must be consistent with other uses of `batch`/`dim` in the same call |
| `Float[Tensor, "batch _ dim"]` | rank 3; `_` is an **anonymous** axis — any size, not tied to anything else (used for the chunk axis when the component doesn't care how many chunks there are) |
| `Float[Tensor, "..."]` | any rank at all (used by `Serial` and `Parallel`, which only route) |
| `Float[Tensor, "*lead dim"]` | rank ≥ 1; `*lead` names the leading axes, so the same name on the return type means *the same leading shape* (`GaussianNoise` is shape-preserving this way) |
| `Float[Tensor, "... {self.hidden_size}"]` | any leading shape, but the **last** axis must equal this instance's `hidden_size` at runtime |
| `Int[Tensor, "batch tokens"]` | integer tensor (token ids, attention masks) |
| `tuple[Float[Tensor, "batch dim"], ...]` | a variable-length tuple of rank-2 tensors — the output of a `Parallel` branch set; beartype checks one element of it per call (see [§2.6](#26-families-the-rung-is-the-class)) |

The `{self.hidden_size}` form is worth pausing on. `NeuralNet(768, 32)` will reject a
`(batch, 384)` input with a clear message *at the boundary*, instead of letting PyTorch raise a
matmul error three layers later.

## 2.3 Why this matters more than usual here

PyTorch's `nn.Linear` acts on the last dimension and silently accepts any leading shape:

```python
linear = nn.Linear(768, 32)
x = torch.randn(16, 30, 768)   # (batch, chunks, dim) — you forgot to aggregate
out = linear(x)                # shape (16, 30, 32). No error. Silently wrong.
```

In a chunked-document setting this failure mode is *constant*: forgetting an aggregator gives
you a model that trains, converges to something, and is quietly modelling the wrong thing. The
annotations turn that into an immediate, loud failure.

## 2.4 How to read the error

```
beartype.roar.BeartypeCallHintParamViolation: Method ...forward() parameter
tensors=tensor([[[...]]]) violates type hint Float[Tensor, 'batch dim'],
as 3D tensor != 2D tensor
```

Decode it in three steps:

1. **Which component?** The method path names the class.
2. **What did it want?** The type hint — here rank 2, `(batch, dim)`.
3. **What did it get?** Rank 3.

Rank 3 where rank 2 was wanted almost always means **a missing aggregator**. Insert a
`MaskedMeanAggregator()` (or `AttentionAggregator`, `RNNAggregator`) before the offending
component. The reverse — rank 2 where rank 3 was wanted — usually means you aggregated too
early, or fed an already-1D vector column (regex, image) into a chunk-aware branch.

A mismatch on the *last* axis size (`{self.hidden_size}`) means the dimension you configured
does not match the embedding dimension in the cache. Read it from the metadata rather than
hardcoding:

```python
hidden_size = cache.metadata[column]["hidden_size"]
```

## 2.5 The tensor-rank ladder

The four shapes from [§1.3](01-core-concepts.md#13-the-tensor-flow-ladder), with the code that
produces and consumes each:

| Rank | Shape | Produced by | Consumed by |
|:---:|---|---|---|
| **1D** | `(D)` — one item | `RegexVectorizer`, `ImageVectorizer` | `torch.stack` in `Collate` → `(B, D)` |
| **2D** | `(C, D)` — one item | `Vectorizer`, `ChunkedRegexVectorizer` | `FixedPadding` / `DynamicPadding` → `(B, C, D)` |
| **2D** | `(B, D)` — a batch | aggregators, `Concatenate2D`, `NeuralNet` | `Gate`, `Highway`, heads |
| **3D** | `(B, C, D)` — a batch | padders, `Concatenate3D`, `Stack2D` | aggregators, `TransformerBlock`, `MoE` |

Note the deliberate collision on rank 2: inside the cache it means *`(C, D)` for one document*;
after collation it means *`(B, D)` for a whole batch*. The `tensordtype` field in the cache
metadata records the **per-item** rank (1 or 2), which is what determines whether you need a
padder or a plain `torch.stack`.

## 2.6 Families: the rung is the class

Every component with a shape contract belongs to a family that fixes the shape it takes and
gives (`vectormesh/components/contracts.py`). A member narrows its family's shape, never widens
it: `Gate` keeps `dim`, `NeuralNet` names its widths.

| Family | Takes → gives | Members |
|---|---|---|
| `VectorComponent` | `(batch, dim)` → `(batch, out)` | `Gate`, `Highway` |
| `SequenceComponent` | `(batch, seq, dim)` → `(batch, seq, out)` | `Attention`, `TransformerBlock` |
| `RankPolymorphicComponent` | `(..., dim)` → `(..., out)` | `NeuralNet`, `Projection`, `MoE`, `SparseMoE`, `GaussianNoise`, `Skip` |
| `BaseAggregator` | `(batch, _, dim)` → `(batch, dim)` | `MeanAggregator`, `MaskedMeanAggregator`, `AttentionAggregator`, `RNNAggregator` |
| `Connector` | a tuple of branches → one tensor | `Concatenate2D`, `Concatenate3D`, `Stack2D` |

`Serial`, `Parallel` and `Lift` belong to no family: their shape is that of the components
they hold.

The rank-polymorphic family is what lets `NeuralNet` and `Projection` be a head on
`(batch, dim)` **and** a per-chunk transform on `(batch, chunks, dim)`, and lets `MoE` blend
experts per chunk on rank-3 input. The vector and sequence families are rank-locked: `Gate`
and `Highway` take `(batch, dim)` only, attention needs a sequence axis.

Vectorizers have families of the same kind, by the rank of one item
(`vectormesh/data/vectorizers.py`):

| Family | One item | Members |
|---|---|---|
| `ItemVectorizer` | `(dim)` | `ImageVectorizer`, `RegexVectorizer` |
| `ChunkedVectorizer` | `(_, dim)` | `Vectorizer`, `PatchImageVectorizer`, `ChunkedRegexVectorizer` |

The family says which rung a vectorizer is on; the private bases `_ImageModelVectorizer` and
`_RegexFeatures` share code between siblings on different rungs, so
`PatchImageVectorizer(_ImageModelVectorizer, ChunkedVectorizer)` has one of each.

### A named axis binds within one call, not inside a tuple or a list

`@jaxtyped` gives one call one set of bindings, so a name used in two arguments must have the
same size in both. Inside a container it does not reach that far: beartype checks **one random
element** of a `tuple[X, ...]` or a `list[X]` per call. The other elements are not checked at
all, so a name cannot bind across them.

```python
@jaxtyped(typechecker=beartype)
def one(tensors: tuple[Float[Tensor, "batch dim"], ...]): ...   # one branch checked

@jaxtyped(typechecker=beartype)
def every(*tensors: Float[Tensor, "batch dim"]): ...            # every branch checked; dim binds
```

That is why `Connector.forward` takes the tuple a `Parallel` returns and hands it on as
`connect(*tensors)`: every branch becomes an argument. In `connect` the names mean what they
say, which makes the choice between a name and `_` the one way these annotations can be wrong
in a direction that rejects valid code:

- `Concatenate2D` uses `"batch _"`: the batch axes must match, the widths may differ, which is
  the case it exists for (a 384-dim embedding next to a 512-dim one). `"batch dim"` would
  reject it.
- `Concatenate3D` uses `"batch chunks _"`: the chunk axes must match, the widths may differ.
- `Stack2D` uses `"batch dim1"`: `torch.stack` requires identical shapes, so there the
  constraint is real.

The `list[...]` a vectorizer's `__call__` returns is sampled the same way: one item is checked
per call.

## 2.7 The one place types are read as data

`VectorCache.get_dtensor` introspects a vectorizer's `__call__` **return annotation** to decide
whether the output column holds 1D or 2D tensors, and picks the HuggingFace `Features` schema
accordingly:

```python
hints = get_type_hints(vectorizer.__call__)     # dict[str, list[Float[Tensor, "_ dim"]]]
...
return len(tensor_type.dim_str.split())          # "_ dim" -> 2
```

So the annotation is not decoration — it is load-bearing. **If you write a new vectorizer, its
`__call__` return type must be annotated accurately**, or the cache will be written with the
wrong on-disk schema.

## 2.8 A note on `Serial` and `Parallel`

`Serial.forward` and `Parallel.forward` are annotated with `TensorInput`, a union of "a tensor"
and "a tuple of tensors" (`vectormesh/types.py`). They are intentionally permissive: the
pipeline containers do not check shapes themselves, they just route. The checking happens in the
*components inside them*, which is why a badly composed pipeline still fails at the exact
component that is wrong rather than at the pipeline boundary.

Next: [The data layer](03-data-layer.md).
