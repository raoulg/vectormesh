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
| `Float[Tensor, "..."]` | any rank at all (used by shape-preserving components like `GaussianNoise` and `Skip`) |
| `Float[Tensor, "... {self.hidden_size}"]` | any leading shape, but the **last** axis must equal this instance's `hidden_size` at runtime |
| `Int[Tensor, "batch tokens"]` | integer tensor (token ids, attention masks) |
| `tuple[Float[Tensor, "batch dim"], ...]` | a variable-length tuple of rank-2 tensors — the output of a `Parallel` branch set |

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

## 2.6 Rank-polymorphic components

Some components are annotated `"... {self.hidden_size}"` and therefore work at *both* rank 2 and
rank 3, acting position-wise on the last axis:

- `NeuralNet`, `Projection` — usable as a head on `(batch, dim)` **or** as a per-chunk
  transform on `(batch, chunks, dim)`
- `MoE` — routes per position, so it blends experts per chunk on rank-3 input
- `GaussianNoise`, `Skip` — annotated `"..."`, shape-preserving at any rank

Others are deliberately rank-locked:

- `Gate`, `Highway` — `(batch, dim)` only
- `Concatenate2D`, `Stack2D` — rank-2 branches only; `Concatenate3D` for rank-3

### A named axis *binds*, including across a variadic tuple

Worth internalising, because it is the one way these annotations can be wrong in a
direction that rejects valid code rather than accepting invalid code:

```python
tensors: tuple[Float[Tensor, "batch dim"], ...]   # every branch must have the SAME dim
tensors: tuple[Float[Tensor, "batch _"], ...]     # branches may differ
```

`dim` is a name, so jaxtyping binds it on the first element and requires every later one to
match. `_` is anonymous and binds nothing. `Concatenate2D` shipped with the first form
until 0.5.0 and therefore refused to concatenate a 384-dim embedding with a 512-dim one —
the exact operation it exists to perform. `Concatenate3D` uses `_` on its feature axis but
a named `chunks`, which is right: the widths may differ, the chunk axes may not.

`Stack2D` keeps the named `dim1`, and that is also right — `torch.stack` genuinely requires
identical shapes, so there the constraint is real rather than accidental.
- aggregators — rank 3 in, rank 2 out, by definition
- `Attention`, `TransformerBlock` — rank 3 only (they need a sequence axis)

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
