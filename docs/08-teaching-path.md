# 8. Teaching path

The notebooks in `notebooks/` are the course spine. Each one introduces exactly one new idea and
reuses everything before it. This chapter maps each notebook to the concepts it carries, the
code it exercises, and the questions it is designed to provoke.

---

## Notebook 0 — `0_vectorizer.ipynb`: where vectors come from

**Dataset:** IMDB (25k train / 25k test movie reviews, binary sentiment), sampled to 64 rows so
it runs in a minute.

**Concepts introduced**

- A pretrained encoder as a *frozen function* from text to vectors.
- Chunking: a 512-token context window means long documents become `(chunks, dim)` matrices, not
  single vectors.
- The cache as an artefact with metadata — `tensordtype`, `hidden_size`, `context_size`,
  `chunk_sizes`.
- **Extending** a cache: adding regex features to an existing embedding cache without
  recomputing the embeddings.
- Two kinds of representation living side by side: a 384-dim dense embedding (opaque, learned)
  and a 43-dim binary regex vector (transparent, hand-designed).

**Code exercised:** `Vectorizer`, `VectorCache.create`, `RegexVectorizer`,
`build_imdb_review_pattern`, `harmonize_imdb_match`, `print_stats`.

**Questions it plants**

- Why is `chunk_sizes` mostly `1` here but heavy-tailed in the legal data? (Review length.)
- `max_features=200` was requested but 43 features were fitted — where did the other 157 go?
- The regex vector is interpretable and the embedding is not. Which one will actually predict
  better, and does that change if you have 100 training examples instead of 25,000?

---

## Notebook 1 — `1_training.ipynb`: from cache to trained model

**Dataset:** the distributed legal cache (`aktes`, ~15.5k Dutch notarial deeds, 32 classes),
subsampled to 1024 rows for speed.

**Concepts introduced**

- Loading a cache and reading its metadata before writing any model code.
- The chunk-count distribution as an actual data-analysis step — plot it, then choose
  `max_chunks` from it.
- Padding as a deliberate lossy decision (`FixedPadding` truncates!).
- Aggregation 3D → 2D, and why the *masked* mean is the right default under zero padding.
- `Serial` as trivial-when-there-is-one-component, valuable-when-there-are-several.
- `OneHot` + `Collate` + `DataLoader` — the plumbing from dataset rows to `(X, y)` tensors.
- Loss/metric choice for a multi-label problem: `BCEWithLogitsLoss` + `F1Score`.

**Code exercised:** `VectorCache.load`, `FixedPadding`, `MaskedMeanAggregator`, `Serial`,
`NeuralNet`, `OneHot`, `Collate`, `F1Score`, `mltrainer.Trainer`.

**The question the notebook asks explicitly**

> Which type of model architecture can handle `(batch, chunks, dim)` 3D tensors where every
> batch has a different chunk size? And which needs a fixed chunk size?

(Answer sketch: anything that reduces over the chunk axis with a shape-agnostic op — masked
mean, GRU, attention with a mask — tolerates `DynamicPadding`. Anything holding a parameter per
chunk *position* does not.)

**Other questions worth pushing on**

- The notebook subsamples to 1024. Predict what changes at 15,532 — and then check.
- `max_chunks=30` truncates the long tail. What fraction of documents lose content, and does
  raising it to 45 (as the MoE script does) help or just cost compute?
- Both `OneHot` and `Collate` are defined inline in the notebook *and* exist in
  `vectormesh.data`. Why show the code rather than just import it?

---

## Notebook 2 — `2_design.ipynb`: architecture as composition

**Concepts introduced**

- Two input streams at once: `CollateParallel` returns `((X1, X2), y)`.
- `Parallel` branches that do *different* work — one branch aggregates 3D, the other takes 2D
  straight to a `NeuralNet`.
- Connectors: `Concatenate2D` turns a tuple back into a tensor.
- Residual/gating: `Skip` around a transform, `Projection` to match dimensions.

**Code exercised:** `RegexVectorizer` fitted on all 15k texts, cache extension,
`CollateParallel`, `Parallel`, `Concatenate2D`, `Projection`, `Skip`.

**Questions**

- Each branch is squeezed to 32 dimensions before fusion. Is that bottleneck helping or hurting?
- What happens if you swap `Concatenate2D` for `Stack2D` followed by an attention layer — fusion
  by attention rather than by concatenation?
- Does the regex branch contribute anything at all? (Ablate it: train the embedding branch
  alone and compare.)
- `Skip` uses pre-norm. Try post-norm and watch training stability.

---

## Notebook 3 — `3_moe.ipynb`: conditional computation

**Concepts introduced**

- Mixture of experts as the N-way generalisation of the gating family already seen:

  ```
  Gate:     sigmoid(Wx) * x
  Highway:  g * T(x) + (1 - g) * x
  MoE:      Σ_i softmax(Wx)_i * expert_i(x)
  ```

- A router as a learned, input-dependent parameter selector.
- Runs on the **full** dataset — the first notebook that does.

**Code exercised:** `MoE`, `MaskedMeanAggregator`, `Serial`, full-dataset `DataLoader`.

**Questions**

- 4 experts × width 768 vs 1 expert × width 3072: same parameter budget. Which wins?
- Inspect the router's softmax output. Does it specialise, or collapse to uniform?
- The implementation is *dense* — every expert runs on every input. Read the Shazeer paper in
  `references/` and work out what sparse top-k routing would need (top-k selection, noisy
  gating, a load-balancing auxiliary loss) and why it is hard to make work at 15k documents.
- The notebook ends with "obviously, this can be improved — left as an exercise". Combine it
  with notebook 2's parallel fusion; `scripts/train_moe_parallel.py` is one answer.

---

## Notebook 4 — `4_image_vectorizer.ipynb`: the same idea, different modality

**Datasets:** a catalog you switch with one variable — `dog_food` (2 classes, trivial),
`rock_paper_scissors` (3), `flowers` (102, genuinely hard), `eurosat` (10, 27k satellite tiles).

**Concepts introduced**

- The pipeline is **modality-agnostic**: images produce `(dim,)` vectors, so they take the same
  `Collate` + `Serial` path as regex features, with `padder=torch.stack` and no aggregator.
- `DatasetSchema.infer` — switching datasets requires changing one string, because column names
  are detected.
- Encoder choice as an explicit experiment: MobileNetV2 (3.5M) → DINOv2-large (304M), where model
  size only affects the **one-time** caching cost.
- Reading a HuggingFace `LOAD REPORT`: `UNEXPECTED` classifier weights are the *desired* outcome
  when using a checkpoint as an embedder; `MISSING` would be the alarming one.
- `remove_columns=["image"]` — dropping raw pixels so the cache is small enough to distribute.
- Feature-space augmentation (`GaussianNoise`) as the cache-compatible answer to pixel-space
  augmentation, with an explicit train/eval determinism check.

**Code exercised:** `ImageVectorizer`, `DatasetSchema`, `VectorCache.create(remove_columns=…)`,
`GaussianNoise`, `Collate(padder=torch.stack)`, `Accuracy`, `CrossEntropyLoss`.

**Built-in exercises**

1. Swap the embedding model (`mobilenet_v2` → `dinov2_small`). Better accuracy on the same head?
   How much slower is caching? Use `flowers` — `dog_food` saturates too fast to show anything.
2. Tune `sigma` over `0.0 / 0.05 / 0.2 / 0.5`. Where does noise help and where does it hurt?
   Shrink the training set to 200 images — does noise matter *more* with less data?
3. Switch datasets via `choice`. Predict the difficulty ordering before running.

---

## Cross-cutting threads

Four questions run through the whole course, and are worth naming out loud:

1. **Where does information get destroyed?** Truncation in `FixedPadding`, mean-pooling over
   tokens then over chunks, the 32-dim bottleneck before fusion. Each is a defensible default
   *and* an ablation waiting to happen.

2. **What is learned versus what is designed?** The encoder is frozen (learned elsewhere), the
   regexes are designed by hand, the head is learned here. Moving a decision across that line —
   e.g. `MeanAggregator` (designed) → `AttentionAggregator` (learned) — is the core move of the
   course.

3. **When is a gate worth its parameters?** `Skip` → `Gate` → `Highway` → `MoE` is a ladder of
   increasingly expressive conditional computation. Each rung costs parameters and training
   stability. Measure, don't assume.

4. **Does the type checker teach you something?** A `BeartypeCallHintParamViolation` is not
   noise — it is the framework telling you which of the four boxes in
   [§1.3](01-core-concepts.md#13-the-tensor-flow-ladder) you are actually standing on.

---

## A suggested assignment arc

| Stage | Task |
|---|---|
| 1 | Reproduce the notebook-1 baseline on the **full** dataset. Record F1. |
| 2 | Ablate the aggregator (mean / masked-mean / attention / RNN). One variable changed per run. |
| 3 | Add the regex stream (notebook 2). Does interpretable structure beat more capacity? |
| 4 | Pick one gating mechanism and justify it from the paper in `references/`. |
| 5 | Build something the notebooks do not contain — chunk-level fusion, `Stack2D` + attention, sparse MoE — and defend the design with an ablation, not a story. |

Throughout: change one thing per run, log to TensorBoard, and quote the cache folder name so the
run is reproducible.
