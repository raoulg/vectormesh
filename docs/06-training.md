# 6. Training

`Trainer` lives in `vectormesh.training`. It depends on nothing but `torch`, `pydantic`, `loguru`
and `tqdm` — no mlflow, ray, or tensorboard import anywhere in the package. Logging to any of
those is opt-in, wired up by you — see [§6.5](#65-reporters-logging-without-a-framework-dependency).

---

## 6.1 The full wiring

```python
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from vectormesh.components import FixedPadding, MaskedMeanAggregator, NeuralNet, Serial
from vectormesh.components.metrics import F1Score
from vectormesh.data import Collate, OneHot
from vectormesh.data.cache import VectorCache
from vectormesh.data.vectorizers import detect_device
from vectormesh.training import Trainer, TrainerSettings

# 1. data
traincache = VectorCache.load(path=trainpath)
validcache = VectorCache.load(path=validpath)

onehot = OneHot(num_classes=32, label_col="labels", target_col="onehot")
train_oh = traincache.dataset.map(onehot)
valid_oh = validcache.dataset.map(onehot)

collate_fn = Collate(
    embedding_col="legal_dutch",
    target_col="onehot",
    padder=FixedPadding(max_chunks=30),
)
trainloader = DataLoader(train_oh, batch_size=32, shuffle=True,  collate_fn=collate_fn)
validloader = DataLoader(valid_oh, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 2. model
hidden_size = traincache.metadata["legal_dutch"]["hidden_size"]
pipeline = Serial([MaskedMeanAggregator(), NeuralNet(hidden_size, 32)])

# 3. training
settings = TrainerSettings(
    epochs=10,
    metrics=[F1Score()],
    logdir=Path("logs").absolute(),
    train_steps=len(trainloader),
    valid_steps=len(validloader),
    earlystop_kwargs=None,  # or a dict -- no default, see §6.4
)

trainer = Trainer(
    model=pipeline,
    settings=settings,
    loss_fn=torch.nn.BCEWithLogitsLoss(),
    optimizer=optim.Adam,
    traindataloader=trainloader,
    validdataloader=validloader,
    scheduler=optim.lr_scheduler.ReduceLROnPlateau,
    device=detect_device(),
)
result = trainer.loop()  # TrainResult(epoch, train_loss, test_loss, metric_dict)
```

`detect_device()` (from `vectormesh.data.vectorizers`) returns `cuda` → `mps` → `cpu` in order of
availability. For a head this small, `cpu` is often competitive — the data is already vectors.

`trainer.loop()` returns a `TrainResult` with the final epoch's `train_loss`, `test_loss` and
`metric_dict`, which is what lets a ray trial call `Trainer` directly and read its result off the
return value; see [§9](09-architecture-search.md).

---

## 6.2 Choosing the loss

| Task shape | Target from `OneHot` | Loss | Metric |
|---|---|---|---|
| **Multi-label** (a document can carry several legal facts) | multi-hot float vector | `BCEWithLogitsLoss` | `F1Score(average="micro")` |
| **Single-label** (one flower species per image) | one-hot float vector | `CrossEntropyLoss` | `Accuracy` (+ `F1Score`) |

Both losses take **logits**, not probabilities — no `Sigmoid`/`Softmax` at the end of your
pipeline. `F1Score` applies its own `sigmoid` internally, consistent with `BCEWithLogitsLoss`.

`CrossEntropyLoss` accepts the one-hot float vector as a soft target, which is why notebook 4 can
use the same `OneHot` + `Collate` machinery as the text notebooks.

---

## 6.3 Metrics against one-hot targets

`Collate` always stacks the target column into `(B, n_classes)`. Both shipped classifiers cope:

- `Accuracy` compares `argmax(yhat)` against `argmax(y)` when shapes match.
- `F1Score` thresholds `sigmoid(yhat)` at 0.5 and computes micro/macro F1 element-wise.

Each metric's `__repr__` is the label used by the reporters (`"F1-micro"`, `"Accuracy"`) — keep
those strings stable so runs stay comparable across experiments.

---

## 6.4 `TrainerSettings` worth knowing

```python
settings = TrainerSettings(
    epochs=150,
    metrics=[F1Score()],
    logdir=Path("logs/MoE_parallel").absolute(),
    train_steps=len(trainloader),
    valid_steps=len(validloader),
    earlystop_kwargs={"save": True, "verbose": True, "patience": 40},
    scheduler_kwargs={"factor": 0.5, "patience": 20},
)
```

- `train_steps` / `valid_steps` — batches per epoch; use `len(loader)` for a full pass.
- `earlystop_kwargs` — **required, no default.** Pass `None` to train the full epoch count, or a
  dict of `EarlyStopping` kwargs. Whether a run stops early changes the number you report,
  non-obviously enough that `TrainerSettings` won't let you leave it unstated. Patience must exceed
  the scheduler's patience, or you stop before the reduced learning rate has a chance to help — the
  scripts use 40 vs 20.
- `scheduler_kwargs` — only used if `Trainer(..., scheduler=...)` is also passed; defaults to
  `ReduceLROnPlateau`'s own `{"factor": 0.1, "patience": 10}` if you don't override it. Worth
  passing explicitly anyway, the same way `earlystop_kwargs` is required — so the value in use is
  visible in your own code, not hidden behind a default you'd have to go look up.
- `optimizer_kwargs` — defaults to `{}`: the optimizer class you pass keeps its own defaults
  (`torch.optim.Adam`: `lr=1e-3`, `weight_decay=0`) unless you override them here.
- `logdir` — pass an **absolute** path; a relative one resolves against the notebook's cwd. This is
  only ever the *parent* of a run's actual directory — see §6.5 for what `Trainer` does with it.
- `TrainerSettings` holds only hyperparameters — where to log to is a `Trainer` constructor
  argument instead (§6.5), because a logging backend is a runtime choice with its own optional
  dependency, not a hyperparameter.

---

## 6.5 Reporters: logging without a framework dependency

`vectormesh.training` imports nothing but `torch`, `pydantic`, `loguru` and `tqdm` — no `mlflow`,
no `ray`, no `tensorboard`. Every epoch, `Trainer.report()` always logs a line via `loguru`; on
top of that it calls every callable in `reporters`, in order:

```python
class Reporter(Protocol):
    def __call__(self, epoch: int, train_loss: float, test_loss: float,
                 metric_dict: dict[str, float]) -> None: ...
```

`Reporter` is a `typing.Protocol`, not a base class — nothing to subclass, nothing to import from
vectormesh. A bare function with that signature already qualifies:

```python
def to_ray(epoch, train_loss, test_loss, metric_dict):
    from ray import tune
    tune.report({"train_loss": train_loss, "test_loss": test_loss, **metric_dict})

trainer = Trainer(..., reporters=[to_ray])
```

Constructing a `Trainer` computes its run directory (`trainer.log_dir`) without creating it on
disk — nothing is written until something actually needs to (currently only
`EarlyStopping.save_checkpoint`, when `earlystop_kwargs={"save": True, ...}`). That means a
reporter that needs a real path — TensorBoard's `SummaryWriter`, for instance — can be built
*after* construction, pointed at the exact directory the run will use, and appended to
`trainer.reporters` (a plain list) before calling `.loop()`:

```python
from torch.utils.tensorboard.writer import SummaryWriter

trainer = Trainer(model=pipeline, settings=settings, ...)
writer = SummaryWriter(log_dir=trainer.log_dir)
trainer.reporters.append(
    lambda epoch, train_loss, test_loss, metric_dict: (
        writer.add_scalar("Loss/train", train_loss, epoch),
        writer.add_scalar("Loss/test", test_loss, epoch),
        *(writer.add_scalar(f"metric/{k}", v, epoch) for k, v in metric_dict.items()),
    )
)
trainer.loop()
writer.close()
```

`scripts/train_moe.py` does exactly this. Copy-paste adapters for mlflow, ray and a plain webhook
— none of them shipped as vectormesh code — are in
[10. Reporters cookbook](10-reporters.md).

---

## 6.6 `Step`: pluggable loss computation

Every epoch, `Trainer` turns a batch into a scalar loss twice — once training (with gradients),
once validating (without). By default that's `loss_fn(model(x), y)`: one forward pass, then
`loss_fn` scores it. That already covers a compound model output — a Gaussian head returning
`(mean, sigma)`, say — as long as `loss_fn` unpacks the tuple itself; nothing special needed.

It stops covering the case where the loss needs the model called more than once to mean anything
— a contrastive loss run on two augmented views, a VAE combining a reconstruction term with a KL
term. For that, pass `step`:

```python
class Step(Protocol):
    def __call__(self, model: nn.Module, x: BatchTensor, y: torch.Tensor) -> torch.Tensor: ...
```

```python
def contrastive_step(model, x1, x2):
    z1, z2 = model(x1), model(x2)
    return nt_xent_loss(z1, z2)

trainer = Trainer(..., step=contrastive_step)
```

`step` replaces the loss computation in **both** `trainbatches()` and `evalbatches()`, so
`test_loss` — and everything downstream of it, early stopping, the LR scheduler, every `Reporter`
— stays coherent no matter which one is in use.

What `step` does **not** make pluggable is `metrics`: `metric_dict[str(m)] = m(y, yhat)` still
assumes a plain `model(x)` produces something worth scoring against `y`. A contrastive pair over
two views usually has no such `yhat` — pass `metrics=[]` in that case. Combining a custom `step`
with non-empty `metrics` costs a second forward pass (metrics call `model(x)` on their own, since
a custom `step` doesn't expose whatever it computed internally), and only makes sense when that
plain forward call is actually meaningful for scoring.

---

## 6.7 The scripts

The `scripts/` folder is the batch counterpart to the notebooks — same pipelines, real dataset
sizes, no subsampling.

| Script | Purpose |
|---|---|
| `build_dataset.py` | instructor-side: JSONL → threshold-filtered train/valid/test splits |
| `create_cache_aktes.py`, `create_cache_imdb.py` | build the initial embedding caches |
| `embed_legal_dutch.py`, `embed_multilegal.py`, `embed_debertav3.py` | cache the same corpus with different encoders (the encoder-comparison experiment) |
| `add_chunked_regex.py` | extend an embedding cache with a **chunk-aligned** regex column |
| `train_moe.py` | MoE over embeddings only |
| `train_moe_parallel.py` | MoE over per-chunk fused embedding + regex |

Run them with `uv run python scripts/<name>.py`.

### The chunk-aligned workflow

`add_chunked_regex.py` is the piece worth reading closely, because it demonstrates metadata as a
contract:

```python
col_meta = metadata["legal_dutch"]
model_tag    = col_meta["model_tag"]
context_size = col_meta["context_size"]
stride       = col_meta.get("stride")          # older caches predate this field
if stride is None:
    stride = context_size // STRIDE_DIVISOR    # documented fallback, with a warning
```

Those three values are then passed straight into `ChunkedRegexVectorizer`, guaranteeing the two
columns chunk identically. The script also filters candidate caches by name (`"regex" not in
p.name`) so it always aligns to the *embedding* cache, never to a previously extended one.

Order of operations:

```bash
uv run python scripts/create_cache_aktes.py    # 1. embeddings  -> artefacts/..._legal_dutch
uv run python scripts/add_chunked_regex.py     # 2. + regex     -> artefacts/..._chunked_regex
uv run python scripts/train_moe_parallel.py    # 3. train
```

---

## 6.8 Reproducibility checklist

Before comparing two runs, confirm they share:

- the same cache (folder name includes the timestamp — note it down);
- the same `max_chunks` — changing it changes how much of each document the model sees;
- the same regex fitting parameters (`min_doc_frequency`, `max_features`) — these change
  `hidden_size` and therefore the model's input width;
- the same split subsampling. The notebooks use `.select(range(1024))` **for demo speed only**;
  results from a 1024-document subset are not comparable to full-dataset results.

---

## 6.9 Tests and tooling

```bash
uv sync                              # install
uv run pytest -m "not integration"   # unit tests, no model downloads
uv run pytest                        # everything, downloads HF models
uv run ruff check src tests          # lint
uv run ty check                      # type check
```

`pytest` is configured (in `pyproject.toml`) to always produce coverage — terminal plus an HTML
report in `htmlcov/`. Ruff has `F722`/`F821` disabled project-wide, because jaxtyping's
string annotations (`Float[Tensor, "batch dim"]`) otherwise read as syntax errors to the linter.

Next: [API reference](07-api-reference.md).
