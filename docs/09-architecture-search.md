# 9. Searching architectures with Ray Tune

The rest of these docs are about *building* a pipeline: picking components, composing them,
getting the shapes right. This chapter is about what to do once you have more than one plausible
pipeline and no principled way to pick between them by staring at the code.

## 9.1 Why search instead of guess

Every component in this library trains on a frozen cache. That means a training run costs
seconds to low minutes, not hours. Once a run is that cheap, hand-tuning stops being a reasonable
default — you can no longer claim you picked a width, a learning rate, or a gating mechanism by
intuition when checking twenty options would have cost a coffee. [Ray Tune](https://docs.ray.io/en/latest/tune/index.html)
is the tool this course uses to hand that search to an algorithm and read what comes back,
including the parts that are disappointing.

This is not about hyperparameter tuning as a ritual. It is about treating "which architecture" as
one more axis you search rather than one you assume.

## 9.2 The cheapest baseline: no training at all

Before spending a single epoch, ask whether the embeddings already separate your classes on their
own. A **nearest-centroid / cosine-similarity classifier** needs no training loop and takes
seconds even on the full cache:

```python
import torch

def cosine_centroid_baseline(train, test, n_classes: int) -> float:
    """Classify by cosine similarity to each class's mean embedding. No gradient step anywhere."""
    embeds = torch.stack([torch.as_tensor(e) for e in train["embed"]])  # (N, dim), already 1D per item
    labels = torch.as_tensor(train["label"])
    centroids = torch.stack([embeds[labels == k].mean(0) for k in range(n_classes)])  # (K, dim)
    centroids = torch.nn.functional.normalize(centroids, dim=-1)

    test_embeds = torch.nn.functional.normalize(
        torch.stack([torch.as_tensor(e) for e in test["embed"]]), dim=-1
    )
    preds = (test_embeds @ centroids.T).argmax(-1)
    return (preds == torch.as_tensor(test["label"])).float().mean().item()
```

(If your items are chunked — `(chunks, dim)` — mean-pool over the chunk axis first, the same
reduction `MaskedMeanAggregator` performs, before computing centroids.)

This number is a **floor**, not a target: if a trained pipeline cannot clear it, the problem is
almost never the architecture — it is more likely the label quality, the encoder choice, or a
data-loading bug. Running this before writing a single `NeuralNet` call catches those problems
before they get mistaken for an architecture result three hours later. It also gives you a second,
independent number to sanity-check the trained baseline in the next section against.

## 9.3 A trained baseline, with a seed spread

A search without a trained baseline is unreadable. If you do not know what the simplest possible
pipeline scores, you cannot tell whether the search's best trial is a real discovery or a
rounding error.

Establish it *before* writing a search space, and establish it with more than one seed:

```python
import numpy as np
import torch

def run(model, train_dl, test_dl) -> float:
    ...  # train + evaluate, return one metric

def seed_spread(make_model, train_dl, test_dl, seeds: int = 5) -> np.ndarray:
    out = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        out.append(run(make_model(), train_dl, test_dl))
    return np.array(out)

baseline = seed_spread(lambda: Serial([MaskedMeanAggregator(), NeuralNet(dim, n_classes)]), tr, te)
print(f"baseline: {baseline.mean():.4f} +/- {baseline.std():.4f}")
```

The `std` is the number that decides what counts as a win later (§9.7). A search result that
beats the mean by less than this spread has not beaten anything — it has produced another sample
from the same distribution.

## 9.4 The search space

One dictionary describes every configuration you are willing to consider. Use `tune.choice` for
categorical axes, `tune.uniform` / `tune.loguniform` for continuous ones:

```python
from ray import tune

param_space = {
    "mechanism": tune.choice(["plain", "gate", "highway", "moe"]),
    "hidden": tune.choice([64, 128, 256]),
    "lr": tune.loguniform(1e-4, 3e-3),
    "n_experts": tune.choice([2, 4, 8]),        # only read when mechanism == "moe"
    "expert_hidden": tune.choice([64, 128]),    # only read when mechanism == "moe"
    "epochs": MAX_EPOCHS,                       # a ceiling, not a fixed count — see §9.5
}
```

Two things worth naming explicitly:

- **Include the axes that could confound the comparison you actually care about.** If the
  question is "does `moe` beat `plain`?", `hidden` and `lr` must be searched *alongside*
  `mechanism` — otherwise a difference you attribute to the mechanism might just be an unlucky
  learning rate.
- **Conditional parameters are a known, accepted inefficiency here.** `n_experts` only matters
  when `mechanism == "moe"`, but most search algorithms (including the `HyperOptSearch` used
  below) do not know that and will still sample it on every trial, including the ones that ignore
  it. This wastes some trials. It is not worth fighting unless the space is large enough that the
  waste dominates — in which case look at Ray Tune's conditional search-space support before
  reaching for a workaround.

## 9.5 The trainable function

A Ray trial runs in its **own process**. Nothing from your notebook's memory is visible inside
it — the trainable function must build everything itself, from the cache load down to the
optimizer:

```python
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from torch import nn
from torch.utils.data import DataLoader

from vectormesh import VectorCache
from vectormesh.components import Gate, Highway, MaskedMeanAggregator, MoE, NeuralNet, Serial
from vectormesh.data import Collate, OneHot

MECHANISMS = {
    "plain": lambda dim, cfg: nn.Identity(),
    "gate": lambda dim, cfg: Gate(dim),
    "highway": lambda dim, cfg: Highway(dim),
    "moe": lambda dim, cfg: MoE(dim, n_experts=cfg["n_experts"], hidden=cfg["expert_hidden"]),
}


def trainable(config: dict) -> None:
    train = VectorCache.load(path=CACHE_PATH, split="train")
    test = VectorCache.load(path=CACHE_PATH, split="test")
    dim, n_classes = train.dim, train.num_classes

    oh = OneHot(num_classes=n_classes, label_col="label", target_col="onehot")
    col = Collate(embedding_col="embed", target_col="onehot")
    tr = DataLoader(train.dataset.map(oh), batch_size=64, shuffle=True, collate_fn=col)
    te = DataLoader(test.dataset.map(oh), batch_size=64, shuffle=False, collate_fn=col)

    mechanism = MECHANISMS[config["mechanism"]](dim, config)
    model = Serial([
        MaskedMeanAggregator(),
        mechanism,
        NeuralNet(dim, config["hidden"], n_classes),
    ])

    torch.manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    loss_fn = nn.CrossEntropyLoss()

    best_acc, patience, bad_epochs = 0.0, 15, 0
    for _ in range(config["epochs"]):  # epochs is a CEILING; the loop below decides when to stop
        model.train()
        for X, y in tr:
            opt.zero_grad()
            loss_fn(model(X), y).backward()
            opt.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in te:
                correct += int((model(X).argmax(1) == y.argmax(1)).sum())
                total += len(y)
        acc = correct / total
        scheduler.step(acc)

        if acc > best_acc:
            best_acc, bad_epochs = acc, 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:  # plateaued — stop spending this trial's compute
            break

    tune.report({"accuracy": best_acc})


tuner = tune.Tuner(
    trainable,
    param_space=param_space,
    tune_config=tune.TuneConfig(
        metric="accuracy", mode="max", search_alg=HyperOptSearch(), num_samples=24,
    ),
    run_config=tune.RunConfig(storage_path=Path("ray_results").resolve(), name="mechanism_search"),
)
results = tuner.fit()
```

The mechanism axis reads directly from `vectormesh.components.gating` — `Skip → Gate → Highway →
MoE` is exactly the ladder of increasingly expressive conditional computation from
[Core concepts §1.6](01-core-concepts.md#16-composition-over-configuration) and
[Teaching path](08-teaching-path.md#notebook-3--3_moeipynb-conditional-computation). Searching
"which mechanism" is a direct instance of that ladder, not a special case.

**On the epoch budget:** set `epochs` in the search space to a generous *ceiling* (large enough
that no config would plausibly still be improving at that point), and let the
`ReduceLROnPlateau` scheduler plus the patience-based break above decide the *actual* stopping
point per trial. A fixed, guessed epoch count silently under-trains some configs and over-trains
others — and that difference then looks like an architecture effect when it is really a training-
budget artefact. `docs/06-training.md §6.4` documents the same `scheduler` +
`earlystop_kwargs` pattern for `mltrainer.Trainer`, including the gotcha that the early-stop
patience must exceed the scheduler's patience, or you stop before a reduced learning rate gets a
chance to help.

## 9.6 Reading a search one axis at a time

The temptation is to read off the winning row and stop. Resist it — a single top row cannot tell
you *why* it won. Instead, look at how the metric varies along each axis separately:

```python
df = results.get_dataframe()
for axis in ("mechanism", "hidden", "lr"):
    print(f"\n{axis}")
    print(df.groupby(f"config/{axis}")["accuracy"].agg(["count", "min", "mean", "max"]))
```

An axis whose groups overlap heavily did not matter for this task — say so, rather than reporting
its "best" value as if it were meaningful. An axis that cleanly separates groups is the actual
finding; a parallel-coordinates plot (`ray.tune`'s own reporting, or `plotly.parallel_coordinates`
on the dataframe) shows all axes at once and makes this easy to see in one picture.

## 9.7 Turning a difference into a claim

A number that beats the baseline mean is not yet a result. Compare it against the trained
baseline's own seed spread (§9.3) — and against the no-training floor from §9.2, which tells you
whether there was ever any real signal to find:

```python
best = df.loc[df["accuracy"].idxmax()]
diff = best["accuracy"] - baseline.mean()
print(f"difference vs baseline mean : {diff:+.4f}")
print(f"in units of baseline sigma  : {diff / baseline.std():+.1f}")
print(f"floor (no training)         : {cosine_floor:.4f}")
```

A difference smaller than one baseline sigma is noise until proven otherwise — rerun the winning
config over several seeds and report a mean and spread of its own before writing anything down
as a conclusion. Also weigh the cost: a mechanism that wins by a fraction of a point but adds
several times the parameters (`sum(p.numel() for p in model.parameters())`) is not obviously
worth adopting, even if the win is real.

## 9.8 Logging discipline: one summary per search

Each Ray trial is its own OS process. If your metric-logging backend uses a local sqlite file
(mlflow's default), concurrent writers from parallel trials will corrupt or block each other. Do
not log from inside `trainable`. Log **once**, after `tuner.fit()` returns, as a single summary
row for the whole search:

```python
import mlflow

mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_PATH}")
mlflow.set_experiment("architecture-search")
with mlflow.start_run(run_name="mechanism_search"):
    mlflow.log_params({"n_trials": len(df)})
    mlflow.log_metric("no_training_floor", cosine_floor)
    mlflow.log_metric("baseline_accuracy", baseline.mean())
    mlflow.log_metric("best_trial_accuracy", float(best["accuracy"]))
    mlflow.log_param("best_mechanism", best["config/mechanism"])
```

This is the same "log everything" habit as the rest of the course, applied at the granularity
that is actually safe: per-search, not per-trial.

## 9.9 A `uv run` gotcha worth knowing

If you launch the notebook or script with `uv run`, Ray propagates that runtime environment to
every worker process it spawns — and each worker then expects a `pyproject.toml` in *its own*
working directory, which usually is not where you started. Turn this off before importing `ray`:

```python
import os
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
import ray  # noqa: E402
```

## 9.10 Where to see this worked end-to-end

This chapter is deliberately independent of any one dataset or notebook, so it stays correct while
courses around it change. For a worked, currently-maintained walkthrough against real caches and
real search spaces, see `notebooks/1_pytorch_intro/04_search.ipynb` and
`notebooks/2_timeseries/06_search.ipynb` in
[MADS-MachineLearning-course](https://github.com/raoulg/MADS-MachineLearning-course) — that
repo is under active refactoring, so if a path there has moved, search it for `ray.tune` or
`HyperOptSearch` rather than trusting the exact filename.

---

Next: nowhere — this is the last chapter. Back to [Teaching path](08-teaching-path.md) for the
notebook-by-notebook map, or [README](README.md) for the full reading order.
