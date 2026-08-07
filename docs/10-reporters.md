# 10. Reporters cookbook

`vectormesh.training` depends on none of the tools in this chapter. Every adapter below is a
handful of lines you paste into your own script — never something you `pip install vectormesh`
and get for free, and never something a missing package can break an unrelated import over. See
[§6.5](06-training.md#65-reporters-logging-without-a-framework-dependency) for the `Reporter`
contract these all satisfy:

```python
def report(epoch: int, train_loss: float, test_loss: float, metric_dict: dict[str, float]) -> None: ...
```

Pass one or more as `Trainer(..., reporters=[...])`, or append to `trainer.reporters` after
construction if the adapter needs `trainer.log_dir` (built, but not created on disk, the moment
`Trainer.__init__` runs — see §6.5).

---

## 10.1 mlflow

```python
import mlflow

mlflow.set_experiment("my-experiment")
run = mlflow.start_run()

def to_mlflow(epoch, train_loss, test_loss, metric_dict):
    mlflow.log_metric("Loss/train", train_loss, step=epoch)
    mlflow.log_metric("Loss/test", test_loss, step=epoch)
    for name, value in metric_dict.items():
        mlflow.log_metric(f"metric/{name}", value, step=epoch)

trainer = Trainer(..., reporters=[to_mlflow])
trainer.loop()
mlflow.end_run()
```

If you also want the learning rate logged, close over the optimizer:

```python
def to_mlflow(epoch, train_loss, test_loss, metric_dict):
    mlflow.log_metric("Loss/train", train_loss, step=epoch)
    mlflow.log_metric("Loss/test", test_loss, step=epoch)
    for name, value in metric_dict.items():
        mlflow.log_metric(f"metric/{name}", value, step=epoch)
    mlflow.log_metric("learning_rate", trainer.optimizer.param_groups[0]["lr"], step=epoch)
```

**Inside a ray search**, don't do this per epoch — a ray trial runs in its own process, and
mlflow's sqlite backend does not want concurrent writers from several trials at once. Log one
summary row per *search*, after `tuner.fit()` returns, the way
[§9.8](09-architecture-search.md) does it — not one run per trial.

---

## 10.2 ray tune

```python
def to_ray(epoch, train_loss, test_loss, metric_dict):
    from ray import tune
    tune.report({"train_loss": train_loss, "test_loss": test_loss, **metric_dict})

trainer = Trainer(..., reporters=[to_ray])
trainer.loop()
```

Note this is `ray.tune.report`, not `ray.train.report` — `ray.train` is the distributed-training
library, `ray.tune` is the hypertuner. Calling `ray.train.report` from inside a tune trainable
does not report anything; the trial then dies during process teardown, and ray usually surfaces
that as an opaque "worker crashed unexpectedly (SYSTEM_ERROR)" rather than a clear error pointing
at the mix-up.

Per-epoch reporting like this matters when a scheduler (ASHA, `PopulationBasedTraining`) prunes
trials on intermediate values. If you only need the final score — the common case, and what
[§9](09-architecture-search.md) uses — skip the reporter entirely and read `trainer.loop()`'s
return value instead:

```python
def trainable(config):
    trainer = Trainer(..., progress=False)  # no reporters needed
    result = trainer.loop()
    tune.report({"test_loss": result.test_loss, **result.metric_dict})
```

---

## 10.3 TensorBoard

Unlike the two adapters above, `SummaryWriter` is stateful — it needs to be built once, pointed at
a real directory, and closed at the end. Build it against `trainer.log_dir` *after* constructing
`Trainer`, since that path is already computed (just not created on disk yet):

```python
from torch.utils.tensorboard.writer import SummaryWriter

trainer = Trainer(model=pipeline, settings=settings, ...)
writer = SummaryWriter(log_dir=trainer.log_dir)

def to_tensorboard(epoch, train_loss, test_loss, metric_dict):
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/test", test_loss, epoch)
    for name, value in metric_dict.items():
        writer.add_scalar(f"metric/{name}", value, epoch)

trainer.reporters.append(to_tensorboard)
trainer.loop()
writer.close()
```

`scripts/train_moe.py` uses exactly this. Watch curves with:

```bash
uv run tensorboard --logdir logs
```

---

## 10.4 A plain webhook / Discord / Slack

No SDK needed for a simple POST:

```python
import requests

def to_webhook(epoch, train_loss, test_loss, metric_dict):
    if epoch % 10 == 0:  # don't spam a chat channel every epoch
        requests.post(WEBHOOK_URL, json={
            "content": f"epoch {epoch}: train {train_loss:.4f} test {test_loss:.4f}"
        })

trainer = Trainer(..., reporters=[to_webhook])
```

---

## 10.5 Composing more than one

`reporters` is just a list — pass as many as you want, they run in order every epoch:

```python
trainer = Trainer(..., reporters=[to_mlflow, to_tensorboard])
```

Or write one function that fans out to several backends yourself, if you'd rather keep the
`Trainer` call site to a single name.

---

## 10.6 Writing your own

Nothing above is special — a `Reporter` is any callable matching the signature in
[§6.5](06-training.md#65-reporters-logging-without-a-framework-dependency). Print to stdout, append
to a CSV, push a custom metric to a Prometheus pushgateway, whatever your setup needs:

```python
import csv

def to_csv(path):
    def report(epoch, train_loss, test_loss, metric_dict):
        # Trainer never creates log_dir for you (§6.5) -- a reporter that writes
        # files is responsible for its own directory, same as EarlyStopping is.
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, test_loss, *metric_dict.values()])
    return report

trainer = Trainer(...)  # reporters=() default; log_dir is computed, not yet created on disk
trainer.reporters.append(to_csv(trainer.log_dir / "history.csv"))
```

(Note the closure factory pattern (`to_csv(path)` returning `report`) — useful whenever a reporter
needs its own setup argument, same shape as `F1Score(average="macro")` configuring a metric.)
