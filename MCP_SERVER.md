# VectorMesh course as an MCP server

This repo ships a small [MCP](https://modelcontextprotocol.io) server,
[`vectormesh_mcp.py`](vectormesh_mcp.py), that turns the course docs into two coaching methods for
a coding assistant (Claude Code, Cursor, Codex, Claude Desktop, …) — one for **designing** a deep
learning architecture for a problem from the library's own principles, and one for **searching**
an architecture with Ray Tune once there's more than one plausible design.

the main idea here is to learn how to work *with* an assistant while building a deep learning architecture, grounded in the vectormesh library's ideas
(the tensor-flow ladder, composition over configuration, is a gate worth its parameters) instead of generic advice.

## How to use it

- **Designing for a problem:** in Claude Code, run `/vectormesh:design` (optionally describe your
  problem). In any client, just say **"Coach me from my problem to a deep learning model using
  the vectormesh server: I want to predict … from …."** Expect a conversation, not an answer: your assistant asks which lesson you are in, then guides you through the stages of design, asking you questions one stage at a time and waiting for your actual answers before moving on.
- **Searching an architecture:** `/vectormesh:search`, or **"Help me set up a Ray Tune search over
  my architecture using the vectormesh server."**
- **Looking something up:** "Using the vectormesh server, get the chapter on components." /
  "Search the vectormesh docs for masked mean." / "What does vectormesh say about chunk alignment?"

The design and search prompts are coaching sessions, not code generators — they're built to ask you what you think before offering an answer, and code comes after the last stage. Using your assistant to answer all eight design stages itself in one go defeats the point; ask it to go back to stage one and actually wait for your answers.

## What it exposes

**Tools**

| Tool | What it does |
| --- | --- |
| `vectormesh_list_concepts` | List all 10 course chapters, in reading order. |
| `vectormesh_get_concept` | Fetch the full markdown of one chapter. |
| `vectormesh_search` | Keyword search across every chapter. |
| `vectormesh_design_checklist` | An eight-stage, table-driven design conversation — see below. |
| `vectormesh_search_budget` | How many search trials fit in a time budget, after the baseline and the winner's rerun. |
| `vectormesh_search_method` | The method for setting up a Ray Tune search, in the right order. |

**`vectormesh_design_checklist` is the centerpiece**, and it is deliberately not a one-shot
document dump. It walks a student from the problem to a shared result through eight stages **in
order**, one at a time, and only advances once the assistant reports back what the student
actually said:

1. **problem** — what is predicted, from what, for whom? Is deep learning needed at all, or does a simple algorithm do it? Which metrics, and what is the acceptance bound: the lowest score that is good enough to use?
2. **tensors** — what is the obvious tensor for this data, its rank, what each axis means, and which preprocessor produces it?
3. **pretrained** — is there a pretrained or foundation model for this modality, can this hardware run it, and does a simpler one already clear the bound? The simplest model that meets the acceptance bound wins.
4. **context** — what extra signal could be added before touching architecture (other sensors, domain knowledge, regex features, a second encoder, seasonality or Fourier features for sequential data)? Does that make it multimodal, and where do the streams meet?
5. **architecture** — the baseline first, then up in complexity: `Serial`/`Parallel`, residual, gated, the fusion point, a range for the model width. Only architectures of the student's lesson or earlier are offered; a later one is named as "comes in lesson N".
6. **search_space** — which hyperparameters and ranges are worth searching, in what budget, and many short trials or fewer long ones?
7. **search_practices** — how will a search result be trusted, not just read (no-training floor, baseline with seed spread, stopping rule, the winner rerun over seeds, effect size, logging discipline)?
8. **results** — which mlflow views answer the hypothesis, does the result clear the bound, and how are the runs shared, named and tagged so others can see who made them and what was explored?

The tool takes an optional `lesson` (1 to 7); if you use the server outside the course, pass lesson 7. Every stage offers only the material of that lesson or earlier. Until the lesson is known, every response asks the assistant to ask for it, the tables are written for lesson 1, and stage 5 returns only that request. Once given, the lesson is kept with the progress; a different lesson on a later call replaces it, and the response says so. `reset=true` clears the progress to start a new conversation. Later stages quote the student's own metric, acceptance bound and no-training floor from stage 1 back to them.

After stage 1 the assistant gives the design a home, `design-log.md` next to your work, and from stage 2 on every stage reminds it to append your answers there. Progress lives in the server's memory, one per client session, so a restarted server starts empty; the log is how to pick up.

Each stage returns a markdown table — question / why it matters / what a good answer names — plus an instruction to the assistant: ask the student, wait for a real answer, don't answer on their behalf, write no code yet, and don't fetch the next stage until this one is actually recorded. The closing summary comes only once every stage has an answer.

**`vectormesh_search_budget`** takes `seconds_per_trial`, `budget_minutes`, and optionally
`parallel_trials` (default 1) and `seeds_for_rerun` (default 3). It reserves the baseline over
seeds and the winner's rerun over as many seeds (`seeds_for_rerun` counts each), says how many trials are left for the search, and puts
the trade — a wide scan of short trials against fewer long ones — to the student as a question.

**Prompts**

- `design` — start the eight-stage design conversation. Takes an optional `idea` (a short
  description; defaults to a generic framing if you haven't narrowed it down yet).
- `search` — coach me through setting up a Ray Tune search. Takes an optional `idea`.

**Resources**

- `vectormesh://concept/{name}` — each chapter's raw markdown, addressable by number (`9`),
  filename (`09-architecture-search.md`), stem, or title.

The content comes from [`docs/`](docs); no build step, no database, no embeddings. The
[Architecture search](docs/09-architecture-search.md) chapter is the Ray Tune companion: it
captures the durable principles (a no-training cosine-similarity floor, a trained baseline with a
seed spread, search-space design, an epoch ceiling plus plateau/early-stop instead of a guessed
epoch count, reading results one axis at a time, log-once discipline) independent of any one
dataset or notebook, so it stays correct as courses built on top of it change.

## Requirements

- [`uv`](https://docs.astral.sh/uv/)
- Python 3.12+

## Install

`uv` runs the server straight from a URL: it downloads the single script, provisions Python and
the `fastmcp` package from the script's own [PEP 723](https://peps.python.org/pep-0723/) metadata,
and the server fetches its docs from the repo over HTTP.

You pin a version with `VECTORMESH_REF` — a git ref (branch, tag, or commit). That one variable
selects both the script (in the URL) and the docs it fetches (inside the script), so a whole
cohort runs exactly the same thing and the two can't drift apart.

Use **`stable`**. It is a tag that moves to the newest release, so a config that names it never
goes out of date, and nothing on `main` reaches you before it has been released. The
[releases](https://github.com/raoulg/vectormesh/tags) also carry a versioned tag `vX.Y.Z` per
release for anyone who needs to freeze one: pick the release whose notes mention the coach you
want. You can also point `VECTORMESH_REF` to a branch or
work from a local clone (see below).

### Claude Code

```bash
claude mcp add vectormesh -s user -e VECTORMESH_REF=stable -- \
  sh -c 'uv run --no-project https://raw.githubusercontent.com/raoulg/vectormesh/$VECTORMESH_REF/vectormesh_mcp.py'
```

### Cursor

Add to `.cursor/mcp.json` (per-project) or `~/.cursor/mcp.json` (global):

```json
{
  "mcpServers": {
    "vectormesh": {
      "command": "sh",
      "args": ["-c", "uv run --no-project https://raw.githubusercontent.com/raoulg/vectormesh/$VECTORMESH_REF/vectormesh_mcp.py"],
      "env": { "VECTORMESH_REF": "stable" }
    }
  }
}
```

### Codex

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.vectormesh]
command = "sh"
args = ["-c", "uv run --no-project https://raw.githubusercontent.com/raoulg/vectormesh/$VECTORMESH_REF/vectormesh_mcp.py"]
env = { VECTORMESH_REF = "stable" }
```

### Claude Desktop

Add to `claude_desktop_config.json` (Settings → Developer → Edit Config), then restart the app:

```json
{
  "mcpServers": {
    "vectormesh": {
      "command": "sh",
      "args": ["-c", "uv run --no-project https://raw.githubusercontent.com/raoulg/vectormesh/$VECTORMESH_REF/vectormesh_mcp.py"],
      "env": { "VECTORMESH_REF": "stable" }
    }
  }
}
```

(The `sh` wrapper is for macOS/Linux; on Windows, drop `sh -c` and put the ref directly in the URL
instead, keeping the `VECTORMESH_REF` env value in sync.)

## Working from a local clone instead

If a student has already cloned `vectormesh` and run `uv sync`, point the same command at the
local file instead of the URL — the server then reads `docs/` straight off disk, so any edits show
up without a restart:

```bash
claude mcp add vectormesh -s user -- uv run --no-project /path/to/vectormesh/vectormesh_mcp.py
```

## Tracking the bleeding edge instead

Set `VECTORMESH_REF=main` to always run whatever is on `main`, including chapters or checklist
changes not in a release yet. Fine for your own use; for a cohort of students, `stable` (or a
versioned tag) keeps everyone on released docs during an assignment.

## Update to a new version

With `VECTORMESH_REF=stable`, restart your assistant: the server is fetched fresh on every
launch, and the tag already points at the newest release.

If you froze a versioned tag instead, pick the release whose notes mention the coach you want
from the [releases](https://github.com/raoulg/vectormesh/tags) and change the `VECTORMESH_REF` value in
your config. In Claude Code, re-running `add` errors if the server already exists, so remove
first:

```bash
claude mcp remove vectormesh -s user
claude mcp add vectormesh -s user -e VECTORMESH_REF=vX.Y.Z -- \
  sh -c 'uv run --no-project https://raw.githubusercontent.com/raoulg/vectormesh/$VECTORMESH_REF/vectormesh_mcp.py'
```

**Cursor / Codex / Claude Desktop** — edit the `VECTORMESH_REF` value in the config, then restart
your assistant so it relaunches the server.

## Releasing (maintainer)

A release is a versioned tag plus moving `stable` onto it, alongside the PyPI publish:

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
git tag -fa stable -m "stable: vX.Y.Z" "vX.Y.Z^{commit}"
git push origin vX.Y.Z
git push -f origin refs/tags/stable
```

Everyone on `stable` gets it the next time their assistant starts.
