# VectorMesh course as an MCP server

This repo ships a small [MCP](https://modelcontextprotocol.io) server,
[`vectormesh_mcp.py`](vectormesh_mcp.py), that turns the course docs into two coaching methods for
a coding assistant (Claude Code, Cursor, Codex, Claude Desktop, …) — one for **designing** a new
component or architecture from the library's own principles, and one for **searching** an
architecture with Ray Tune once there's more than one plausible design.

This is deliberately different from a code-review server. There is nothing to review yet — the
point is to work *with* an assistant while building something, grounded in *this* library's ideas
(the tensor-flow ladder, composition over configuration, is a gate worth its parameters) instead
of generic advice.

## What it exposes

**Tools**

| Tool | What it does |
| --- | --- |
| `vectormesh_list_concepts` | List all 9 course chapters, in reading order. |
| `vectormesh_get_concept` | Fetch the full markdown of one chapter. |
| `vectormesh_search` | Keyword search across every chapter. |
| `vectormesh_design_checklist` | A five-stage, table-driven design conversation — see below. |
| `vectormesh_search_method` | The method for setting up a Ray Tune search, in the right order. |

**`vectormesh_design_checklist` is the centerpiece**, and it is deliberately not a one-shot
document dump. It walks a student through five stages **in order**, one at a time, and only
advances once the assistant reports back what the student actually said:

1. **tensors** — what shape is the data actually in, right now?
2. **enrichments** — what extra signal could be added before touching architecture (regex
   features, a second encoder, hand-built signal-processing features for sequential data)?
3. **flows** — given 1 and 2, which compositions (`Serial`/`Parallel`, gating, fusion point) are
   even on the table?
4. **search_space** — of the open questions from 1-3, which are worth turning into a Ray Tune
   search space?
5. **search_practices** — how will a search result be trusted, not just read (no-training
   floor, seed spread, epoch budget, effect size, logging discipline)?

Each stage returns a markdown table — question / why it matters / what a good answer names —
plus an instruction to the assistant: ask the student, wait for a real answer, don't answer on
their behalf, and don't fetch the next stage until this one is actually recorded. Progress is
held in memory for the current session only.

**Prompts**

- `design` — start the five-stage design conversation. Takes an optional `idea` (a short
  description; defaults to a generic framing if you haven't narrowed it down yet).
- `search` — coach me through setting up a Ray Tune search. Takes an optional `idea`.

**Resources**

- `vectormesh://concept/{name}` — each chapter's raw markdown, addressable by number (`9`),
  filename (`09-architecture-search.md`), stem, or title.

The content comes from [`docs/`](docs); no build step, no database, no embeddings. The new
[Architecture search](docs/09-architecture-search.md) chapter is the Ray Tune companion this
library's own docs were missing — it captures the durable principles (a no-training
cosine-similarity floor, a trained baseline with a seed spread, search-space design, an epoch
ceiling plus plateau/early-stop instead of a guessed epoch count, reading results one axis at a
time, log-once discipline) independent of any one dataset or notebook, so it stays correct as
courses built on top of it change.

## Requirements

- [`uv`](https://docs.astral.sh/uv/)
- Python 3.10+

The script pins its own dependency to `mcp[cli]>=1.2.0,<2.0.0` in its PEP 723 header — `mcp` 2.0
removed `mcp.server.fastmcp` (the API this server is built on) with no deprecation window. If you
touch that header, keep the upper bound or the zero-clone install will silently break the next
time someone runs it fresh.

## Install

`uv` runs the server straight from a URL: it downloads the single script, provisions Python and
the `mcp` package from the script's own [PEP 723](https://peps.python.org/pep-0723/) metadata, and
the server fetches its docs from the repo over HTTP.

You pin a version with `VECTORMESH_REF` — a git ref (branch, tag, or commit). That one variable
selects both the script (in the URL) and the docs it fetches (inside the script), so a whole
cohort runs exactly the same thing and the two can't drift apart. The current release is
**`v0.8.1`** — pin to it so the whole cohort runs the exact same docs; add it with one of the
following.

### Claude Code

```bash
claude mcp add vectormesh -e VECTORMESH_REF=v0.8.1 -- \
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
      "env": { "VECTORMESH_REF": "v0.8.1" }
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
env = { VECTORMESH_REF = "v0.8.1" }
```

### Claude Desktop

Add to `claude_desktop_config.json` (Settings → Developer → Edit Config), then restart the app:

```json
{
  "mcpServers": {
    "vectormesh": {
      "command": "sh",
      "args": ["-c", "uv run --no-project https://raw.githubusercontent.com/raoulg/vectormesh/$VECTORMESH_REF/vectormesh_mcp.py"],
      "env": { "VECTORMESH_REF": "v0.8.1" }
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
claude mcp add vectormesh -- uv run --no-project /path/to/vectormesh/vectormesh_mcp.py
```

## Tracking the bleeding edge instead

Set `VECTORMESH_REF=main` (in place of `v0.8.1` above) to always run whatever is on `main`,
including chapters or checklist changes not in a release yet. Fine for your own use; for a cohort
of students, pin to a tag instead so everyone runs the exact same docs during an assignment.

## Update to a new version

Pick the newest tag from the [releases](https://github.com/raoulg/vectormesh/tags), then point
`VECTORMESH_REF` at it. Because the ref lives in the URL, switching it fetches the new script
automatically — no cache clearing needed.

**Claude Code** — re-running `add` errors if the server already exists, so remove first, then add
with the new tag (change `v0.8.1` to the target):

```bash
claude mcp remove vectormesh -s user
claude mcp add vectormesh -s user -e VECTORMESH_REF=v0.8.1 -- \
  sh -c 'uv run --no-project https://raw.githubusercontent.com/raoulg/vectormesh/$VECTORMESH_REF/vectormesh_mcp.py'
```

**Cursor / Codex / Claude Desktop** — edit the `VECTORMESH_REF` value in the config, then restart
your assistant so it relaunches the server.

## Use it

- **Designing something new:** in Claude Code, run `/vectormesh:design` (optionally describe your
  idea). In any client, just say **"Help me design a new vectormesh component using the
  vectormesh server."** Expect a conversation, not an answer: your assistant will ask you the
  tensors/enrichments/flows/search-space/search-practices questions one stage at a time and wait
  for your actual answers before moving on.
- **Searching an architecture:** `/vectormesh:search`, or **"Help me set up a Ray Tune search over
  my architecture using the vectormesh server."**
- **Looking something up:** "Using the vectormesh server, get the chapter on components." /
  "Search the vectormesh docs for masked mean." / "What does vectormesh say about chunk
  alignment?"

The design and search prompts are coaching sessions, not code generators — they're built to ask
you what you think before offering an answer, and to sketch code only when you ask for it. If
your assistant tries to answer all five design stages itself in one go, say so — that defeats the
point; ask it to go back to stage one and actually wait for your answers.
